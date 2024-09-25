# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import json
import math
import os
import sys
import time
from collections import OrderedDict
from contextlib import nullcontext
from datetime import datetime, timedelta
from functools import partial

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from accelerate.utils import set_module_tensor_to_device
from mmengine import mkdir_or_exist
from mmengine.dist import infer_launcher, init_dist
from mmengine.runner import set_random_seed
from mmengine.utils import get_git_hash
from mmengine.utils.dl_utils import collect_env
from peft import LoraConfig, get_peft_model
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import \
    apply_activation_checkpointing
from torch.distributed.checkpoint.state_dict import (StateDictOptions,
                                                     get_model_state_dict,
                                                     get_state_dict)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.api import CPUOffload, ShardingStrategy
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.wrap import _or_policy
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import ConcatDataset, DataLoader
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from transformers.utils.import_utils import (is_flash_attn_2_available,
                                             is_torch_sdpa_available)

from xtuner._lite import AutoTokenizer, get_logger
from xtuner._lite.accelerate import (LORA_TARGET_MAP, dispatch_modules,
                                     packed_sequence)
from xtuner._lite.chat import CHAT_TEMPLATE_MAP
from xtuner._lite.datasets import (OPENAI_FORMAT_MAP, SoftPackerForText,
                                   TextCollator, TextOnlineTokenizeDataset,
                                   TextTokenizedDataset, TextTokenizeFunction)
from xtuner._lite.datasets.load import (LOAD_FN_MAP, load_datasets,
                                        load_from_cache)
from xtuner._lite.parallel import (LengthGroupedSampler, ParallelSampler,
                                   get_dp_mesh, get_dp_world_size,
                                   get_sp_group, get_sp_mesh,
                                   get_sp_world_size,
                                   reduce_sequence_parallel_loss,
                                   setup_parallel, split_for_sequence_parallel)
from xtuner._lite.parallel.fsdp import (RECOMPUTE_MODULES, LoadWoInit,
                                        all_required_grad_wrap_policy,
                                        checkpoint_check_fn, dp_lazy_init,
                                        dp_sp_lazy_init,
                                        layer_auto_wrap_policy)

logger = get_logger()

SUPPORT_DATA_FORMATS = OPENAI_FORMAT_MAP.keys()


def log_format(rank, debug=False):

    formatter = f'[XTuner][RANK {rank}]'
    formatter += '[{time:YYYY-MM-DD HH:mm:ss}][<level>{level}</level>]'

    if debug:
        formatter += '[<cyan>{name}</cyan>:'
        formatter += '<cyan>{function}</cyan>:'
        formatter += '<cyan>{line}</cyan>]'

    formatter += ' <level>{message}</level>'
    return formatter


def parse_args():
    parser = argparse.ArgumentParser(description='Train LLM')

    model_args = parser.add_argument_group('model', 'Model Related Settings')
    model_args.add_argument('--llm', help='repo id or local path of the model')
    model_args.add_argument(
        '-t',
        '--tokenizer',
        help=('repo id or local path of the tokenizer. '
              'Defaults to the same as `model`'))
    model_args.add_argument(
        '--chat-template',
        choices=CHAT_TEMPLATE_MAP.keys(),
        help=('repo id or local path of the tokenizer. '
              'Defaults to the same as `model`'))
    model_args.add_argument(
        '--dtype',
        default='auto',
        choices=['fp16', 'bf16', 'auto'],
        help=("the dtype of the model forward. When set to 'auto', it will "
              'automatically determine whether bf16 is available, '
              'prioritizing the use of bf16.'))

    model_args.add_argument(
        '--shard-strategy',
        default='full',
        choices=['full', 'hybrid'],
        help=('The sharding strategy to be used for distributed training.'))

    data_args = parser.add_argument_group('data', 'Dataset Related Settings')
    data_args.add_argument(
        '--datasets',
        nargs='*',
        help=('repo id or local path or dir of the datasets. For repo ids, '
              'the `dset-sources` needs to be appropriately set to '
              '`modelscope` or `huggingface`. For local dir, all json and '
              'jsonl files will be loaded by default. The type of loaded '
              'files can be controlled by setting `dset-file-type`'))
    data_args.add_argument(
        '--dset-file-types',
        nargs='*',
        default=LOAD_FN_MAP.keys(),
        choices=LOAD_FN_MAP.keys(),
        help='the file type that needs to be loaded')
    data_args.add_argument(
        '--dset-sources',
        nargs='*',
        default=['local'],
        choices=['local', 'huggingface', 'modelscope'],
        help=('the source of each dataset; it can accept one or the same '
              'number of args as the number of `datasets`, with one arg '
              'indicating that all datasets come from the same source. '
              '`local` represents the local path, `huggingface` represents '
              'the open-source data in the Huggingface Hub, `modelscope` '
              'indicates the open-source data in the Modelscope Hub.'))
    data_args.add_argument(
        '--dset-formats',
        nargs='*',
        default=['openai'],
        help=('the format of each dataset; it can accept one or the same '
              'number of args as the number of `datasets`, with one arg '
              'indicating that all datasets are the same format.'))
    data_args.add_argument(
        '--dset-sample-ratios',
        nargs='*',
        default=[1.0],
        help=('the sample ratio of each dataset; it can accept one or the '
              'same number of args as the number of `datasets`, with one arg '
              'indicating that all datasets use the same sample ratio.'))
    data_args.add_argument(
        '--dset-cache-dir',
        help=('the cache dir of the loaded datasets. When the `datasets` is '
              'set, the loaded datasets will be cached to this dir. If the '
              '`datasets` are not set, the cached dataset in this dir will be '
              'loaded.'))
    data_args.add_argument(
        '--dset-from-cache',
        action='store_true',
        help=('Load data directly from `dset-cache-dir`. This can save time '
              'on online tokenization, but if the tokenizer changed, '
              'recaching is needed.'))
    data_args.add_argument(
        '--dset-pack-level',
        choices=['hard', 'soft'],
        help=('the level of data packing. When `hard`, multiple data will be '
              'packed to `max_length`, potentially causing some data to be '
              'truncated, and the length of the packed data will always '
              'be `max_length`; When `soft`, it will pack multiple  data '
              'into nearly `max_length` without truncating the data.'))
    data_args.add_argument(
        '--max-length',
        type=int,
        default=2048,
        help=('the maximum length of each piece of data, any excess will be '
              'truncated.'))
    data_args.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='how many subprocesses to use for data loading.')
    data_args.add_argument(
        '--num-proc',
        type=int,
        default=8,
        help='how many subprocesses to use for data mapping.')
    data_args.add_argument('--file-pattern', type=str, default=None)
    data_args.add_argument('--group-by-length', action='store_true')

    optim_args = parser.add_argument_group('optim', 'Optim Related Settings')
    optim_args.add_argument(
        '--mirco-batch-size',
        type=int,
        default=1,
        help='batch size for each forward + backward pass')

    parser.add_argument('-c', '--config', default=None)
    parser.add_argument(
        '--work-dir',
        default='work_dirs',
        help='the dir to save logs and checkpoints')
    parser.add_argument(
        '--seed', type=int, default=0, help='random seed for the training')
    parser.add_argument(
        '--debug', action='store_true', help='Set logger level to `DEBUG`')
    args = parser.parse_args()
    return args


def is_interval(step, total_steps, interval):
    return (step + 1) % interval == 0 or (step + 1) == total_steps


def map_meta_modules(model, meta_model):
    modules = {name: mod for name, mod in model.named_modules()}
    meta_module_map = {
        mod: modules[name]
        for name, mod in meta_model.named_modules()
    }
    return meta_module_map


def build_llm_model(args, config, world_size, dtype=torch.float32):
    with LoadWoInit():
        llm = AutoModel.from_pretrained(
            args.llm,
            config=config,
            trust_remote_code=True,
            attn_implementation=config.attn_implementation,
            torch_dtype=config.torch_dtype)

    # Ensure all numerical values in the optimizer are fp32.
    # FSDP will use low precision during forward.
    llm.to(dtype)
    return llm


import types

from mmengine import MessageHub
from transformers.modeling_outputs import SequenceClassifierOutputWithPast


def reward_model_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    labels=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    transformer_outputs = self.model(
        input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    hidden_states = transformer_outputs[0]
    logits = self.score(hidden_states)

    if input_ids is not None:
        batch_size = input_ids.shape[0]
    else:
        batch_size = inputs_embeds.shape[0]

    if self.config.pad_token_id is None and batch_size != 1:
        raise ValueError(
            'Cannot handle batch sizes > 1 if no padding token is defined.')
    if self.config.pad_token_id is None:
        sequence_lengths = -1
    else:
        if input_ids is not None:
            # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
            sequence_lengths = torch.eq(
                input_ids, self.config.pad_token_id).int().argmax(-1) - 1
            sequence_lengths = sequence_lengths % input_ids.shape[-1]
            sequence_lengths = sequence_lengths.to(logits.device)
        else:
            sequence_lengths = -1

    # MODIFIED
    attn_context = MessageHub.get_instance('packed_sequence')
    cumulative_lengths = attn_context.get_info('cumulative_lengths')
    idx = cumulative_lengths[1:-1]  # 删去第一个 0 和 pad token 带来的最后一段 seq
    idx -= 1
    assert batch_size == 1
    pooled_logits = logits[:, idx]

    # pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

    loss = None

    if not return_dict:
        output = (pooled_logits, ) + transformer_outputs[1:]
        return ((loss, ) + output) if loss is not None else output

    return SequenceClassifierOutputWithPast(
        loss=loss,
        logits=pooled_logits,
        past_key_values=transformer_outputs.past_key_values,
        hidden_states=transformer_outputs.hidden_states,
        attentions=transformer_outputs.attentions,
    )


def dispatch_reward_model_forward(model):
    for module in model.modules():
        if module.__class__.__name__ == 'Qwen2ForRewardModel':
            module.forward = types.MethodType(reward_model_forward, module)


@logger.catch
@torch.no_grad()
def sft(args):
    ###########################################################################
    #                           1. Environment                                #
    ###########################################################################
    dist_launcher = infer_launcher()
    init_dist(dist_launcher)
    set_random_seed(args.seed)

    world_size = int(os.environ['WORLD_SIZE'])

    setup_parallel()
    dp_mesh = get_dp_mesh()
    dp_size = get_dp_world_size()

    global_batch_size = world_size
    mirco_batch_size = 1

    if args.dset_cache_dir and os.path.isdir(args.dset_cache_dir):
        if len(os.listdir(args.dset_cache_dir)) and not args.dset_from_cache:
            raise RuntimeError(f'`{args.dset_cache_dir}` is not an empty '
                               'folder, which may lead to inaccurate '
                               'cache results.')

    rank = dist.get_rank()
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    objects = [timestamp]
    dist.broadcast_object_list(objects, src=0)
    timestamp = objects[0]

    args.work_dir = os.path.join(args.work_dir, timestamp)
    mkdir_or_exist(args.work_dir)

    log_file = os.path.join(args.work_dir, f'rank{rank}.log')

    # Change the log format printed in the terminal
    lvl = 'DEBUG' if args.debug else 'INFO'
    logger.add(sys.stderr, level=lvl, format=log_format(rank, args.debug))
    # Change the format saved in the log file
    logger.add(log_file, format=log_format(rank), backtrace=True, catch=True)

    logger.info(args)
    if rank == 0:
        env = collect_env()
        import transformers

        import xtuner
        env['Transformers'] = transformers.__version__
        env['XTuner'] = f'{xtuner.__version__}+{get_git_hash(digits=6)}'
        runtime_env = OrderedDict()
        runtime_env.update(env)
        runtime_env['Seed'] = args.seed
        runtime_env['World Size'] = world_size
        runtime_env['Distributed launcher'] = dist_launcher

        runtime_env_info = '\n    ' + '\n    '.join(
            f'{k}: {v}' for k, v in runtime_env.items())
        dash_line = '-' * 60
        logger.info('\n' + dash_line + '\nRuntime environment:' +
                    runtime_env_info + '\n' + dash_line + '\n')
    # -------------------    Environment  End  ------------------------------ #

    ###########################################################################
    #                     2. Dataset & Dataloader                             #
    ###########################################################################

    start_load_data_t = time.time()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer if args.tokenizer else args.llm,
        trust_remote_code=True,
        padding_side='right')

    if args.dset_from_cache:
        if args.dset_pack_level == 'soft':
            init_fn = partial(
                SoftPackerForText.from_cache, max_length=args.max_length)
        elif args.dset_pack_level == 'hard':
            raise NotImplementedError
        else:
            init_fn = partial(
                TextTokenizeFunction.from_cache, max_length=args.max_length)
        _datasets = load_from_cache(args.dset_cache_dir, init_fn)
        dist.barrier()

    else:
        chat_template = CHAT_TEMPLATE_MAP[args.chat_template]
        tokenize_fns = []
        init_fns = []
        for dset_format in args.dset_formats:
            # If your data format is not in `SUPPORT_DATA_FORMATS`, you should
            # redefine a `tokenize_fn`, defining how to convert a piece of raw
            # data into tokenized data.
            # The tokenized data must include `input_ids`, `labels``,
            # and `num_tokens`.
            tokenize_fn = TextTokenizeFunction(tokenizer, chat_template,
                                               dset_format)

            if args.dset_pack_level == 'soft':
                init_fn = partial(
                    SoftPackerForText, max_length=args.max_length)
            elif args.dset_cache_dir:
                init_fn = partial(
                    TextTokenizedDataset, max_length=args.max_length)
            else:
                init_fn = partial(
                    TextOnlineTokenizeDataset, tokenize_fn=tokenize_fn)
                # Online tokenization is used when not using a pack dataset,
                # saving startup time.
                tokenize_fn = None

            tokenize_fns.append(tokenize_fn)
            init_fns.append(init_fn)

        _datasets = load_datasets(
            paths=args.datasets,
            cache_dir=args.dset_cache_dir,
            file_types=args.dset_file_types,
            sources=args.dset_sources,
            sample_ratios=args.dset_sample_ratios,
            num_proc=args.num_proc,
            map_fns=tokenize_fns,
            init_fns=init_fns,
            file_pattern=args.file_pattern)

    if (args.dset_pack_level or args.cache_dir) and rank == 0 and args.debug:
        # Only the tokenized datasets can count the number of tokens
        num_tokens = sum(dset.total_tokens for dset in _datasets)
        logger.debug(f'[Dataset] {num_tokens} tokens.')

    train_dataset = ConcatDataset(_datasets)

    if args.dset_pack_level and rank == 0:
        ori_samples = sum([dset.num_samples for dset in _datasets])
        packed_samples = len(train_dataset)
        logger.info(f'[Dataset] (Original) {ori_samples} samples.')
        logger.info(f'[Dataset] (Packed) {packed_samples} samples.')

    pack_batch = is_flash_attn_2_available()
    collator = TextCollator(pack_batch=pack_batch)

    if args.group_by_length:
        sampler = LengthGroupedSampler(train_dataset, dp_mesh,
                                       global_batch_size)
    else:
        sampler = ParallelSampler(
            train_dataset, dp_mesh, global_batch_size, shuffle=False)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=mirco_batch_size,
        num_workers=args.num_workers,
        # Ensure to round up or drop last based on the `global_batch_size`,
        # if you want to replace a custom sampler.
        sampler=sampler,
        collate_fn=collator,
        persistent_workers=args.num_workers > 0)

    if rank == 0:
        logger.info(f'[Dataloader] {len(train_dataloader)} batches.')
        _first_batch = [train_dataset[i] for i in range(mirco_batch_size)]
        _first_batch = collator(_first_batch)
        _decoded = tokenizer.batch_decode(_first_batch['input_ids'])
        logger.debug(f'[Dataloader] Training Batch:\n{_first_batch}')
        logger.debug(f'[Dataloader] Training Batch(Decoded):\n{_decoded}')

    load_data_cost_time = time.time() - start_load_data_t
    logger.info(f'[Dataset & Dataloader] Cost {load_data_cost_time:.2f}s')
    # -------------------    Dataset & Dataloader  End  --------------------- #

    ###########################################################################
    #                          3. FSDP                                        #
    ###########################################################################

    start_model_t = time.time()

    if args.dtype == 'auto':
        args.dtype = 'bf16' if torch.cuda.is_bf16_supported() else 'fp16'

    if args.dtype == 'fp16':
        dtype = torch.float16
        autocast = torch.cuda.amp.autocast(enabled=True, dtype=dtype)
        scaler = ShardedGradScaler()
    elif args.dtype == 'bf16':
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            autocast = torch.cuda.amp.autocast(enabled=True, dtype=dtype)
            scaler = None
        else:
            raise RuntimeError('The device does not support `bf16`, '
                               'please set `dtype` to `fp16`.')
    else:
        raise RuntimeError('`dtype` only supports `fp16`, `bf16` or `auto`, '
                           f'but found {args.dtype}.')

    llm_cfg = AutoConfig.from_pretrained(args.llm, trust_remote_code=True)
    if is_flash_attn_2_available():
        llm_cfg.attn_implementation = 'flash_attention_2'
    elif is_torch_sdpa_available():
        llm_cfg.attn_implementation = 'sdpa'

    llm_cfg.use_cache = False
    llm_cfg.torch_dtype = dtype

    with torch.device('meta'):
        meta_llm = build_llm_model(args, llm_cfg, world_size, dtype)

    if pack_batch:
        dispatch_modules(meta_llm)

    dispatch_reward_model_forward(meta_llm)

    # Only load parameters on rank 0 to avoid each rank repeatedly loading the
    # same model into the CPU, wasting memory
    if rank == 0:
        with torch.device('cpu'):
            llm = build_llm_model(args, llm_cfg, world_size, dtype)
        rank0_meta_llm = copy.deepcopy(meta_llm)
        meta_llm_map = map_meta_modules(llm, meta_llm)
    else:
        meta_llm_map = None

    dist.barrier()

    param_init_fn = partial(
        dp_lazy_init, module_map=meta_llm_map, dp_mesh=dp_mesh)

    if args.shard_strategy == 'full':
        fsdp_device_mesh = init_device_mesh('cuda', (world_size, ))
        strategy = ShardingStrategy.FULL_SHARD
    elif args.shard_strategy == 'hybrid':
        fsdp_device_mesh = init_device_mesh('cuda', (dp_size // 8, 8))
        strategy = ShardingStrategy.HYBRID_SHARD
    else:
        raise ValueError

    torch.cuda.reset_peak_memory_stats()
    shard_llm = FSDP(
        meta_llm,
        device_mesh=fsdp_device_mesh,
        sharding_strategy=strategy,
        auto_wrap_policy=layer_auto_wrap_policy,
        device_id=torch.cuda.current_device(),
        use_orig_params=True,
        param_init_fn=param_init_fn,
        sync_module_states=True,
    )

    max_memory = torch.cuda.max_memory_allocated()
    logger.info('[Model] During building the FSDP model, the peak GPU memory '
                f'is {max_memory/1024**3:.1f}GB.')

    fsdp_cost_time = time.time() - start_model_t
    logger.info(f'[Model] Cost {fsdp_cost_time:.2f}s')
    # --------------------------    FSDP  End  ------------------------------ #

    ###########################################################################
    #                          5. Training                                    #
    ###########################################################################

    total_steps = len(train_dataloader)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    max_memory = torch.cuda.max_memory_allocated()
    logger.info('[Train] Begin Train Loop. The current GPU memory is '
                f'{(max_memory / 1024**3):.1f}GB')
    shard_llm.train()
    train_dataloader.sampler.set_epoch(0, 0)
    data_iterator = iter(train_dataloader)
    for step in range(total_steps):
        torch.cuda.reset_peak_memory_stats()

        data = next(data_iterator)

        input_ids = data['input_ids'].cuda()
        labels = data['labels'].cuda()
        attention_mask = data['attention_mask'].cuda()
        num_tokens = data['num_tokens'].cuda()
        content = data['content']

        packed_ctx = packed_sequence(num_tokens, enable=pack_batch)

        with packed_ctx:
            outputs = shard_llm(
                input_ids=input_ids, attention_mask=attention_mask)

        pooled_logits = outputs.logits
        assert pooled_logits.numel() == len(
            content), f'{pooled_logits.numel()} {len(content)}'
        with open('qwen_rm_math.jsonl', 'a') as file:
            for i, score in enumerate(pooled_logits.flatten().cpu().tolist()):
                data = {
                    'query': content[i][0],
                    'output': content[i][1],
                    'score': score
                }
                json.dump(data, file)
                file.write('\n')

        logger.info(
            f'Step: {step + 1} / {total_steps}, processed number: {len(content)}.'
        )


if __name__ == '__main__':

    args = parse_args()
    sft(args)
