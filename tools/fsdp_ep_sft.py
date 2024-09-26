# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
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
import torch.nn as nn
from mmengine import mkdir_or_exist
from mmengine.dist import infer_launcher, init_dist
from mmengine.runner import set_random_seed
from mmengine.utils import get_git_hash
from mmengine.utils.dl_utils import collect_env
from peft import LoraConfig, get_peft_model
from torch.distributed._tensor import Shard, distribute_tensor
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import \
    apply_activation_checkpointing
from torch.distributed.checkpoint.state_dict import (StateDictOptions,
                                                     get_state_dict,
                                                     set_state_dict)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffload
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.wrap import _or_policy
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import ConcatDataset, DataLoader
from transformers import AutoConfig
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
from xtuner._lite.modelings import DeepseekV2ForCausalLM
from xtuner._lite.parallel import (LengthGroupedSampler, ParallelSampler,
                                   get_dp_mesh, get_dp_world_size, get_ep_mesh,
                                   get_experts_fsdp_mesh, get_sp_group,
                                   get_sp_world_size,
                                   reduce_sequence_parallel_loss,
                                   setup_parallel, split_for_sequence_parallel)
from xtuner._lite.parallel.experts import (del_moe_blocks, fsdp_moe_blocks,
                                           reduce_ep_grad, save_hf_model)
from xtuner._lite.parallel.fsdp import (LoadWoInit,
                                        all_required_grad_wrap_policy,
                                        checkpoint_check_fn, ep_lazy_init,
                                        layer_and_emb_wrap_policy)

logger = get_logger()

SUPPORT_DATA_FORMATS = OPENAI_FORMAT_MAP.keys()


def parse_args():
    parser = argparse.ArgumentParser(description='Train LLM')

    model_args = parser.add_argument_group('model', 'Group 1 description')
    model_args.add_argument('--llm', help='config file name or path.')
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
        '--use-lora', action='store_true', help='Apply the adapter to LLM.')
    model_args.add_argument(
        '--lora-targets',
        default=None,
        nargs='*',
        help='The names of the modules to apply the adapter to. ')
    model_args.add_argument(
        '--lora-r', default=64, type=int, help="Not updating vit's parameters")
    model_args.add_argument(
        '--lora-alpha',
        default=16,
        type=int,
        help='The alpha parameter for Lora scaling.')
    model_args.add_argument(
        '--lora-dropout',
        default=0.1,
        type=float,
        help='The dropout probability for Lora layers.')
    model_args.add_argument(
        '--lora-bias',
        default='none',
        help='The dropout probability for Lora layers.')
    model_args.add_argument(
        '--dtype',
        default='auto',
        choices=['fp16', 'bf16', 'auto'],
        help=("the dtype of the model forward. When set to 'auto', it will "
              'automatically determine whether bf16 is available, '
              'prioritizing the use of bf16.'))
    model_args.add_argument(
        '--selective-recompute',
        default=1.0,
        type=float,
        help=('the ratio of re-computation for transforemer layers. '
              'The maximum is 1; the larger the value, the less memory '
              'required for training. The default is 1, meaning all layers '
              'need to be re-computated.'))
    model_args.add_argument('--cpu-offload', action='store_true', help=(''))
    model_args.add_argument(
        '--shard-strategy',
        default='full',
        choices=['full', 'hybrid'],
        help=('The sharding strategy to be used for distributed training.'))

    data_args = parser.add_argument_group('data', 'Group 1 description')
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
        default=8,
        help='how many subprocesses to use for data loading.')
    data_args.add_argument(
        '--num-proc',
        type=int,
        default=8,
        help='how many subprocesses to use for data mapping.')
    data_args.add_argument('--group-by-length', action='store_true')
    data_args.add_argument('--file-pattern', type=str, default=None)

    dist_args = parser.add_argument_group('dist', 'Group 1 description')
    dist_args.add_argument('--sp-size', type=int, default=1, help='')
    dist_args.add_argument('--ep-size', type=int, default=1, help='')

    optim_args = parser.add_argument_group('optimizer', 'Group 1 description')
    optim_args.add_argument(
        '--mirco-batch-size',
        type=int,
        default=1,
        help='batch size for each forward + backward pass')
    optim_args.add_argument(
        '--global-batch-size',
        type=int,
        default=16,
        help='batch size for each optimizer step')
    optim_args.add_argument(
        '--lr',
        '--learning-rate',
        default=4e-5,
        type=float,
        help='the dir to save logs and models')
    optim_args.add_argument('--lr-min', default=1.5e-6, type=float)
    optim_args.add_argument(
        '--wd', default=0.01, type=float, help='weight decay.')
    optim_args.add_argument(
        '--max-grad-norm', default=1, type=float, help='gradient clipping')
    optim_args.add_argument(
        '-e', '--epochs', default=1, type=int, help='total training epochs.')
    optim_args.add_argument(
        '--warmup-ratio',
        default=0.03,
        type=float,
        help=('the proportion of training steps for learning rate warm-up in '
              'relation to the total training steps.'))

    parser.add_argument('-c', '--config', default=None)
    parser.add_argument(
        '--work-dir',
        default='work_dirs',
        help='the dir to save logs and models')
    parser.add_argument(
        '--checkpoint-interval',
        default=-1,
        type=float,
        help=('how many steps to save a checkpoint; it can be a floating '
              'point number less than 1, or an integer greater than or equal '
              "to 1. When it's a floating point, it will be multiplied by the "
              'total number of training steps.'))
    parser.add_argument(
        '--checkpoint-drop-optimizer',
        action='store_true',
        help=('only model parameters are saved when saving a checkpoint. '
              'This can significantly reduce the size of checkpoint files, '
              'but the saved checkpoints cannot be resumed.'))
    parser.add_argument('--log-interval', default=1, type=int)
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='specify checkpoint path to be resumed from.')
    parser.add_argument(
        '--seed', type=int, default=0, help='Random seed for the training')
    parser.add_argument(
        '--debug', action='store_true', help='Set logger level to `DEBUG`')
    args = parser.parse_args()
    return args


def is_interval(step, total_steps, interval):
    return (step + 1) % interval == 0 or (step + 1) == total_steps


def map_meta_modules(model, meta_model):
    modules = {name: mod for name, mod in model.named_modules()}
    # breakpoint()
    meta_module_map = {
        mod: modules[name]
        for name, mod in meta_model.named_modules()
    }
    return meta_module_map


def build_llm_model(args, config, ep_size, world_size, dtype=torch.float32):

    CONFIGS_TO_CUSTOM_MODELS = {
        'DeepseekV2Config': DeepseekV2ForCausalLM,
    }
    assert type(config).__name__ in CONFIGS_TO_CUSTOM_MODELS, type(
        config).__name__
    model_cls = CONFIGS_TO_CUSTOM_MODELS[type(config).__name__]

    moe_implementation = 'origin' if args.ep_size == 1 else 'ep'

    with LoadWoInit():
        llm = model_cls.from_pretrained(
            args.llm,
            moe_implementation=moe_implementation,
            ep_size=ep_size,
            trust_remote_code=True,
            torch_dtype=dtype,
            attn_implementation=config.attn_implementation)

    # Ensure all numerical values in the optimizer are fp32.
    # FSDP will use low precision during forward.
    llm.to(dtype)
    llm.config.use_cache = False

    if args.use_lora:
        llm.requires_grad_(False)
        if world_size > 1:
            llm.to(dtype)

        if args.lora_targets is None:
            llm_cls = llm.__class__.__name__
            args.lora_targets = LORA_TARGET_MAP[llm_cls]
        llm_lora_cfg = LoraConfig(
            target_modules=args.lora_targets,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias,
            task_type='CAUSAL_LM')
        llm = get_peft_model(llm, llm_lora_cfg)

    return llm


def log_format(rank, debug=False):

    formatter = f'[XTuner][RANK {rank}]'
    formatter += '[{time:YYYY-MM-DD HH:mm:ss}][<level>{level}</level>]'

    if debug:
        formatter += '[<cyan>{name}</cyan>:'
        formatter += '<cyan>{function}</cyan>:'
        formatter += '<cyan>{line}</cyan>]'

    formatter += ' <level>{message}</level>'
    return formatter


@logger.catch
def sft(args):
    ###########################################################################
    #                           1. Environment                                #
    ###########################################################################
    dist_launcher = infer_launcher()
    init_dist(dist_launcher)
    set_random_seed(args.seed)

    rank = dist.get_rank()
    world_size = int(os.environ['WORLD_SIZE'])
    sp_size = args.sp_size
    ep_size = args.ep_size
    setup_parallel(sp_size=sp_size, ep_size=ep_size)
    dp_size = get_dp_world_size()

    if args.global_batch_size < dp_size or args.global_batch_size % dp_size:
        raise ValueError(f'The `global_batch_size`({args.global_batch_size}) '
                         'should be divisible by the '
                         f'world_size({world_size}).')

    if (args.global_batch_size / dp_size) % args.mirco_batch_size:
        raise ValueError(f'The `global_batch_size`({args.global_batch_size}) '
                         f'should be divisible by the world_size({world_size})'
                         f' * `mirco_batch_size`({args.mirco_batch_size})')

    if args.dset_cache_dir and os.path.isdir(args.dset_cache_dir):
        if len(os.listdir(args.dset_cache_dir)) and not args.dset_from_cache:
            raise RuntimeError(f'`{args.dset_cache_dir}` is not an empty '
                               'folder, which may lead to inaccurate '
                               'cache results.')

    dp_mesh = get_dp_mesh()
    ep_mesh = get_ep_mesh()
    # fsdp device mesh for experts part of the moe model
    experts_fsdp_mesh = get_experts_fsdp_mesh()
    # fsdp device mesh for other parts of the moe model
    fsdp_mesh = init_device_mesh('cuda', (world_size, ))

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

    if (args.dset_pack_level or args.cache_dir) and rank == 0:
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
                                       args.global_batch_size)
    else:
        sampler = ParallelSampler(
            train_dataset, dp_mesh, args.global_batch_size, shuffle=True)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.mirco_batch_size,
        num_workers=args.num_workers,
        # Ensure to round up or drop last based on the `global_batch_size`,
        # if you want to replace a custom sampler.
        sampler=sampler,
        collate_fn=collator,
        persistent_workers=args.num_workers > 0)

    if rank == 0:
        logger.info(f'[Dataloader] {len(train_dataloader)} batches.')
        _first_batch = [train_dataset[i] for i in range(args.mirco_batch_size)]
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
        model = build_llm_model(
            args,
            llm_cfg,
            ep_size=ep_size,
            world_size=world_size,
            dtype=torch.float32)

    def _apply_ep(module, device_mesh):
        for m in module.modules():
            if type(m).__name__ in ('ExpertEp', 'GroupedLinear'):
                w1w3 = nn.Parameter(
                    distribute_tensor(m.w1w3, ep_mesh, [Shard(0)]))
                m.register_parameter('w1w3', w1w3)
                w2 = nn.Parameter(distribute_tensor(m.w2, ep_mesh, [Shard(0)]))
                m.register_parameter('w2', w2)

    _apply_ep(model, ep_mesh)
    if pack_batch:
        dispatch_modules(model)

    xtuner_load_timeout = timedelta(minutes=60)
    group_gloo = dist.new_group(backend='gloo', timeout=xtuner_load_timeout)

    if rank == 0:
        with torch.device('cpu'):
            master_model = build_llm_model(
                args, llm_cfg, ep_size=1, world_size=world_size, dtype=dtype)

        rank0_meta_llm = copy.deepcopy(master_model)
        if hasattr(rank0_meta_llm.config, 'moe_implementation'):
            del rank0_meta_llm.config.moe_implementation
        rank0_meta_llm.config.ep_size = 1
        master_mod_map = map_meta_modules(master_model, model)

    else:
        master_model = None
        master_mod_map = None

    dist.monitored_barrier(group=group_gloo, timeout=xtuner_load_timeout)
    logger.info('after barrier')

    decoder_layer_name = type(model.model.layers[0]).__name__

    lazy_param_init_fn = partial(
        ep_lazy_init,
        module_map=master_mod_map,
        ep_mesh=ep_mesh,
        experts_fsdp_mesh=experts_fsdp_mesh)

    policies = [
        partial(
            layer_and_emb_wrap_policy,
            vocab_size=llm_cfg.vocab_size,
            layer_cls=(decoder_layer_name, ))
    ]
    if args.use_lora:
        policies.append(all_required_grad_wrap_policy)

    if args.shard_strategy == 'full':
        fsdp_device_mesh = fsdp_mesh
        strategy = ShardingStrategy.FULL_SHARD
    elif args.shard_strategy == 'hybrid':
        raise NotImplementedError
    else:
        raise ValueError

    torch.cuda.reset_peak_memory_stats()
    print(f'before {torch.cuda.max_memory_allocated()/1e9}')

    model, experts = del_moe_blocks(model, moe_block_name='GroupedLinear')

    shard_model = FSDP(
        model,
        sharding_strategy=strategy,
        cpu_offload=CPUOffload(offload_params=args.cpu_offload),
        device_mesh=fsdp_device_mesh,
        mixed_precision=MixedPrecision(
            param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype),
        device_id=torch.cuda.current_device(),
        use_orig_params=True,
        param_init_fn=lazy_param_init_fn,
        sync_module_states=True,
        auto_wrap_policy=partial(_or_policy, policies=policies))

    torch.cuda.reset_peak_memory_stats()
    print(f'after other {torch.cuda.max_memory_allocated()/1e9}')

    shard_model = fsdp_moe_blocks(shard_model, experts, experts_fsdp_mesh,
                                  lazy_param_init_fn, torch.bfloat16)

    torch.cuda.reset_peak_memory_stats()
    print(f'after moe {torch.cuda.max_memory_allocated()/1e9}')

    if args.selective_recompute:
        check_fn = partial(
            checkpoint_check_fn,
            target=(decoder_layer_name, ),
            selective=args.selective_recompute)
        apply_activation_checkpointing(shard_model, check_fn=check_fn)

    fsdp_cost_time = time.time() - start_model_t
    logger.info(f'[Model] Cost {fsdp_cost_time:.2f}s')
    # --------------------------    FSDP  End  ------------------------------ #

    ###########################################################################
    #                      4. Optimizer & Scheduler                           #
    ###########################################################################

    requried_grad_params = [
        param for param in shard_model.parameters() if param.requires_grad
    ]

    optimizer = AdamW(
        requried_grad_params,
        fused=True,
        lr=args.lr,
        weight_decay=args.wd,
        betas=(0.9, 0.95))

    global_batch_size = args.global_batch_size
    mirco_batch_size = args.mirco_batch_size

    # `iter` means once forward+backward
    # `step` means once optimizer step
    # `iters_per_step` means gradient accumulative counts
    iters_per_step = global_batch_size // mirco_batch_size // dp_size
    iters_per_epoch = len(train_dataloader)
    steps_per_epoch = math.ceil(iters_per_epoch / iters_per_step)

    total_epochs = args.epochs
    total_steps = steps_per_epoch * total_epochs

    if args.checkpoint_interval == -1:
        checkpoint_interval = total_steps
    elif args.checkpoint_interval < 1:
        checkpoint_interval = int(total_steps * args.checkpoint_interval)
    else:
        checkpoint_interval = int(args.checkpoint_interval)

    warmup_steps = int(args.warmup_ratio * total_steps)

    def warmup_fn(x):
        return x / warmup_steps if x < warmup_steps else 1

    warmup_scheduler = LambdaLR(optimizer, warmup_fn)

    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=args.lr_min)

    start_step = 0

    # ----------------    Optimizer & Scheduler End   ----------------------- #

    ###########################################################################
    #                          5. Training                                    #
    ###########################################################################

    if args.resume:

        model_state_dict, optim_state_dict = get_state_dict(
            shard_model, optimizer)
        warump_state_dict = warmup_scheduler.state_dict()
        cosine_state_dict = cosine_scheduler.state_dict()

        state_dict = {
            'model': model_state_dict,
            'optimizer': optim_state_dict,
            'step': start_step,
            'total_steps': total_steps,
            'warmup_scheduler': warmup_scheduler.state_dict(),
            'cosine_scheduler': cosine_scheduler.state_dict()
        }
        reader = dcp.FileSystemReader(args.resume)
        dcp.load(state_dict, reader)

        if state_dict['total_steps'] != total_steps:
            raise RuntimeError

        set_state_dict(
            shard_model,
            optimizer,
            model_state_dict=model_state_dict,
            optim_state_dict=optim_state_dict)

        warmup_scheduler.load_state_dict(warump_state_dict)
        cosine_scheduler.load_state_dict(cosine_state_dict)

        start_step = state_dict['step']

    start_train_t = time.time()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    max_memory = torch.cuda.max_memory_allocated()
    logger.info('[Train] Begin Train Loop. The current GPU memory is '
                f'{(max_memory / 1024**3):.1f}GB')

    shard_model.train()
    for step in range(start_step, total_steps):

        epoch = step // steps_per_epoch
        epoch_inner_step = step % steps_per_epoch
        if epoch_inner_step == 0 or step == start_step:
            # For the first step of each epoch, the data order needs to be
            # readjusted.
            # Or after resuming, for the first step, the dataloader needs to
            # be adjusted to the position before resume.

            train_dataloader.sampler.set_epoch(epoch, epoch_inner_step)
            data_iterator = iter(train_dataloader)
            logger.info(f'Epoch {epoch + 1} Total Step {step + 1} starts!')

        if step < warmup_steps:
            warmup_scheduler.step()
            cur_lr = warmup_scheduler.get_last_lr()[0]
        else:
            cosine_scheduler.step()
            cur_lr = cosine_scheduler.get_last_lr()[0]

        torch.cuda.reset_peak_memory_stats()

        step_loss = 0
        step_data_time = 0
        step_start_t = time.time()
        step_consumed_tokens = 0
        # torch.cuda.empty_cache()
        for _ in range(iters_per_step):

            _data_start_t = time.time()
            data = next(data_iterator)
            step_data_time += time.time() - _data_start_t

            input_ids = data['input_ids'].cuda()
            labels = data['labels'].cuda()
            attention_mask = data['attention_mask'].cuda()
            num_tokens = data['num_tokens'].cuda()

            packed_ctx = packed_sequence(
                num_tokens, enable=pack_batch, sp_size=get_sp_world_size())

            with packed_ctx, autocast if args.use_lora else nullcontext():
                if get_sp_world_size() > 1:
                    sp_group = get_sp_group()
                    # `dim` is 1 as the shape of tensor is (bs, seq_len, ...)
                    input_ids = split_for_sequence_parallel(
                        input_ids, dim=1, sp_group=sp_group)
                    labels = split_for_sequence_parallel(
                        labels, dim=1, sp_group=sp_group)

                outputs = shard_model(
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=attention_mask,
                )
                loss = outputs.loss

                if get_sp_world_size() > 1:
                    tokens_cal_loss = (labels != -100).sum()
                    loss = reduce_sequence_parallel_loss(
                        loss, tokens_cal_loss, sp_group)

                avg_iter_loss = loss / iters_per_step
                if scaler and args.use_lora:
                    scaler.scale(avg_iter_loss).backward()
                else:
                    avg_iter_loss.backward()

            if args.dset_pack_level == 'soft':
                # During a soft pack process, the data with a length that is
                # still smaller than the max length after packing, will be
                # padded to the max length. The last element of num tokens
                # represents the count of pad tokens.
                step_consumed_tokens += num_tokens[:-1].sum(
                ) / get_sp_world_size()
            else:
                step_consumed_tokens += num_tokens.sum() / get_sp_world_size()
            step_loss += avg_iter_loss.item()

        reduce_ep_grad(shard_model)

        grad_norm = shard_model.clip_grad_norm_(args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        step_time = time.time() - step_start_t
        eta = step_time * (total_steps - step)
        eta = timedelta(seconds=int(eta))
        tgs = int(step_consumed_tokens / step_time)
        max_memory = torch.cuda.max_memory_allocated()
        if is_interval(step, total_steps, args.log_interval):
            # step_loss = sum(step_losses) / len(step_losses)
            logger.info(f'(Epoch {epoch + 1}) Step {step+1}/{total_steps}  '
                        f'lr: {cur_lr:.6f}  loss: {step_loss:.3f}  '
                        f'grad_norm: {grad_norm:.2f}  '
                        f'max_memory: {(max_memory / 1024**3):.1f}GB  '
                        f'step_consumed_tokens: {step_consumed_tokens}  '
                        f'tgs: {tgs}  data_time: {step_data_time:.2f}s  '
                        f'time: {step_time:.2f}s  '
                        f'eta: {eta}')

        if is_interval(step, total_steps, checkpoint_interval):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            max_memory = torch.cuda.max_memory_allocated()
            logger.info('[Checkpoint] Before saving checkpoint, the peak GPU '
                        f'memory is {max_memory/1024**3:.1f}GB.')

            num_digits = len(str(abs(total_steps)))
            work_dir = args.work_dir
            ckpt_dir = os.path.join(work_dir, f'ckpt-{step+1:0{num_digits}}')
            hf_dir = os.path.join(work_dir, f'hf-{step+1:0{num_digits}}')
            _options = StateDictOptions(cpu_offload=True, full_state_dict=True)

            saved_llm = copy.deepcopy(rank0_meta_llm) if rank == 0 else None
            save_hf_model(shard_model, saved_llm, tokenizer, hf_dir)

            if args.checkpoint_drop_optimizer:
                logger.warning('The saved checkpoint cannot be resumed. '
                               'If you want to save a resumable checkpoint, '
                               'please remove `--checkpoint-drop-optimizer` '
                               'from the command.')
            else:
                # FSDP cannot be saved via torch.save
                # Refer to https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html  # noqa: E501
                _options = StateDictOptions(
                    cpu_offload=True, ignore_frozen_params=True)
                (shard_model_state_dict,
                 shard_optimizer_state_dict) = get_state_dict(
                     shard_model, optimizer, options=_options)

                state_dict = {
                    'model': shard_model_state_dict,
                    'optimizer': shard_optimizer_state_dict,
                    'step': step,
                    'total_steps': total_steps,
                    'warmup_scheduler': warmup_scheduler.state_dict(),
                    'cosine_scheduler': cosine_scheduler.state_dict()
                }

                writer = dcp.FileSystemWriter(ckpt_dir)
                mkdir_or_exist(ckpt_dir)
                dcp.save(state_dict, writer)

            max_memory = torch.cuda.max_memory_allocated()
            logger.info('[Checkpoint] During saving checkpoint, the peak GPU '
                        f'memory is {max_memory/1024**3:.1f}GB.')

    train_cost_time = time.time() - start_train_t
    logger.info(f'[Train] Cost {timedelta(seconds=int(train_cost_time))}')
    # ------------------------    Training  End  ---------------------------- #


if __name__ == '__main__':

    args = parse_args()
    sft(args)