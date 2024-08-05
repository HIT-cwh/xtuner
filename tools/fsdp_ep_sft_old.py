# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import gc
import math
import os
import random
import sys
import time
from datetime import datetime, timedelta
from functools import partial

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from mmengine import MessageHub, mkdir_or_exist
from mmengine.dataset import DefaultSampler
from mmengine.dist import init_dist
from torch.distributed._tensor import (DTensor, Replicate, Shard,
                                       distribute_tensor)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import \
    apply_activation_checkpointing
from torch.distributed.checkpoint.state_dict import (get_model_state_dict,
                                                     get_optimizer_state_dict,
                                                     get_state_dict,
                                                     set_state_dict)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (MixedPrecision, ShardingStrategy,
                                    StateDictType)
from torch.distributed.fsdp.wrap import (size_based_auto_wrap_policy,
                                         transformer_auto_wrap_policy)
from torch.distributed.tensor.parallel import (ColwiseParallel,
                                               RowwiseParallel,
                                               parallelize_module)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch.utils.data import ConcatDataset, DataLoader
from transformers.modeling_utils import PreTrainedModel
from transformers.utils.import_utils import (is_flash_attn_2_available,
                                             is_torch_sdpa_available)

from xtuner._lite import AutoModelForCausalLM, AutoTokenizer, get_logger
from xtuner._lite.accelerate import (LORA_TARGET_MAP, dispatch_modules,
                                     packed_sequence)
# from xtuner._lite.accelerate import packed_sequence_fwd_and_bwd
from xtuner._lite.chat import ChatTemplate
# from xtuner._lite.datasets import FinetuneDataset
from xtuner._lite.datasets import (OPENAI_FORMAT_MAP, SoftPackerForText,
                                   TextCollator, TextOnlineTokenizeDataset,
                                   TextTokenizedDataset, TextTokenizeFunction)
from xtuner._lite.datasets.load import (LOAD_FN_MAP, load_datasets,
                                        load_from_cache)
from xtuner._lite.modelings import DeepseekV2ForCausalLM
# from xtuner._lite.parallel.expert_parallel.setup_distributed import (
#     init_ep_dist, get_ep_group, get_fsdp_mesh, get_ep_mesh, get_device_mesh, get_ep_world_size)
# from xtuner._lite.parallel.sequence_parallel import (
#     init_sp_device_mesh, get_dp_mesh, get_sp_mesh, get_sp_group, get_sp_world_size,
#     get_dp_world_size, split_for_sequence_parallel, reduce_sequence_parallel_loss
# )
# from xtuner._lite.parallel import ParallelSampler
from xtuner._lite.parallel import (
    LengthGroupedSampler, ParallelSampler, get_dp_mesh, get_dp_world_size,
    get_ep_mesh, get_ep_world_size, get_experts_fsdp_mesh, get_sp_group,
    get_sp_mesh, get_sp_world_size, reduce_sequence_parallel_loss,
    setup_parallel, split_for_sequence_parallel)
from xtuner.model.transformers_models import (DeepseekV2Config,
                                              DeepseekV2ForCausalLM)
from xtuner.model.transformers_models.deepseek_v2.modeling_deepseek import \
    DeepseekV2DecoderLayer
from xtuner.model.utils import LoadWoInit
from xtuner.utils import get_origin_state_dict

# from torch.profiler import profile, record_function, ProfilerActivity

logger = get_logger()


def parallel_formatter(fsdp_rank, ep_rank, dp_rank, sp_rank, debug=False):

    formatter = f'[FSDP_RANK {fsdp_rank}][EP_RANK {ep_rank}][DP_RANK {dp_rank}][SP_RANK {sp_rank}]'
    formatter += '[{time:YYYY-MM-DD HH:mm:ss}][<level>{level}</level>]'

    if debug:
        formatter += '[<cyan>{name}</cyan>:'
        formatter += '<cyan>{function}</cyan>:'
        formatter += '<cyan>{line}</cyan>]'

    formatter += ' <level>{message}</level>'
    return formatter


def boolean_string(s):
    if s.lower() not in ['true', 'false']:
        raise ValueError('Not a valid boolean string')
    return s.lower() == 'true'


def parse_args():
    parser = argparse.ArgumentParser(description='Train LLM')

    model_args = parser.add_argument_group('model', 'Group 1 description')
    model_args.add_argument('-m', '--model', help='config file name or path.')
    model_args.add_argument('-t', '--tokenizer', default=None)
    model_args.add_argument(
        '--selective-checkpointing', default=1.0, type=float)

    data_args = parser.add_argument_group('data', 'Group 1 description')
    data_args.add_argument('--dataset', help='')
    data_args.add_argument('--dataset-format', default='openai', help='')
    data_args.add_argument('--dataset-cache', help='')
    data_args.add_argument('--max-length', type=int, default=2048, help='')
    data_args.add_argument('--mirco-batch-size', type=int, default=1, help='')
    data_args.add_argument('--num-workers', type=int, default=8, help='')
    data_args.add_argument('--sample-ratio', type=float, default=1., help='')
    data_args.add_argument('--cache-dir', type=str, default=None, help='')

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
    data_args.add_argument('--group-by-length', action='store_true')

    dist_args = parser.add_argument_group('dist', 'Group 1 description')
    dist_args.add_argument('--sp-size', type=int, default=1, help='')
    dist_args.add_argument('--ep-size', type=int, default=1, help='')

    optim_args = parser.add_argument_group('optimizer', 'Group 1 description')
    optim_args.add_argument(
        '--global-batch-size', type=int, default=16, help='')
    optim_args.add_argument(
        '--lr',
        '--learning-rate',
        default=4e-5,
        type=float,
        help='the dir to save logs and models')
    optim_args.add_argument('--lr-min', default=1.5e-6, type=float)
    optim_args.add_argument('--weight-decay', default=0.01, type=float)
    optim_args.add_argument('--max-grad-norm', default=1, type=float)
    optim_args.add_argument('-e', '--epochs', default=1, type=int)
    optim_args.add_argument('--warmup-ratio', default=0.03, type=float)

    # engine_args = parser.add_argument_group('engine', 'Group 1 description')
    parser.add_argument('--use-varlen', action='store_true', default=False)
    parser.add_argument('-c', '--config', default=None)
    parser.add_argument(
        '--work-dir',
        default='work_dirs',
        help='the dir to save logs and models')
    parser.add_argument('--checkpoint-interval', default=10000, type=int)
    parser.add_argument(
        '--save-optimizer',
        default=True,
        type=boolean_string,
        nargs='?',
        const=True,
        help='Whether to save the optimizer')
    parser.add_argument('--model-to-hf', action='store_true', default=False)
    parser.add_argument('--log-interval', default=1, type=int)
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='specify checkpoint path to be resumed from.')
    parser.add_argument(
        '--seed', type=int, default=0, help='Random seed for the training')
    args = parser.parse_args()
    return args


def is_interval(step, total_steps, interval):
    return (step + 1) % interval == 0 or (step + 1) == total_steps


@torch.no_grad()
def reduce_ep_grad(shard_model):
    ep_size = get_ep_world_size()
    for module in shard_model.modules():
        if type(module).__name__ == 'ExpertEp':
            if module.w1w3.grad is not None:
                module.w1w3.grad.div_(ep_size)
            if module.w2.grad is not None:
                module.w2.grad.div_(ep_size)


def get_moe_blocks(model):
    moe_blocks = []
    for module in model.modules():
        if type(module).__name__ == 'ExpertEp':
            moe_blocks.append(module)
    return moe_blocks


def _save_to_state_dict(module,
                        destination,
                        prefix,
                        keep_vars,
                        save_dtype=torch.float16):
    _EXTRA_STATE_KEY_SUFFIX = '_extra_state'
    for name, param in module._parameters.items():
        if param is not None:
            destination[prefix +
                        name] = param if keep_vars else param.detach().to(
                            save_dtype)
    for name, buf in module._buffers.items():
        if buf is not None and name not in module._non_persistent_buffers_set:
            destination[prefix + name] = buf if keep_vars else buf.detach().to(
                save_dtype)
    extra_state_key = prefix + _EXTRA_STATE_KEY_SUFFIX
    if getattr(module.__class__, 'get_extra_state',
               nn.Module.get_extra_state) is not nn.Module.get_extra_state:
        destination[extra_state_key] = module.get_extra_state()


from collections import OrderedDict


def get_ep_state_dict(module,
                      destination=None,
                      prefix='',
                      keep_vars=False,
                      save_dtype=torch.float16):
    if destination is None:
        destination = OrderedDict()
        destination._metadata = OrderedDict()

    local_metadata = dict(version=module._version)
    if hasattr(destination, '_metadata'):
        destination._metadata[prefix[:-1]] = local_metadata

    for hook in module._state_dict_pre_hooks.values():
        hook(module, prefix, keep_vars)
    _save_to_state_dict(module, destination, prefix, keep_vars, save_dtype)
    for name, child in module._modules.items():
        if child is not None:
            get_ep_state_dict(
                child,
                destination=destination,
                prefix=prefix + name + '.',
                keep_vars=keep_vars,
                save_dtype=save_dtype)
    for hook in module._state_dict_hooks.values():
        hook_result = hook(module, destination, prefix, local_metadata)
        if hook_result is not None:
            destination = hook_result

    if isinstance(module, FSDP) and type(
            module._fsdp_wrapped_module).__name__ == 'ExpertEp':
        if dist.get_rank() == 0:
            print(prefix, module)
        w1w3 = destination.pop(prefix + 'w1w3', None)
        if w1w3 is not None:
            assert isinstance(w1w3, DTensor), f'prefix {prefix} {w1w3}'
            w1w3 = w1w3.redistribute(
                placements=(Replicate(), ), async_op=False).to_local()
            if dist.get_rank() == 0:
                destination[prefix + 'w1w3'] = w1w3.cpu()

        w2 = destination.pop(prefix + 'w2', None)
        if w2 is not None:
            assert isinstance(w2, DTensor), f'prefix {prefix} {w2}'
            w2 = w2.redistribute(
                placements=(Replicate(), ), async_op=False).to_local()
            if dist.get_rank() == 0:
                destination[prefix + 'w2'] = w2.cpu()

    return destination


@torch.no_grad()
def save_hf_model(shard_model,
                  origin_model,
                  tokenizer,
                  ckpt_dir,
                  save_dtype=torch.float16):
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(shard_model, StateDictType.FULL_STATE_DICT, cfg):
        state_dict = get_ep_state_dict(shard_model, save_dtype=save_dtype)

    if (not dist.is_initialized()) or dist.get_rank() == 0:
        assert isinstance(origin_model, PreTrainedModel)
        state_dict = get_origin_state_dict(state_dict, origin_model)
        # todo: fix origin_model.config, delete ep related
        print(f'Saving LLM to {ckpt_dir}')
        origin_model.save_pretrained(ckpt_dir, state_dict=state_dict)
        print(f'Saving LLM tokenizer to {ckpt_dir}')
        tokenizer.save_pretrained(ckpt_dir)


@logger.catch
def sft(args):

    init_dist('pytorch')
    rank = dist.get_rank()
    world_size = int(os.environ['WORLD_SIZE'])
    sp_size = args.sp_size
    ep_size = args.ep_size
    setup_parallel(sp_size=sp_size, ep_size=ep_size)
    dp_size = get_dp_world_size()
    dp_mesh = get_dp_mesh()
    ep_mesh = get_ep_mesh()
    sp_mesh = get_sp_mesh()
    # fsdp device mesh for experts part of the moe model
    experts_fsdp_mesh = get_experts_fsdp_mesh()
    # fsdp device mesh for other parts of the moe model
    fsdp_mesh = init_device_mesh('cuda', (world_size, ))

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer if args.tokenizer else args.model,
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

    # sp_size = args.sp_size
    # ep_size = args.ep_size
    # fsdp_size = world_size // ep_size

    # init_ep_dist(fsdp_size, ep_size)
    # fsdp_mesh = get_fsdp_mesh()
    # ep_mesh = get_ep_mesh()

    # init_sp_device_mesh(sp_size)
    # dp_mesh = get_dp_mesh()
    # sp_mesh = get_sp_mesh()
    # dp_size = get_dp_world_size()

    # device_mesh = init_device_mesh('cuda', (world_size, ), mesh_dim_names=('world', ))['world']

    # fsdp_rank = fsdp_mesh.get_local_rank()
    # ep_rank = ep_mesh.get_local_rank()
    # dp_rank = dp_mesh.get_local_rank()
    # sp_rank = sp_mesh.get_local_rank()
    fsdp_rank = experts_fsdp_mesh.get_local_rank()
    ep_rank = ep_mesh.get_local_rank()
    dp_rank = dp_mesh.get_local_rank()
    sp_rank = sp_mesh.get_local_rank() if sp_size > 1 else 0
    # fsdp_mesh = experts_fsdp_mesh

    mkdir_or_exist(args.work_dir)

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    log_file = os.path.join(
        args.work_dir,
        f'{timestamp}.fsdp_rank{fsdp_rank}.ep_rank{ep_rank}.dp_rank{dp_rank}.sp_rank{sp_rank}.log'
    )

    formatter = parallel_formatter(fsdp_rank, ep_rank, dp_rank, sp_rank)
    # Change the log format printed in the terminal
    logger.add(sys.stderr, format=formatter)
    # Change the format saved in the log file
    logger.add(log_file, format=formatter, backtrace=True, catch=True)

    with torch.device('meta'):
        if ep_size == 1:
            model = DeepseekV2ForCausalLM.from_pretrained(
                args.model,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                attn_implementation='flash_attention_2',
                moe_implementation='origin')
        else:
            model = DeepseekV2ForCausalLM.from_pretrained(
                args.model,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                moe_implementation='ep',
                ep_size=ep_size,
                attn_implementation='flash_attention_2',
            )
        model.config.use_cache = False

    def _apply_ep(module, device_mesh):
        for m in module.modules():
            if type(m).__name__ == 'ExpertEp':
                w1w3 = nn.Parameter(
                    distribute_tensor(m.w1w3, ep_mesh, [Shard(0)]))
                m.register_parameter('w1w3', w1w3)
                w2 = nn.Parameter(distribute_tensor(m.w2, ep_mesh, [Shard(0)]))
                m.register_parameter('w2', w2)

    _apply_ep(model, ep_mesh)
    dispatch_modules(model)

    xtuner_load_timeout = timedelta(minutes=60)
    group_gloo = dist.new_group(backend='gloo', timeout=xtuner_load_timeout)

    if fsdp_rank == 0 and ep_rank == 0:
        with torch.device('cpu'):
            if ep_size == 1:
                master_model = DeepseekV2ForCausalLM.from_pretrained(
                    args.model,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    attn_implementation='flash_attention_2',
                    moe_implementation='origin')
            else:
                master_model = DeepseekV2ForCausalLM.from_pretrained(
                    args.model,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    moe_implementation='ep',
                    ep_size=1,
                    attn_implementation='flash_attention_2',
                )

        master_mods = {name: mod for name, mod in master_model.named_modules()}
        master_mod_map = {
            mod: master_mods[name]
            for name, mod in model.named_modules()
        }
    else:
        master_model = None
        master_mod_map = None

    dist.monitored_barrier(group=group_gloo, timeout=xtuner_load_timeout)
    logger.info('after barrier')

    @torch.no_grad
    def lazy_param_init_fn(module, ignore_modules=None):
        if isinstance(ignore_modules, list) and module in ignore_modules:
            return

        device = torch.cuda.current_device()
        module.to_empty(device=torch.cuda.current_device(), recurse=False)

        if fsdp_rank == 0 and ep_rank == 0:
            master_module = master_mod_map[module]
            master_params = {
                name: param
                for name, param in master_module.named_parameters(
                    recurse=False)
            }
            master_buffers = {
                name: buffer
                for name, buffer in master_module.named_buffers(recurse=False)
            }
        else:
            master_params = None
            master_buffers = None

        if fsdp_rank == 0:
            for name, param in module.named_parameters(recurse=False):
                if isinstance(param, DTensor):
                    if ep_mesh.get_local_rank() == 0:
                        p_copy = master_params[name]
                        p_copy = p_copy.to(device).to(torch.float32)
                    else:
                        p_copy = torch.empty(
                            param.shape, dtype=torch.float32, device=device)

                    mesh = param.device_mesh
                    placements = param.placements

                    p_dtensor = distribute_tensor(p_copy, mesh, placements)
                    param.data.copy_(p_dtensor)
                    del p_dtensor
                else:
                    if ep_mesh.get_local_rank() == 0:
                        p_copy = master_params[name]
                        p_copy = p_copy.to(device).to(torch.float32)
                    else:
                        p_copy = torch.empty_like(param)

                    ep_group = ep_mesh.get_group()
                    torch.distributed.broadcast(p_copy, 0, ep_group)
                    param.data.copy_(p_copy)

            for name, buffer in module.named_buffers(recurse=False):
                if ep_mesh.get_local_rank() == 0:
                    b_copy = master_buffers[name]
                    b_copy = b_copy.to(device).to(torch.float32)
                else:
                    b_copy = torch.empty_like(buffer)

                ep_group = ep_mesh.get_group()
                torch.distributed.broadcast(b_copy, 0, ep_group)
                buffer.data.copy_(b_copy)

        torch.cuda.empty_cache()

    layer_type = type(model.model.layers[0])
    for name, module in model.named_modules():
        module.name = name

    torch.cuda.reset_peak_memory_stats()
    print(f'before {torch.cuda.max_memory_allocated()/1e9}')

    experts = {}
    for name, module in model.named_modules():
        if type(module).__name__ == 'ExpertEp':
            experts[name] = module

    for name in experts.keys():
        splits = name.split('.')
        # module name 'model.layers.1.mlp.experts.0', experts is a ModuleList
        # we directly delete the whole ModuleList
        splits = splits[:-1]
        parent = model
        for module_name in splits[:-1]:
            if module_name.isdigit():
                parent = parent[int(module_name)]
            else:
                parent = getattr(parent, module_name)
        assert not isinstance(parent, nn.ModuleList)
        delattr(parent, splits[-1])

    checkpoint_layer = type(model.model.layers[0])

    shard_model = FSDP(
        model,
        device_mesh=fsdp_mesh,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16),
        device_id=torch.cuda.current_device(),
        use_orig_params=True,
        # param_init_fn=partial(lazy_param_init_fn, ignore_modules=moe_blocks),
        param_init_fn=lazy_param_init_fn,
        sync_module_states=True,
        auto_wrap_policy=partial(
            transformer_auto_wrap_policy, transformer_layer_cls={layer_type}),
        # ignored_modules=moe_blocks,
    )

    torch.cuda.reset_peak_memory_stats()
    print(f'after other {torch.cuda.max_memory_allocated()/1e9}')

    keys = list(experts.keys())
    for name in keys:
        expert = experts.pop(name)
        splits = name.split('.')
        # delete ModuleList '0'
        splits = splits[:-1]
        module_path = ['_fsdp_wrapped_module'
                       ] + splits[:3] + ['_fsdp_wrapped_module'] + splits[-2:]
        expert = FSDP(
            expert,
            device_mesh=experts_fsdp_mesh,
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16),
            device_id=torch.cuda.current_device(),
            use_orig_params=True,
            param_init_fn=lazy_param_init_fn,
            sync_module_states=True,
        )
        expert = nn.ModuleList([expert])
        parent = shard_model
        for module_name in module_path[:-1]:
            if module_name.isdigit():
                parent = parent[int(module_name)]
            else:
                parent = getattr(parent, module_name)
        parent.register_module(module_path[-1], expert)

        if dist.get_rank() == 0:
            torch.cuda.reset_peak_memory_stats()
            print(f'after pop {name}: {torch.cuda.max_memory_allocated()/1e9}')

    torch.cuda.reset_peak_memory_stats()
    print(f'after moe {torch.cuda.max_memory_allocated()/1e9}')

    if args.selective_checkpointing:

        def checkpoint_check_fn(submodule, target):
            ret = False
            if isinstance(submodule, target):
                if random.uniform(0, 1) < args.selective_checkpointing:
                    ret = True
            return ret

        checkpoint_check_fn = partial(
            checkpoint_check_fn, target=checkpoint_layer)

        apply_activation_checkpointing(
            shard_model, check_fn=checkpoint_check_fn)

    optimizer = AdamW(
        shard_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95))

    # train_dataloader = DataLoader(
    #     dataset,
    #     batch_size=args.mirco_batch_size,
    #     num_workers=args.num_workers,
    #     # sampler=DefaultSampler(dataset, shuffle=True),
    #     sampler=ParallelSampler(dataset, dp_mesh, shuffle=True),
    #     collate_fn=FinetuneDataset.dataloader_collate_fn,
    #     persistent_workers=True)

    global_batch_size = args.global_batch_size
    mirco_batch_size = args.mirco_batch_size

    iters_per_step = global_batch_size // mirco_batch_size // dp_size
    iters_per_epoch = len(train_dataloader)
    steps_per_epoch = math.ceil(iters_per_epoch / iters_per_step)

    total_epochs = args.epochs
    total_steps = steps_per_epoch * total_epochs

    warmup_steps = int(args.warmup_ratio * total_steps)

    def warmup_fn(x):
        return x / warmup_steps if x < warmup_steps else 1

    warmup_scheduler = LambdaLR(optimizer, warmup_fn)

    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=args.lr_min)

    start_step = 0
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

    shard_model.train()
    for step in range(start_step, total_steps):

        epoch = step // steps_per_epoch
        if step % steps_per_epoch == 0 or step == start_step:
            # For the first step of each epoch, the data order needs to be
            # readjusted.
            # Or after resuming, for the first step, the dataloader needs to
            # be adjusted to the position before resume.
            inner_step = step % steps_per_epoch
            train_dataloader.sampler.set_epoch(epoch, inner_step)
            data_iterator = iter(train_dataloader)
            logger.info(f'Epoch {epoch + 1} Total Step {step + 1} starts!')

        if step <= warmup_steps:
            warmup_scheduler.step()
            cur_lr = warmup_scheduler.get_lr()[0]
        else:
            cosine_scheduler.step()
            cur_lr = cosine_scheduler.get_lr()[0]

        torch.cuda.reset_peak_memory_stats()

        step_losses = []
        step_data_time = 0
        step_start_t = time.time()
        step_consumed_tokens = 0
        # torch.cuda.empty_cache()
        if (step + 1) % steps_per_epoch == 0:
            # last step in one epoch
            iters_per_step_cur = iters_per_epoch - (steps_per_epoch -
                                                    1) * iters_per_step
        else:
            iters_per_step_cur = iters_per_step
        for i in range(iters_per_step_cur):
            # if step * iters_per_step + i == iters_per_epoch:
            #     break

            _data_start_t = time.time()
            data = next(data_iterator)
            step_data_time += time.time() - _data_start_t

            input_ids = data['input_ids'].cuda()
            labels = data['labels'].cuda()
            attention_mask = data['attention_mask'].cuda()
            num_tokens = data['num_tokens'].cuda()

            packed_ctx = packed_sequence(
                num_tokens, enable=True, sp_size=get_sp_world_size())

            with packed_ctx:
                outputs = shard_model(
                    input_ids=input_ids,
                    labels=labels,
                    # position_ids=position_ids,
                    attention_mask=attention_mask,
                )
                loss = outputs.loss
                scaled_loss = loss / iters_per_step_cur
                scaled_loss.backward()
                step_consumed_tokens += attention_mask.sum()

            step_losses.append(loss.item())

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
            step_loss = sum(step_losses) / len(step_losses)
            logger.info(
                f'(Epoch {epoch + 1}) Step {step+1}/{total_steps}  '
                f'lr: {cur_lr:.6f}  loss: {step_loss:.3f}  '
                f'grad_norm: {grad_norm:.2f}  '
                f'max_memory: {(max_memory / 1024**3):.1f}GB  step_consumed_tokens: {step_consumed_tokens}  '
                f'tgs: {tgs}  data_time: {step_data_time:.2f}s  '
                f'time: {step_time:.2f}s  '
                f'eta: {eta}')

        if is_interval(step, total_steps, args.checkpoint_interval):
            # FSDP cannot be saved via torch.load
            # Refer to https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html  # noqa: E501
            # model_state_dict, optimizer_state_dict = get_state_dict(
            #     shard_model, optimizer)
            work_dir = args.work_dir
            num_digits = len(str(abs(total_steps)))
            ckpt_dir = os.path.join(work_dir, f'ckpt-{step+1:0{num_digits}}')
            if args.save_optimizer:
                model_state_dict = get_model_state_dict(shard_model)

                state_dict = {
                    'model': model_state_dict,
                    'step': step,
                    'total_steps': total_steps,
                    'warmup_scheduler': warmup_scheduler.state_dict(),
                    'cosine_scheduler': cosine_scheduler.state_dict()
                }

                optimizer_state_dict = get_optimizer_state_dict(
                    shard_model, optimizer)
                state_dict['optimizer'] = state_dict

                writer = dcp.FileSystemWriter(ckpt_dir)
                mkdir_or_exist(ckpt_dir)
                dcp.save(state_dict, writer)

            if args.model_to_hf:
                hf_model_dir = os.path.join(ckpt_dir, 'hf_model')
                save_hf_model(shard_model, master_model, tokenizer,
                              hf_model_dir)


if __name__ == '__main__':

    args = parse_args()

    sft(args)
