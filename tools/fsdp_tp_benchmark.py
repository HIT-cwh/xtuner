import argparse
import copy
import math
import os
import sys
import time
from datetime import datetime, timedelta
from functools import partial

import torch
import torch.distributed as dist
import torch.nn as nn
from lmdeploy.pytorch.config import CacheConfig, ModelConfig
from lmdeploy.pytorch.engine.cache_engine import CacheEngine
from mmengine import MessageHub, mkdir_or_exist
from mmengine.dist import infer_launcher, init_dist
from mmengine.runner import set_random_seed
from torch import Tensor
from torch.distributed._tensor import DTensor, Shard, distribute_tensor
from torch.distributed.device_mesh import init_device_mesh
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import AutoConfig, AutoModelForCausalLM

from xtuner._lite import AutoTokenizer, get_logger
from xtuner._lite.accelerate import (LORA_TARGET_MAP, dispatch_modules,
                                     packed_sequence)
from xtuner._lite.modelings.internlm2 import (InternLM2Config,
                                              InternLM2ForCausalLM)
from xtuner._lite.modelings.internlm2_moe import (InternLM2MoEConfig,
                                                  InternLM2MoEForCausalLM)
from xtuner._lite.parallel import (LengthGroupedSampler, ParallelSampler,
                                   get_dp_mesh, get_dp_world_size, get_ep_mesh,
                                   get_experts_fsdp_mesh, get_sp_group,
                                   get_sp_world_size, get_tp_mesh,
                                   reduce_sequence_parallel_loss,
                                   setup_parallel, split_for_sequence_parallel)
from xtuner._lite.parallel.fsdp import (LoadWoInit,
                                        all_required_grad_wrap_policy,
                                        checkpoint_check_fn, ep_lazy_init,
                                        layer_and_emb_wrap_policy)

logger = get_logger()


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
    parser.add_argument(
        '--llm',
        default=
        '/cpfs01/shared/public/public_hdd/llmeval/model_weights/hf_hub/models--internlm--internlm2_5-7b/snapshots/daa886c96bc86f54f03c725db5316adbbc3eb5db'
    )
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--input-len', type=int, default=256)
    parser.add_argument('--output-len', type=int, default=128)
    parser.add_argument('--work-dir', type=str, default=None)
    parser.add_argument('--tp-size', type=int, default=1)
    parser.add_argument('--max-length', type=int, default=None)
    parser.add_argument('--profile', action='store_true')

    custom_model_args = parser.add_argument_group('model',
                                                  'Custom model structure')
    custom_model_args.add_argument('--hidden-size', type=int, default=None)
    custom_model_args.add_argument(
        '--num-attention-heads', type=int, default=None)
    custom_model_args.add_argument(
        '--num-key-value-heads', type=int, default=None)
    custom_model_args.add_argument(
        '--intermediate-size', type=int, default=None)
    custom_model_args.add_argument(
        '--num-hidden-layers', type=int, default=None)

    args = parser.parse_args()
    return args


def map_meta_modules(model, meta_model):
    modules = {name: mod for name, mod in model.named_modules()}
    meta_module_map = {
        mod: modules[name]
        for name, mod in meta_model.named_modules()
    }
    return meta_module_map


@torch.no_grad
def ep_lazy_init(module, module_map):

    device = torch.cuda.current_device()
    module.to_empty(device=torch.cuda.current_device(), recurse=False)

    if dist.get_rank() == 0:
        master_module = module_map[module]
        master_params = {
            name: param
            for name, param in master_module.named_parameters(recurse=False)
        }
        master_buffers = {
            name: buffer
            for name, buffer in master_module.named_buffers(recurse=False)
        }
    else:
        master_params = None
        master_buffers = None

    for name, param in module.named_parameters(recurse=False):
        if isinstance(param, DTensor):
            if dist.get_rank() == 0:
                p_copy = master_params[name]
                p_copy = p_copy.to(device).to(param.dtype)
            else:
                p_copy = torch.empty(
                    param.shape, dtype=param.dtype, device=device)

            mesh = param.device_mesh
            placements = param.placements

            p_dtensor = distribute_tensor(p_copy, mesh, placements)
            param.data.copy_(p_dtensor)
        else:
            if dist.get_rank() == 0:
                p_copy = master_params[name]
                p_copy = p_copy.to(device).to(param.dtype)
            else:
                p_copy = torch.empty_like(param)

            torch.distributed.broadcast(p_copy, 0)
            param.data.copy_(p_copy)

    for name, buffer in module.named_buffers(recurse=False):
        if dist.get_rank() == 0:
            b_copy = master_buffers[name]
            b_copy = b_copy.to(device).to(buffer.dtype)
        else:
            b_copy = torch.empty_like(buffer)

        torch.distributed.broadcast(b_copy, 0)
        buffer.data.copy_(b_copy)

    torch.cuda.empty_cache()


def sample(logits, top_k=40, top_p=0, temperature=1.0):
    # if (logits.argmax(-1) == 0).sum() > 0:
    #     logger.info(logits)
    return logits.argmax(-1)
    # Apply temperature if necessary
    if temperature != 1.0:
        logits = logits / temperature

    # Apply top-k if necessary
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        _, indices = logits.topk(top_k)
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask.scatter_(-1, indices, False)
        logits.masked_fill_(mask, -torch.inf)

    probs = logits.softmax(-1)
    return torch.multinomial(probs, 1).squeeze(-1)


def pack_sequence(input_ids: list[Tensor]):
    num_tokens = []
    for ids in input_ids:
        num_tokens.append(ids.shape[1])
    packed_input_ids = torch.cat(input_ids, dim=1)
    return packed_input_ids, torch.IntTensor(num_tokens)


def packed_cumulative_length(num_tokens):
    device = num_tokens.device
    _zero_length = torch.zeros(1, device=device)
    _pad_length = torch.cat([_zero_length, num_tokens]).int()
    cumulative_lengths = torch.cumsum(_pad_length, 0).int()
    return cumulative_lengths


def contiguous_batching_generate(model,
                                 input_ids: list[Tensor],
                                 stop_token_ids=[],
                                 max_batch_size=64,
                                 max_new_tokens=128,
                                 max_length=2048,
                                 tp_size=1,
                                 device='cuda',
                                 use_half_prefilling=False):

    model.config.use_cache = True

    block_size = 256
    max_batch_size = min(max_batch_size, len(input_ids))
    num_blocks = max_length // block_size * max_batch_size
    cache_config = CacheConfig(max_batch_size, block_size, 0, num_blocks)
    model_config = ModelConfig.from_hf_config(model.config)
    ########
    model_config.head_dim = 128
    model_config.k_head_dim = 128
    model_config.v_head_dim = 128
    ########
    cache_engine = CacheEngine(cache_config, model_config, world_size=tp_size)
    block_table = torch.arange(num_blocks).reshape(max_batch_size, -1)

    _packed_ids, _num_tokens = pack_sequence(input_ids[:max_batch_size])
    _position_ids = [
        torch.arange(seq.numel()) for seq in input_ids[:max_batch_size]
    ]
    _packed_pos_ids = torch.cat(_position_ids, dim=0).unsqueeze(0)
    _cumulative_length = packed_cumulative_length(_num_tokens)

    next_input_ids = _packed_ids.to(device)
    next_position_ids = _packed_pos_ids.to(device)
    next_start_pos = _cumulative_length[:-1].to(device)
    next_end_pos = (_cumulative_length[1:] - 1).to(device)
    next_query_length = _num_tokens.to(device)
    next_cache_length = _num_tokens.to(device)
    next_block_table = block_table.to(device).to(torch.int32)
    next_cumulative_length = _cumulative_length.to(device)
    next_is_prefilling = True

    num_sessions = len(input_ids)
    computing = [i for i in range(max_batch_size)]
    waiting = [i for i in range(max_batch_size, num_sessions)]

    responses = [[] for _ in range(num_sessions)]

    torch.cuda.synchronize()
    t1 = time.time()

    with torch.profiler.record_function('generate'):
        idx = -1

        while len(waiting) or len(computing):
            idx += 1

            with torch.profiler.record_function(f'idx {idx}'):

                if next_is_prefilling and use_half_prefilling:
                    # 分段 prefill
                    bs = max_batch_size // 2
                    block_offsets = next_block_table[:bs]
                    kv_seq_length = next_cache_length[:bs]
                    q_seq_length = next_query_length[:bs]
                    length_all = kv_seq_length.sum().item()
                    position_ids = next_position_ids[:, :length_all]
                    max_kv_seq_length = kv_seq_length.max()
                    max_q_seq_length = q_seq_length.max()
                    _cumulative_length_half = packed_cumulative_length(
                        q_seq_length)
                    next_end_pos = (_cumulative_length_half[1:] - 1).to(device)
                    q_start_loc = _cumulative_length_half[:-1].to(device)
                    cumulative_length = _cumulative_length_half.to(device)
                    is_prefilling = True

                    attn_ctx = MessageHub.get_instance('paged_attention')
                    attn_ctx.update_info('block_offsets', block_offsets)
                    attn_ctx.update_info('kv_seq_length', kv_seq_length)
                    attn_ctx.update_info('q_seq_length', q_seq_length)
                    attn_ctx.update_info('position_ids', position_ids)
                    attn_ctx.update_info('max_kv_seq_length',
                                         max_kv_seq_length)
                    attn_ctx.update_info('max_q_seq_length', max_q_seq_length)
                    attn_ctx.update_info('q_start_loc', q_start_loc)
                    attn_ctx.update_info('cumulative_length',
                                         cumulative_length)
                    attn_ctx.update_info('is_prefilling', is_prefilling)
                    attn_ctx.update_info('debug', False)

                    gpu_cache_half = []
                    num_blocks_half = num_blocks // 2
                    for i, cache_per_layer in enumerate(
                            cache_engine.gpu_cache):
                        gpu_cache_half.append(
                            (cache_per_layer[0][:num_blocks_half],
                             cache_per_layer[1][:num_blocks_half]))

                    outputs = model(
                        input_ids=next_input_ids[:, :length_all],
                        position_ids=position_ids,
                        past_key_values=gpu_cache_half,
                        cache_position=position_ids,
                    )
                    sampled1 = sample(outputs.logits[0, next_end_pos])

                    for key in list(attn_ctx.runtime_info.keys()):
                        attn_ctx.pop_info(key)

                    block_offsets = next_block_table[:bs]
                    kv_seq_length = next_cache_length[bs:]
                    q_seq_length = next_query_length[bs:]
                    length_all = kv_seq_length.sum().item()
                    position_ids = next_position_ids[:, -length_all:]
                    max_kv_seq_length = kv_seq_length.max()
                    max_q_seq_length = q_seq_length.max()
                    _cumulative_length_half = packed_cumulative_length(
                        q_seq_length)
                    next_end_pos = (_cumulative_length_half[1:] - 1).to(device)
                    q_start_loc = _cumulative_length_half[:-1].to(device)
                    cumulative_length = _cumulative_length_half.to(device)
                    is_prefilling = True

                    attn_ctx = MessageHub.get_instance('paged_attention')
                    attn_ctx.update_info('block_offsets', block_offsets)
                    attn_ctx.update_info('kv_seq_length', kv_seq_length)
                    attn_ctx.update_info('q_seq_length', q_seq_length)
                    attn_ctx.update_info('position_ids', position_ids)
                    attn_ctx.update_info('max_kv_seq_length',
                                         max_kv_seq_length)
                    attn_ctx.update_info('max_q_seq_length', max_q_seq_length)
                    attn_ctx.update_info('q_start_loc', q_start_loc)
                    attn_ctx.update_info('cumulative_length',
                                         cumulative_length)
                    attn_ctx.update_info('is_prefilling', is_prefilling)
                    attn_ctx.update_info('debug', True)

                    gpu_cache_half = []
                    for i, cache_per_layer in enumerate(
                            cache_engine.gpu_cache):
                        gpu_cache_half.append(
                            (cache_per_layer[0][num_blocks_half:],
                             cache_per_layer[1][num_blocks_half:]))

                    outputs = model(
                        input_ids=next_input_ids[:, -length_all:],
                        position_ids=position_ids,
                        past_key_values=gpu_cache_half,
                        cache_position=position_ids,
                    )
                    sampled2 = sample(outputs.logits[0, next_end_pos])

                    sampled = torch.cat([sampled1, sampled2])

                else:

                    attn_ctx = MessageHub.get_instance('paged_attention')
                    attn_ctx.update_info('block_offsets', next_block_table)
                    attn_ctx.update_info('kv_seq_length', next_cache_length)
                    attn_ctx.update_info('q_seq_length', next_query_length)
                    attn_ctx.update_info('position_ids', next_position_ids)
                    attn_ctx.update_info('max_kv_seq_length',
                                         next_cache_length.max())
                    attn_ctx.update_info('max_q_seq_length',
                                         next_query_length.max())
                    attn_ctx.update_info('q_start_loc', next_start_pos)
                    attn_ctx.update_info('cumulative_length',
                                         next_cumulative_length)
                    attn_ctx.update_info('is_prefilling', next_is_prefilling)

                    outputs = model(
                        input_ids=next_input_ids,
                        position_ids=next_position_ids,
                        past_key_values=cache_engine.gpu_cache,
                        cache_position=next_position_ids,
                    )
                    # TODO (pppppM) support sampling
                    sampled = sample(outputs.logits[0, next_end_pos])

                for key in list(attn_ctx.runtime_info.keys()):
                    attn_ctx.pop_info(key)

                _next_input_ids = []
                _next_position_ids = []
                _next_computing = []
                _next_query_length = []
                _next_cache_length = []
                _next_block_table = []

                for i, sess_id in enumerate(computing):
                    token_id = sampled[i]
                    responses[sess_id].append(token_id.item())

                    _sess_new_tokens = len(responses[sess_id])
                    _sess_len = _sess_new_tokens + input_ids[sess_id].numel()

                    stop = (
                        _sess_new_tokens >= max_new_tokens
                        or _sess_len >= max_length
                        or token_id in stop_token_ids)

                    if stop:
                        # session ended
                        if len(waiting):
                            # next step is prefilling
                            new_sess_id = waiting.pop(0)
                            new_sess = input_ids[new_sess_id].to(device)

                            _new_sess_len = new_sess.size(-1)
                            # new session override the cache of the stopped session
                            _next_block_table.append(next_block_table[i])
                            _next_computing.append(new_sess_id)
                            _next_input_ids.append(new_sess)
                            _next_position_ids.append(
                                torch.arange(_new_sess_len))
                            _next_query_length.append(_new_sess_len)
                            _next_cache_length.append(_new_sess_len)
                    else:
                        # next step is decoding
                        _next_computing.append(sess_id)
                        _next_block_table.append(next_block_table[i])
                        _next_input_ids.append(token_id.reshape(1, -1))
                        _next_position_ids.append(
                            torch.arange(_sess_len - 1, _sess_len))
                        _next_query_length.append(1)
                        _next_cache_length.append(_sess_len)

                computing = _next_computing
                if len(computing) == 0:
                    # All sessions have ended.
                    assert len(waiting) == 0
                    break

                _packed_ids, _num_tokens = pack_sequence(_next_input_ids)
                _cumulative_length = packed_cumulative_length(_num_tokens)

                next_input_ids = _packed_ids.to(device)
                next_position_ids = torch.cat(
                    _next_position_ids, dim=0).unsqueeze(0)
                next_position_ids = next_position_ids.to(device)
                next_start_pos = _cumulative_length[:-1].to(device)
                next_end_pos = (_cumulative_length[1:] - 1).to(device)
                next_query_length = torch.IntTensor(_next_query_length).to(
                    device)
                next_cache_length = torch.IntTensor(_next_cache_length).to(
                    device)
                next_block_table = torch.stack(_next_block_table).to(device)

                next_cumulative_length = _cumulative_length.to(device)
                next_is_prefilling = False

        torch.cuda.synchronize()
    if dist.get_rank() == 0:
        # logger.info(responses)
        logger.success(f'time: {time.time() - t1}')

    for i in range(len(cache_engine.gpu_cache)):
        cache_engine.gpu_cache.pop()

    del cache_engine
    torch.cuda.empty_cache()

    model.config.use_cache = False

    return responses


# from torch.distributed.tensor.parallel import (
#     parallelize_module,
#     ColwiseParallel,
#     RowwiseParallel,
#     PrepareModuleInput,
#     SequenceParallel
# )
# from torch.distributed._tensor import Replicate
# def parallelize(model, tp_mesh):
#     layer_tp_plan = {
#         'attention.wqkv': ColwiseParallel(),
#         'attention.wo': RowwiseParallel(),
#         'feed_forward.w1': ColwiseParallel(),
#         'feed_forward.w2': RowwiseParallel(),
#         'feed_forward.w3': ColwiseParallel(),
#     }

#     for layer in model.model.layers:
#         attention = layer.attention
#         attention.num_heads = attention.num_heads // tp_mesh.size()
#         attention.hidden_size = attention.hidden_size // tp_mesh.size()
#         parallelize_module(
#             module=layer,
#             device_mesh=tp_mesh,
#             parallelize_plan=layer_tp_plan,
#         )

#     # model = parallelize_module(
#     #     module=model,
#     #     device_mesh=tp_mesh,
#     #     parallelize_plan={
#     #         'model.tok_embeddings':
#     #         RowwiseParallel(input_layouts=Replicate(), ),
#     #         'output': ColwiseParallel(output_layouts=Replicate(), ),
#     #     })

#     return model

import torch.nn.functional as F


class ColwiseLinear(nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        self.register_parameter('bias', None)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    @classmethod
    def from_linear(cls, tp_mesh, mod):
        in_features = mod.in_features
        out_features = mod.out_features // tp_mesh.size()
        with torch.device(mod.weight.device):
            col_fc = cls(in_features, out_features).to(mod.weight.dtype)
        return col_fc


class RowwiseLinear(nn.Module):

    def __init__(self, in_features: int, out_features: int, tp_mesh):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        self.register_parameter('bias', None)
        self.tp_group = tp_mesh.get_group()

    def forward(self, input):
        out = F.linear(input, self.weight, self.bias)
        dist.all_reduce(out, group=self.tp_group)
        return out

    @classmethod
    def from_linear(cls, tp_mesh, mod):
        in_features = mod.in_features // tp_mesh.size()
        out_features = mod.out_features
        with torch.device(mod.weight.device):
            col_fc = cls(in_features, out_features,
                         tp_mesh).to(mod.weight.dtype)
        return col_fc


def parallelize(model, tp_mesh):
    for layer in model.model.layers:
        attention = layer.attention
        attention.num_heads = attention.num_heads // tp_mesh.size()
        attention.hidden_size = attention.hidden_size // tp_mesh.size()
        attention.wqkv = ColwiseLinear.from_linear(tp_mesh, attention.wqkv)
        attention.wo = RowwiseLinear.from_linear(tp_mesh, attention.wo)
        feed_forward = layer.feed_forward
        feed_forward.w1 = ColwiseLinear.from_linear(tp_mesh, feed_forward.w1)
        feed_forward.w3 = ColwiseLinear.from_linear(tp_mesh, feed_forward.w3)
        feed_forward.w2 = RowwiseLinear.from_linear(tp_mesh, feed_forward.w2)
    return model


@torch.no_grad
def tp_lazy_init(module, module_map, tp_mesh):

    device = torch.cuda.current_device()
    module.to_empty(device=torch.cuda.current_device(), recurse=False)
    rank = dist.get_rank()

    if dist.get_rank() == 0:
        master_module = module_map[module]
        master_params = {
            name: param
            for name, param in master_module.named_parameters(recurse=False)
        }
        master_buffers = {
            name: buffer
            for name, buffer in master_module.named_buffers(recurse=False)
        }
    else:
        master_params = None
        master_buffers = None

    tp_rank = tp_mesh.get_local_rank()
    if isinstance(module, ColwiseLinear):
        out_features = module.out_features
        in_features = module.in_features
        if rank == 0:
            p_copy = master_module.weight.to(device).to(module.weight.dtype)
        else:
            p_copy = torch.empty((out_features * tp_mesh.size(), in_features),
                                 dtype=module.weight.dtype,
                                 device=device)
        torch.distributed.broadcast(p_copy, 0)
        w = p_copy[tp_rank * out_features:(tp_rank + 1) * out_features]
        module.weight.data.copy_(w)
    elif isinstance(module, RowwiseLinear):
        out_features = module.out_features
        in_features = module.in_features
        if rank == 0:
            p_copy = master_module.weight.to(device).to(module.weight.dtype)
        else:
            p_copy = torch.empty((out_features, in_features * tp_mesh.size()),
                                 dtype=module.weight.dtype,
                                 device=device)
        torch.distributed.broadcast(p_copy, 0)
        w = p_copy[:, tp_rank * in_features:(tp_rank + 1) * in_features]
        module.weight.data.copy_(w)
    else:
        for name, param in module.named_parameters(recurse=False):
            if dist.get_rank() == 0:
                p_copy = master_params[name]
                p_copy = p_copy.to(device).to(param.dtype)
            else:
                p_copy = torch.empty_like(param)

            torch.distributed.broadcast(p_copy, 0)
            param.data.copy_(p_copy)

        for name, buffer in module.named_buffers(recurse=False):
            if dist.get_rank() == 0:
                b_copy = master_buffers[name]
                b_copy = b_copy.to(device).to(buffer.dtype)
            else:
                b_copy = torch.empty_like(buffer)

            torch.distributed.broadcast(b_copy, 0)
            buffer.data.copy_(b_copy)

    torch.cuda.empty_cache()


@torch.no_grad()
def benchmark(args):

    dist_launcher = infer_launcher()
    init_dist(dist_launcher)

    rank = dist.get_rank()

    if args.work_dir is not None:
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        objects = [timestamp]
        dist.broadcast_object_list(objects, src=0)
        timestamp = objects[0]
        args.work_dir = os.path.join(args.work_dir, timestamp)
        mkdir_or_exist(args.work_dir)
        log_file = os.path.join(args.work_dir, f'rank{rank}.log')
        logger.add(
            log_file, format=log_format(rank), backtrace=True, catch=True)

    logger.add(sys.stderr, level='INFO', format=log_format(rank, ))

    set_random_seed(0)
    rank = dist.get_rank()
    setup_parallel(tp_size=args.tp_size)
    dp_size = get_dp_world_size()
    dp_mesh = get_dp_mesh()
    tp_mesh = get_tp_mesh()

    path = args.llm
    config = InternLM2Config.from_pretrained(path)
    config.attn_implementation = 'flash_attention_2'

    if args.hidden_size is not None:
        config.hidden_size = args.hidden_size
    if args.intermediate_size is not None:
        config.intermediate_size = args.intermediate_size
    if args.num_attention_heads is not None:
        config.num_attention_heads = args.num_attention_heads
    if args.num_key_value_heads is not None:
        config.num_key_value_heads = args.num_key_value_heads
    if args.num_hidden_layers is not None:
        config.num_hidden_layers = args.num_hidden_layers

    with torch.device('meta'):
        model = InternLM2ForCausalLM._from_config(
            config,
            attn_implementation='flash_attention_2',
            torch_dtype=torch.bfloat16)

    numel = 0
    for param in model.parameters():
        numel += param.numel() / 1e9
    if rank == 0:
        print(numel)

    if dist.get_rank() == 0:
        print('after build meta')

    if args.tp_size > 1:
        model = parallelize(model, tp_mesh)

    if dist.get_rank() == 0:
        print('after parallelize')

    if rank == 0:
        # LoadWoInit 不负责 tensor.normal_ 这种初始化，所以模型仍然初始化了，这个为了避免实例化模型在cpu上初始化参数耗时
        with torch.device('cpu'), LoadWoInit():
            master_model = InternLM2ForCausalLM._from_config(
                config,
                attn_implementation='flash_attention_2',
                torch_dtype=torch.bfloat16)
        master_model.to(device='cpu')
        master_mod_map = map_meta_modules(master_model, model)
        logger.info('after build master model')
    else:
        master_model = None
        master_mod_map = None

    # lazy_param_init_fn = partial(
    #     ep_lazy_init,
    #     module_map=master_mod_map)

    # model.apply(lazy_param_init_fn)

    lazy_param_init_fn = partial(
        tp_lazy_init, module_map=master_mod_map, tp_mesh=tp_mesh)

    model.apply(lazy_param_init_fn)

    dispatch_modules(model)

    bs = args.bs
    input_len = args.input_len
    output_len = args.output_len

    set_random_seed(tp_mesh.get_local_rank())
    input_ids = [
        torch.randint(0, 10000, (1, input_len)).long().cuda()
        for _ in range(bs)
    ]
    logger.success(f'bs {bs} input_len {input_len} output_len {output_len}')

    if args.profile:

        with profile(activities=[
                            ProfilerActivity.CPU, ProfilerActivity.CUDA
                    ]) as prof:

            torch.cuda.empty_cache()
            model.eval()
            contiguous_batching_generate(
                model,
                input_ids,
                max_batch_size=bs,
                max_new_tokens=output_len,
                max_length=math.ceil((input_len + output_len - 1) / 256) * 256 if args.max_length is None else args.max_length,
                use_half_prefilling=False,
                tp_size=args.tp_size)
            torch.cuda.empty_cache()
            contiguous_batching_generate(
                model,
                input_ids,
                max_batch_size=bs,
                max_new_tokens=output_len,
                max_length=math.ceil((input_len + output_len - 1) / 256) * 256 if args.max_length is None else args.max_length,
                use_half_prefilling=False,
                tp_size=args.tp_size)
            torch.cuda.empty_cache()
            contiguous_batching_generate(
                model,
                input_ids,
                max_batch_size=bs,
                max_new_tokens=output_len,
                max_length=math.ceil((input_len + output_len - 1) / 256) * 256 if args.max_length is None else args.max_length,
                use_half_prefilling=True,
                tp_size=args.tp_size)
        if rank == 0:
            prof.export_chrome_trace(f'xtuner_fixtp_70b_{bs}_{input_len}_{output_len}.json')
    else:
        torch.cuda.empty_cache()
        contiguous_batching_generate(
            model,
            input_ids,
            max_batch_size=bs,
            max_new_tokens=output_len,
            max_length=math.ceil((input_len + output_len - 1) / 256) * 256 if args.max_length is None else args.max_length,
            use_half_prefilling=True,
            tp_size=args.tp_size)
        torch.cuda.empty_cache()
        contiguous_batching_generate(
            model,
            input_ids,
            max_batch_size=bs,
            max_new_tokens=output_len,
            max_length=math.ceil((input_len + output_len - 1) / 256) * 256 if args.max_length is None else args.max_length,
            use_half_prefilling=True,
            tp_size=args.tp_size)
        torch.cuda.empty_cache()
        contiguous_batching_generate(
            model,
            input_ids,
            max_batch_size=bs,
            max_new_tokens=output_len,
            max_length=math.ceil((input_len + output_len - 1) / 256) * 256 if args.max_length is None else args.max_length,
            use_half_prefilling=False,
            tp_size=args.tp_size)
        

    return

    # bss = [
    #     64, 64, 64, 128, 128, 128, 256,
    #     64, 64, 64, 128, 128, 128, 256,
    #     64, 64, 64, 128, 128, 128, 256,
    # ]

    # input_lens = [
    #     256, 1024, 2048, 256, 1024, 2048, 256,
    #     1, 1, 1, 1, 1, 1, 1,
    #     256, 512, 1024, 256, 512, 1024, 256
    #     ]
    # output_lens = [
    #     1, 1, 1, 1, 1, 1, 1,
    #     256, 1024, 2048, 256, 1024, 2048, 256,
    #     128, 128, 128, 128, 128, 128, 128
    #     ]

    bss = [64, 128, 256, 512, 1024, 1536, 2048, 3072]

    input_lens = [1, 1, 1, 1, 1, 1, 1, 1]
    output_lens = [128, 128, 128, 128, 128, 128, 128, 128]

    for bs, input_len, output_len in zip(bss, input_lens, output_lens):
        torch.cuda.empty_cache()
        if rank == 0:
            logger.info(
                f'bs {bs} input_len {input_len} output_len {output_len} ...')
        input_ids = [
            torch.randint(0, 10000, (1, input_len)).long().cuda()
            for _ in range(bs)
        ]
        contiguous_batching_generate(
            model,
            input_ids,
            max_batch_size=bs,
            max_new_tokens=output_len,
            max_length=math.ceil((input_len + output_len) / 256) * 256,
            use_half_prefilling=True)
        torch.cuda.empty_cache()

        contiguous_batching_generate(
            model,
            input_ids,
            max_batch_size=bs,
            max_new_tokens=output_len,
            max_length=math.ceil((input_len + output_len) / 256) * 256,
            use_half_prefilling=False)
        torch.cuda.empty_cache()

        contiguous_batching_generate(
            model,
            input_ids,
            max_batch_size=bs,
            max_new_tokens=output_len,
            max_length=math.ceil((input_len + output_len) / 256) * 256,
            use_half_prefilling=True)
        del input_ids
        time.sleep(5)


if __name__ == '__main__':
    args = parse_args()
    benchmark(args)
