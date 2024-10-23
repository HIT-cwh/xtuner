import argparse
import copy
import math
import os
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timedelta
from functools import partial

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.init as init
from lmdeploy.pytorch.config import CacheConfig, ModelConfig
from lmdeploy.pytorch.engine.cache_engine import CacheEngine
from mmengine import MessageHub, mkdir_or_exist
from mmengine.dist import infer_launcher, init_dist
from mmengine.runner import set_random_seed
from torch import Tensor
from torch.distributed._tensor import DTensor, Shard, distribute_tensor
from torch.distributed.device_mesh import init_device_mesh
from torch.profiler import ProfilerActivity, profile, record_function

from xtuner._lite import AutoTokenizer, get_logger
from xtuner._lite.accelerate import (LORA_TARGET_MAP, dispatch_modules,
                                     packed_sequence)
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
    parser.add_argument('--ep-size', type=int, default=1)
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--input-len', type=int, default=256)
    parser.add_argument('--output-len', type=int, default=128)
    parser.add_argument('--work-dir', type=str, default=None)
    parser.add_argument('--max-length', type=int, default=None)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--n-routed-experts', type=int, default=80)
    parser.add_argument('--n-shared-experts', type=int, default=2)
    parser.add_argument('--num-experts-per-tok', type=int, default=8)
    parser.add_argument('--num-hidden-layers', type=int, default=56)

    parser.add_argument(
        '--llm',
        default=
        '/cpfs01/shared/public/public_hdd/llmeval/model_weights/hf_hub/models--internlm--internlm2_5-7b/snapshots/daa886c96bc86f54f03c725db5316adbbc3eb5db'
    )
    parser.add_argument('--tp_size', type=int, default=None)
    parser.add_argument('--hidden_size', type=int, default=None)
    parser.add_argument('--intermediate_size', type=int, default=None)
    parser.add_argument('--num_attention_heads', type=int, default=None)
    parser.add_argument('--num_key_value_heads', type=int, default=None)
    parser.add_argument('--num_hidden_layers', type=int, default=None)
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
def ep_lazy_init(module, module_map, ep_mesh):

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
    rank = dist.get_rank()
    ep_rank = ep_mesh.get_local_rank()

    if type(module).__name__ == 'GroupedLinear':
        n_experts = module.w1w3.shape[0]
        if rank == 0:
            w1w3_copy = master_module.w1w3.to(device).to(module.w1w3.dtype)
            w2_copy = master_module.w2.to(device).to(module.w2.dtype)
        else:
            w1w3_copy = torch.empty(
                (n_experts * ep_mesh.size(), *module.w1w3.shape[1:]),
                dtype=module.w1w3.dtype,
                device=device)
            w2_copy = torch.empty(
                (n_experts * ep_mesh.size(), *module.w2.shape[1:]),
                dtype=module.w2.dtype,
                device=device)
        torch.distributed.broadcast(w1w3_copy, 0)
        torch.distributed.broadcast(w2_copy, 0)
        w1w3 = w1w3_copy[ep_rank * n_experts:(ep_rank + 1) * n_experts]
        w2 = w2_copy[ep_rank * n_experts:(ep_rank + 1) * n_experts]
        module.w1w3.data.copy_(w1w3)
        module.w2.data.copy_(w2)
    else:
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

                ep_group = ep_mesh.get_group()
                torch.distributed.broadcast(p_copy, 0)
                param.data.copy_(p_copy)

        for name, buffer in module.named_buffers(recurse=False):
            if dist.get_rank() == 0:
                b_copy = master_buffers[name]
                b_copy = b_copy.to(device).to(buffer.dtype)
            else:
                b_copy = torch.empty_like(buffer)

            ep_group = ep_mesh.get_group()
            torch.distributed.broadcast(b_copy, 0)
            buffer.data.copy_(b_copy)

    torch.cuda.empty_cache()


def map_meta_modules(model, meta_model):
    modules = {name: mod for name, mod in model.named_modules()}
    meta_module_map = {
        mod: modules[name]
        for name, mod in meta_model.named_modules()
    }
    return meta_module_map


def cal(llm, cfg):
    numel_act = 0
    numel_total = 0
    numel_moe = 0
    numel_wo_moe = 0
    numel_attn = 0
    numel_act_moe = 0
    for name, param in llm.named_parameters():
        if 'expert' in name:
            numel_moe += param.numel()
        else:
            numel_wo_moe += param.numel()
        if '.experts.' in name:
            numel_act += param.numel(
            ) * cfg.num_experts_per_tok / cfg.n_routed_experts
        else:
            numel_act += param.numel()
        if 'attention' in name:
            numel_attn += param.numel()
        numel_total += param.numel()
        # print(name, param.numel()/1e9)
    print(
        f'Total act param: {numel_act / 1e9}, Total param: {numel_total / 1e9}, MoE param: {numel_moe / 1e9}, '
        f'Other param: {numel_wo_moe / 1e9}, Attn param: {numel_attn / 1e9}, MoE act param: {(numel_act - numel_wo_moe) / 1e9}'
    )


@torch.no_grad()
def benchmark_moe_block(args, bs_range):

    rank = dist.get_rank()
    logger.add(sys.stderr, level='SUCCESS', format=log_format(rank, ))

    set_random_seed(0)
    ep_size = args.ep_size
    setup_parallel(ep_size=ep_size)
    dp_size = get_dp_world_size()
    ep_mesh = get_ep_mesh()
    dp_mesh = get_dp_mesh()

    config = InternLM2MoEConfig(
        max_position_embeddings=8192,
        attn_implementation='flash_attention_2',
        moe_intermediate_size=1536,
        n_shared_experts=args.n_shared_experts,
        n_routed_experts=args.n_routed_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        ep_size=ep_size,
        routed_scaling_factor=1.0,
        topk_method='gready',
        n_group=1,
        topk_group=1,
        moe_layer_freq=1,
        first_k_dense_replace=1,
        norm_topk_prob=False,
        scoring_func='softmax',
        aux_loss_alpha=0.001,
        seq_aux=True,
        hidden_size=5120,
        intermediate_size=12288,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=40,
        num_key_value_heads=2,
        bias=False,
        rope_theta=50000000,
        architectures='InternLM2ForCausalLM',  # for lmdeploy
        torch_dtype=torch.bfloat16,

        # debug=True
    )

    with torch.device('meta'):
        model = InternLM2MoEForCausalLM._from_config(
            config,
            attn_implementation='flash_attention_2',
            torch_dtype=torch.bfloat16)
    
    if rank == 0:
        cal(model, config)
    
    moe_block = model.model.layers[1].feed_forward

    if rank == 0:
        with torch.device('meta'):
            master_model = InternLM2MoEForCausalLM._from_config(
                config,
                attn_implementation='flash_attention_2',
                torch_dtype=torch.bfloat16)
        master_moe_block = master_model.model.layers[1].feed_forward

        master_moe_block.to_empty(device=torch.cuda.current_device(), recurse=True)
        std = model.config.initializer_range
        for module in master_moe_block.modules():
            if isinstance(module, nn.Linear):
                device = module.weight.data.device
                module.weight.data.normal_(mean=0.0, std=std)
                if module.bias is not None:
                    module.bias.data.zero_()
                module.to(device)
            elif type(module).__name__ == 'GroupedLinear':
                device = module.w1w3.data.device
                module.cuda()
                for data in module.w1w3.data:
                    data.normal_(mean=0.0, std=std)
                for data in module.w2.data:
                    data.normal_(mean=0.0, std=std)
                module.to(device)
        master_moe_block.gate.reset_parameters()
        master_mod_map = map_meta_modules(master_moe_block, moe_block)
    else:
        master_moe_block, master_mod_map = None, None

    def _apply_ep(module, device_mesh):
        for m in module.modules():
            if type(m).__name__ in ('ExpertEp', 'GroupedLinear'):
                n_experts = m.w1w3.shape[0] // device_mesh.size()
                ep_rank = device_mesh.get_local_rank()
                w1w3 = nn.Parameter(m.w1w3[ep_rank * n_experts:(ep_rank + 1) *
                                           n_experts])
                m.register_parameter('w1w3', w1w3)
                w2 = nn.Parameter(m.w2[ep_rank * n_experts:(ep_rank + 1) *
                                       n_experts])
                m.register_parameter('w2', w2)

    _apply_ep(moe_block, ep_mesh)

    lazy_param_init_fn = partial(
        ep_lazy_init, module_map=master_mod_map, ep_mesh=ep_mesh)
    moe_block.apply(lazy_param_init_fn)

    # x = torch.randn(1, 64, 5120, device='cuda', dtype=torch.bfloat16) 
    # out = moe_block(x)

    tgs_per_block = []
    tgs_all_block = []
    for seq in bs_range:
        set_random_seed(dp_mesh.get_local_rank())
        xs = [torch.randn(1, seq, 5120, device='cuda', dtype=torch.bfloat16) for _ in range(args.num_hidden_layers)]
        moe_block(xs[0])
        torch.cuda.synchronize()
        t1 = time.time()
        with profile(activities=[
                            ProfilerActivity.CPU, ProfilerActivity.CUDA
                    ]) as prof:
            for x in xs:
                moe_block(x)
        if rank == 0:
            prof.export_chrome_trace(f'moe_block.json')
        torch.cuda.synchronize()
        t2 = time.time()
        if rank == 0:
            print(f'seq {seq}, time per block: {(t2 - t1) / args.num_hidden_layers}, time all blocks: {t2 - t1}, tgs {seq / (t2 - t1)}')
            tgs_all_block.append(seq / (t2 - t1))
            tgs_per_block.append(seq / (t2 - t1) * args.num_hidden_layers)
    return tgs_all_block, tgs_per_block


@torch.no_grad()
def benchmark_moe_block_lite(args, bs_range):

    rank = dist.get_rank()
    logger.add(sys.stderr, level='SUCCESS', format=log_format(rank, ))

    set_random_seed(0)
    ep_size = args.ep_size
    setup_parallel(ep_size=ep_size)
    dp_size = get_dp_world_size()
    ep_mesh = get_ep_mesh()
    dp_mesh = get_dp_mesh()

    config = InternLM2MoEConfig(
        max_position_embeddings=8192,
        attn_implementation='flash_attention_2',
        moe_intermediate_size=1408,
        n_shared_experts=2,
        n_routed_experts=64,
        num_experts_per_tok=6,
        ep_size=ep_size,
        routed_scaling_factor=1.0,
        topk_method='gready',
        n_group=1,
        topk_group=1,
        moe_layer_freq=1,
        first_k_dense_replace=1,
        norm_topk_prob=False,
        scoring_func='softmax',
        aux_loss_alpha=0.001,
        seq_aux=True,

        hidden_size=2048,
        intermediate_size=11008,
        num_hidden_layers=27,
        num_attention_heads=16,
        num_key_value_heads=4,
        bias=False,
        rope_theta=50000000,
        architectures='InternLM2ForCausalLM',  # for lmdeploy
        torch_dtype=torch.bfloat16,

        # debug=True
    )

    with torch.device('meta'):
        model = InternLM2MoEForCausalLM._from_config(
            config,
            attn_implementation='flash_attention_2',
            torch_dtype=torch.bfloat16)
    
    if rank == 0:
        cal(model, config)
    
    moe_block = model.model.layers[1].feed_forward

    if rank == 0:
        with torch.device('meta'):
            master_model = InternLM2MoEForCausalLM._from_config(
                config,
                attn_implementation='flash_attention_2',
                torch_dtype=torch.bfloat16)
        master_moe_block = master_model.model.layers[1].feed_forward

        master_moe_block.to_empty(device=torch.cuda.current_device(), recurse=True)
        std = model.config.initializer_range
        for module in master_moe_block.modules():
            if isinstance(module, nn.Linear):
                device = module.weight.data.device
                module.weight.data.normal_(mean=0.0, std=std)
                if module.bias is not None:
                    module.bias.data.zero_()
                module.to(device)
            elif type(module).__name__ == 'GroupedLinear':
                device = module.w1w3.data.device
                module.cuda()
                for data in module.w1w3.data:
                    data.normal_(mean=0.0, std=std)
                for data in module.w2.data:
                    data.normal_(mean=0.0, std=std)
                module.to(device)
        master_moe_block.gate.reset_parameters()
        master_mod_map = map_meta_modules(master_moe_block, moe_block)
    else:
        master_moe_block, master_mod_map = None, None

    def _apply_ep(module, device_mesh):
        for m in module.modules():
            if type(m).__name__ in ('ExpertEp', 'GroupedLinear'):
                n_experts = m.w1w3.shape[0] // device_mesh.size()
                ep_rank = device_mesh.get_local_rank()
                w1w3 = nn.Parameter(m.w1w3[ep_rank * n_experts:(ep_rank + 1) *
                                           n_experts])
                m.register_parameter('w1w3', w1w3)
                w2 = nn.Parameter(m.w2[ep_rank * n_experts:(ep_rank + 1) *
                                       n_experts])
                m.register_parameter('w2', w2)

    _apply_ep(moe_block, ep_mesh)

    lazy_param_init_fn = partial(
        ep_lazy_init, module_map=master_mod_map, ep_mesh=ep_mesh)
    moe_block.apply(lazy_param_init_fn)

    # x = torch.randn(1, 64, 5120, device='cuda', dtype=torch.bfloat16) 
    # out = moe_block(x)

    tgs_per_block = []
    tgs_all_block = []
    for seq in bs_range:
        set_random_seed(dp_mesh.get_local_rank())
        xs = [torch.randn(1, seq, config.hidden_size, device='cuda', dtype=torch.bfloat16) for _ in range(args.num_hidden_layers)]
        moe_block(xs[0])
        torch.cuda.synchronize()
        t1 = time.time()
        for x in xs:
            moe_block(x)
        torch.cuda.synchronize()
        t2 = time.time()
        if rank == 0:
            print(f'seq {seq}, time per block: {(t2 - t1) / args.num_hidden_layers}, time all blocks: {t2 - t1}, tgs {seq / (t2 - t1)}')
            tgs_all_block.append(seq / (t2 - t1))
            tgs_per_block.append(seq / (t2 - t1) * args.num_hidden_layers)
    return tgs_all_block, tgs_per_block


from xtuner._lite.modelings.internlm2 import (InternLM2Config,
                                              InternLM2ForCausalLM)

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

    tp_rank = tp_mesh.get_local_rank() if tp_mesh else 0
    tp_size = tp_mesh.size() if tp_mesh else 1
    if isinstance(module, ColwiseLinear):
        out_features = module.out_features
        in_features = module.in_features
        if rank == 0:
            p_copy = master_module.weight.to(device).to(module.weight.dtype)
        else:
            p_copy = torch.empty((out_features * tp_size, in_features),
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
            p_copy = torch.empty((out_features, in_features * tp_size),
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
def benchmark_dense_block(args, bs_range):
    # dist_launcher = infer_launcher()
    # init_dist(dist_launcher)

    rank = dist.get_rank()
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

    if args.tp_size > 1:
        model = parallelize(model, tp_mesh)
    
    mlp_block = model.model.layers[1].feed_forward
    if rank == 0:
        with torch.device('meta'):
            master_model = InternLM2ForCausalLM._from_config(
                config,
                attn_implementation='flash_attention_2',
                torch_dtype=torch.bfloat16)
        master_mlp_block = master_model.model.layers[1].feed_forward
        master_mlp_block.to_empty(device=torch.cuda.current_device(), recurse=True)
        std = model.config.initializer_range
        for module in master_mlp_block.modules():
            if isinstance(module, nn.Linear):
                device = module.weight.data.device
                module.weight.data.normal_(mean=0.0, std=std)
                if module.bias is not None:
                    module.bias.data.zero_()
                module.to(device)
        master_mod_map = map_meta_modules(master_mlp_block, mlp_block)
    else:
        master_mlp_block, master_mod_map = None, None
    
    lazy_param_init_fn = partial(
        tp_lazy_init, module_map=master_mod_map, tp_mesh=tp_mesh)
    mlp_block.apply(lazy_param_init_fn)
    tgs_per_block = []
    tgs_all_blocks = []
    for seq in bs_range:
        set_random_seed(dp_mesh.get_local_rank())
        xs = [torch.randn(1, seq, config.hidden_size, device='cuda', dtype=torch.bfloat16) for _ in range(args.num_hidden_layers)]
        mlp_block(xs[0])
        torch.cuda.synchronize()
        t1 = time.time()
        with profile(activities=[
                            ProfilerActivity.CPU, ProfilerActivity.CUDA
                    ]) as prof:
            for x in xs:
                mlp_block(x)
        if rank == 0:
            prof.export_chrome_trace(f'mlp_block.json')
        torch.cuda.synchronize()
        t2 = time.time()
        if rank == 0:
            print(f'seq {seq}, time per block: {(t2 - t1) / args.num_hidden_layers}, time all blocks: {t2 - t1}, tgs {seq / (t2 - t1) / args.tp_size}')
            tgs_all_blocks.append(seq / (t2 - t1) / args.tp_size)
            tgs_per_block.append(seq / (t2 - t1) * args.num_hidden_layers / args.tp_size)
    return tgs_all_blocks, tgs_per_block


def benchmark_100b():
    import matplotlib.pyplot as plt
    dist_launcher = infer_launcher()
    init_dist(dist_launcher)

    # bs_range = [8] + list(range(16, 257, 16)) + list(range(512, 8193, 512))
    bs_range = [8, 16, 32, 64, 96, 128, 160, 352]

    args = parse_args()
    args.ep_size = 8
    tgs_all_blocks1, tgs_per_block1 = benchmark_moe_block(args, bs_range)
    torch.cuda.empty_cache()
    # args.n_routed_experts = 160
    # args.num_experts_per_tok = 3
    # args.num_hidden_layers = 52
    # tgs_all_blocks2, tgs_per_block2 = benchmark_moe_block(args, bs_range)
    # torch.cuda.empty_cache()

    # args.num_experts_per_tok = 6
    # args.num_hidden_layers = 60
    # tgs_all_blocks4, tgs_per_block4 = benchmark_moe_block(args, bs_range)
    # torch.cuda.empty_cache()

    args.tp_size = 8
    args.hidden_size = 8192
    args.num_attention_heads = 64
    args.num_key_value_heads = 8
    args.intermediate_size = 28672
    args.num_hidden_layers = 80
    tgs_all_blocks3, tgs_per_block3 = benchmark_dense_block(args, bs_range)

    if dist.get_rank() == 0:
        x = list(bs_range)
        plt.plot(x, tgs_all_blocks1, label='MoE Block 80-10 EP 8')
        plt.plot(x, tgs_all_blocks2, label='MoE Block 160-5 EP 8')
        plt.plot(x, tgs_all_blocks4, label='MoE Block 160-8 EP 8')
        plt.plot(x, tgs_all_blocks3, label='Dense 70B MLP TP8')
        # plt.plot(x, y3, marker='^', label='Dense 70B TP4')

        plt.xlabel('SeqLen per GPU')
        plt.ylabel('MoE / MLP Block * layers TGS')
        plt.legend()
        plt.savefig('tgs_all_blocks.png')

        plt.clf()

        plt.plot(x, tgs_per_block1, label='MoE Block 80-10 EP 8')
        plt.plot(x, tgs_per_block2, label='MoE Block 160-5 EP 8')
        plt.plot(x, tgs_per_block4, label='MoE Block 160-8 EP 8')
        plt.plot(x, tgs_per_block3, label='Dense 70B MLP TP8')
        # plt.plot(x, y3, marker='^', label='Dense 70B TP4')

        plt.xlabel('SeqLen per GPU')
        plt.ylabel('MoE / MLP Block TGS')
        plt.legend()
        plt.savefig('tgs_per_blocks.png')


def benchmark_7b():
    import matplotlib.pyplot as plt
    dist_launcher = infer_launcher()
    init_dist(dist_launcher)

    bs_range = [8, 16, 32, 64, 96, 128, 256] + list(range(512, 8193, 512))
    args = parse_args()
    args.ep_size = 1

    tgs_all_blocks1, tgs_per_block1 = benchmark_moe_block_lite(args, bs_range)
    torch.cuda.empty_cache()

    args.tp_size = 1
    tgs_all_blocks2, tgs_per_block2 = benchmark_dense_block(args, bs_range)

    if dist.get_rank() == 0:
        x = list(bs_range)
        plt.plot(x, tgs_all_blocks1, label='MoE Block 64-8 EP 1')
        plt.plot(x, tgs_all_blocks2, label='Dense 7B MLP TP 1')
        # plt.plot(x, y3, marker='^', label='Dense 70B TP4')

        plt.xlabel('SeqLen per GPU')
        plt.ylabel('MoE / MLP Block * layers TGS')
        plt.legend()
        plt.savefig('tgs_all_blocks_lite.png')

        plt.clf()

        plt.plot(x, tgs_per_block1, label='MoE Block 64-8 EP 1')
        plt.plot(x, tgs_per_block2, label='Dense 7B MLP TP 1')
        # plt.plot(x, y3, marker='^', label='Dense 70B TP4')

        plt.xlabel('SeqLen per GPU')
        plt.ylabel('MoE / MLP Block TGS')
        plt.legend()
        plt.savefig('tgs_per_block_lite.png')


if __name__ == '__main__':
    benchmark_100b()
        
