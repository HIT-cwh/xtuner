import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision

from ..setup import get_ep_world_size


def del_moe_blocks(model, moe_block_name='ExpertEp'):
    experts = {}
    for name, module in model.named_modules():
        if type(module).__name__ == moe_block_name:
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

    return model, experts


def fsdp_moe_blocks(shard_model, experts, experts_fsdp_mesh, param_init_fn,
                    dtype):
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
                param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype),
            device_id=torch.cuda.current_device(),
            use_orig_params=True,
            param_init_fn=param_init_fn,
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

    return shard_model


@torch.no_grad()
def reduce_ep_grad(shard_model):
    ep_size = get_ep_world_size()
    for module in shard_model.modules():
        if type(module).__name__ in ('ExpertEp', 'GroupedLinear'):
            if module.w1w3.grad is not None:
                module.w1w3.grad.div_(ep_size)
            if module.w2.grad is not None:
                module.w2.grad.div_(ep_size)
