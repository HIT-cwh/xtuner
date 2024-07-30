import json
import os
import re
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.nn as nn
from mmengine import print_log
from torch.distributed._tensor import DTensor, Replicate
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.modeling_utils import PreTrainedModel, load_state_dict
from transformers.utils import (SAFE_WEIGHTS_INDEX_NAME, WEIGHTS_INDEX_NAME,
                                is_safetensors_available)

from ..comm import barrier

SUPPORT_MODELS = (
    'DeepseekV2ForCausalLM',
    'MixtralForCausalLM',
)

ORDER_MAPPING = dict(
    DeepseekV2ForCausalLM=dict(down_proj=0, gate_proj=1, up_proj=2),
    MixtralForCausalLM=dict(down_proj=1, gate_proj=0, up_proj=2),
)

PARAM_NAME_MAPPING = dict(
    DeepseekV2ForCausalLM=dict(
        gate_proj='gate_proj', up_proj='up_proj', down_proj='down_proj'),
    MixtralForCausalLM=dict(gate_proj='w1', up_proj='w3', down_proj='w2'),
)


def print_on_rank0(info):
    if (not dist.is_initialized()) or dist.get_rank() == 0:
        print_log(info, 'current')


def get_expert_num_per_shard(model):
    for module in model.modules():
        if hasattr(module, 'expert_in_one_shard'):
            return module.expert_in_one_shard


def mix_sort(expert_name):
    components = re.findall(r'(\D+|\d+)', expert_name)
    out = [int(comp) if comp.isdigit() else comp for comp in components]
    return tuple(out)


def _get_merged_param_name(origin_param_name, expert_num_per_shard):
    split_name = origin_param_name.split('.experts.')
    expert_idx = re.findall(r'\d+', split_name[1])[0]
    expert_idx = int(expert_idx)
    assert expert_idx % expert_num_per_shard == 0
    shard_idx = expert_idx // expert_num_per_shard
    w1w3 = split_name[0] + f'.experts.{shard_idx}.w1w3'
    w2 = split_name[0] + f'.experts.{shard_idx}.w2'
    return w1w3, w2


def _merge_experts_weight(state_dict, expert_num_per_shard, order_mapping):
    experts_name = [key for key in state_dict.keys() if '.experts.' in key]
    experts_name = sorted(experts_name, key=mix_sort)
    linear_num_per_expert = 3
    linear_num_per_shard = expert_num_per_shard * linear_num_per_expert
    expert_shard_num = len(experts_name) // linear_num_per_shard
    for shard_idx in range(expert_shard_num):
        begin, end = shard_idx * linear_num_per_shard, (
            shard_idx + 1) * linear_num_per_shard
        experts_name_cur = experts_name[begin:end]

        down_proj_weight = [
            state_dict.pop(key)
            for key in experts_name_cur[order_mapping['down_proj']::3]
        ]
        gate_proj_weight = [
            state_dict.pop(key)
            for key in experts_name_cur[order_mapping['gate_proj']::3]
        ]
        up_proj_weight = [
            state_dict.pop(key)
            for key in experts_name_cur[order_mapping['up_proj']::3]
        ]
        w1 = torch.stack(gate_proj_weight)
        w3 = torch.stack(up_proj_weight)
        w1w3 = torch.cat([w1, w3], dim=1)
        assert w1w3.ndim == 3, w1w3.shape
        w2 = torch.stack(down_proj_weight)
        assert w2.ndim == 3, w2.shape
        merged_key_w1w3, merged_key_w2 = _get_merged_param_name(
            experts_name_cur[0], expert_num_per_shard)
        print_on_rank0(f'merged key {merged_key_w1w3}')
        state_dict[merged_key_w1w3] = w1w3
        print_on_rank0(f'merged key {merged_key_w2}')
        state_dict[merged_key_w2] = w2

    return


def load_state_dict_into_model(model_to_load, pretrained_model_path):

    model_name = type(model_to_load).__name__
    if model_name not in SUPPORT_MODELS:
        raise RuntimeError(
            f'Only models in {SUPPORT_MODELS} may need to load pretrained '
            f'weights via `load_state_dict_into_model`, but got {model_name}.')
    order_mapping = ORDER_MAPPING[model_name]

    index_file = os.path.join(pretrained_model_path, WEIGHTS_INDEX_NAME)
    safe_index_file = os.path.join(pretrained_model_path,
                                   SAFE_WEIGHTS_INDEX_NAME)
    index_present = os.path.isfile(index_file)
    safe_index_present = os.path.isfile(safe_index_file)
    assert index_present or (safe_index_present and is_safetensors_available())
    if safe_index_present and is_safetensors_available():
        load_index = safe_index_file
    else:
        load_index = index_file
    with open(load_index, encoding='utf-8') as f:
        index = json.load(f)
    weight_map = index['weight_map']
    unloaded_shard_files = list(set(weight_map.values()))
    unloaded_shard_files.sort(reverse=True)

    expert_num_per_shard = get_expert_num_per_shard(model_to_load)
    error_msgs = []

    def load(module: nn.Module, state_dict, unloaded_shard_files, prefix=''):
        params_to_gather = []
        param_names = []
        for name, param in module.named_parameters(
                prefix=prefix[:-1], recurse=False):
            while name not in state_dict:
                assert len(unloaded_shard_files) > 0
                shard_file = unloaded_shard_files.pop()
                shard_file = os.path.join(pretrained_model_path, shard_file)
                print_on_rank0(
                    f'{name} not in state_dict, loading {shard_file}')
                new_shard = load_state_dict(shard_file, is_quantized=False)
                state_dict.update(new_shard)
                _merge_experts_weight(state_dict, expert_num_per_shard,
                                      order_mapping)
            params_to_gather.append(param)
            param_names.append(name)
        if len(params_to_gather) > 0:
            args = (state_dict, prefix, {}, True, [], [], error_msgs)
            if is_deepspeed_zero3_enabled():
                import deepspeed
                with deepspeed.zero.GatheredParameters(
                        params_to_gather, modifier_rank=0):
                    if dist.get_rank() == 0:
                        module._load_from_state_dict(*args)
            else:
                module._load_from_state_dict(*args)

        for name in param_names:
            print_on_rank0(f'state_dict pop {name}')
            state_dict.pop(name)

        for name, child in module._modules.items():
            if child is not None:
                load(child, state_dict, unloaded_shard_files,
                     prefix + name + '.')

    state_dict = OrderedDict()
    load(model_to_load, state_dict, unloaded_shard_files, prefix='')
    print_on_rank0(f'{state_dict.keys()}')
    del state_dict

    return error_msgs


def _get_origin_param_name(merged_param_name, expert_num_per_shard, is_w1w3,
                           param_name_mapping):
    split_name = merged_param_name.split('.experts.')
    shard_idx = re.findall(r'\d+', split_name[1])[0]
    shard_idx = int(shard_idx)
    origin_param_names = [None] * (expert_num_per_shard * (1 + int(is_w1w3)))
    expert_idx_begin = expert_num_per_shard * shard_idx
    for i in range(expert_num_per_shard):
        if is_w1w3:
            gate_proj, up_proj = param_name_mapping[
                'gate_proj'], param_name_mapping['up_proj']
            gate = split_name[
                0] + f'.experts.{expert_idx_begin + i}.{gate_proj}.weight'
            up = split_name[
                0] + f'.experts.{expert_idx_begin + i}.{up_proj}.weight'
            origin_param_names[i * 2] = gate
            origin_param_names[i * 2 + 1] = up
        else:
            down_proj = param_name_mapping['down_proj']
            down = split_name[
                0] + f'.experts.{expert_idx_begin + i}.{down_proj}.weight'
            origin_param_names[i] = down
    return origin_param_names


def _split_param(merged_param, is_w1w3):
    if is_w1w3:
        expert_num, _, hidden_dim = merged_param.shape
        merged_param = merged_param.view(expert_num * 2, -1, hidden_dim)
        return torch.unbind(merged_param, dim=0)
    else:
        # (e, hidden_dim, ffn_dim)
        return torch.unbind(merged_param, dim=0)


def get_origin_state_dict(state_dict, model):

    model_name = type(model).__name__
    if model_name not in SUPPORT_MODELS:
        raise RuntimeError(
            f'Only models in {SUPPORT_MODELS} may need to convert state_dict '
            f'via `get_origin_state_dict` interface, but got {model_name}.')
    param_name_mapping = PARAM_NAME_MAPPING[model_name]

    expert_num_per_shard = get_expert_num_per_shard(model)
    experts_param_name = [
        name for name in state_dict.keys() if '.experts.' in name
    ]
    for expert_param_name in experts_param_name:
        print_on_rank0(f'processing {expert_param_name} ...')
        is_w1w3 = expert_param_name.split('.')[-1] == 'w1w3'
        origin_param_names = _get_origin_param_name(expert_param_name,
                                                    expert_num_per_shard,
                                                    is_w1w3,
                                                    param_name_mapping)
        merged_param = state_dict.pop(expert_param_name)
        origin_params = _split_param(merged_param, is_w1w3)
        assert len(origin_param_names) == len(origin_params)
        for name, param in zip(origin_param_names, origin_params):
            state_dict[name] = param
    return state_dict


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
        config = shard_model.config
        moe_implementation = getattr(config, 'moe_implementation', 'origin')
        use_ep = moe_implementation == 'ep'
        if use_ep:
            state_dict = get_origin_state_dict(state_dict, origin_model)
        # todo: fix origin_model.config, delete ep related
        print(f'Saving LLM to {ckpt_dir}')
        origin_model.save_pretrained(ckpt_dir, state_dict=state_dict)
        print(f'Saving LLM tokenizer to {ckpt_dir}')
        tokenizer.save_pretrained(ckpt_dir)

    barrier()
