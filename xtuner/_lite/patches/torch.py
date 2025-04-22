# Copyright (c) OpenMMLab. All rights reserved.
import inspect
from functools import wraps
from typing import List, cast
import math

import torch
from mmengine.utils import digit_version, import_modules_from_strings

from xtuner._lite import get_logger

logger = get_logger()


def replace_partition_fn(func):
    from functorch.compile import default_partition

    @wraps(func)
    def wrapper(**kwargs):
        if "partition_fn" in kwargs:
            kwargs["partition_fn"] = default_partition
        return func(**kwargs)

    return wrapper


def dispatch_torch_compile():
    if digit_version(torch.__version__)[:2] == (2, 6):
        logger.info("dispatch_torch_compile")
        module = import_modules_from_strings("torch._inductor.compile_fx")
        if hasattr(module, "aot_autograd"):
            module.aot_autograd = replace_partition_fn(module.aot_autograd)


def all_gather_inputs(self) -> List[torch.Tensor]:  # 1D
    from torch.distributed.fsdp._fully_shard._fsdp_common import (
        _to_dtype_if_needed,
        compiled_autograd_enabled,
    )
    from torch.distributed.fsdp._fully_shard._fsdp_param import ShardedState

    self._assert_in_states(ShardedState.SHARDED, ShardedState.SHARDED_POST_FORWARD)
    if self.sharded_state == ShardedState.SHARDED:
        if not compiled_autograd_enabled() and hasattr(
            self._sharded_local_tensor, "fsdp_pre_all_gather"
        ):
            # ------------------- modified --------------------#
            if getattr(
                self._sharded_local_tensor,
                "_use_padded_sharded_param_all_gather",
                False,
            ):
                sharded_local_tensor = self._sharded_param_data
                if hasattr(
                    self._sharded_local_tensor, "_precomputed_scale"
                ) and hasattr(sharded_local_tensor, "_precomputed_scale"):
                    sharded_local_tensor._precomputed_scale = (
                        self._sharded_local_tensor._precomputed_scale
                    )
            else:
                sharded_local_tensor = self._sharded_local_tensor
            # ---------------------------------------------------#
            if self.offload_to_cpu:
                sharded_local_tensor = sharded_local_tensor.to(
                    self.device, non_blocking=True
                )
            pre_all_gather_signature = inspect.signature(
                sharded_local_tensor.fsdp_pre_all_gather
            )
            num_fn_params = len(pre_all_gather_signature.parameters)
            # Old signature only passes mesh; keep for BC for now
            assert num_fn_params in (
                1,
                5,
            ), (
                f"Invalid fsdp_pre_all_gather: {pre_all_gather_signature}\n"
                "Expects fsdp_pre_all_gather(self, mesh: DeviceMesh, "
                "module: nn.Module, mp_policy: MixedPrecisionPolicy)"
            )
            if num_fn_params == 1:
                (
                    all_gather_inputs,
                    self._extensions_data.all_gather_metadata,
                ) = sharded_local_tensor.fsdp_pre_all_gather(self.shard_mesh)
            else:
                (
                    all_gather_inputs,
                    self._extensions_data.all_gather_metadata,
                ) = sharded_local_tensor.fsdp_pre_all_gather(
                    self.shard_mesh,
                    self._orig_size,
                    self._contiguous_orig_stride,
                    self._module_info.module,
                    self.mp_policy,
                )
                if (
                    sharded_local_tensor.size() != self.padded_sharded_param_size
                    and any(
                        all_gather_input.size() != self.padded_sharded_param_size
                        for all_gather_input in all_gather_inputs
                    )
                ):
                    # NOTE: Since this error can only be raised on the
                    # ranks that have padding, this can manifest as a NCCL
                    # watchdog timeout, as the other ranks will not error.
                    raise AssertionError(
                        "When a parameter is unevenly sharded by FSDP "
                        f"(orig size={self._orig_size}, FSDP world size={self.mesh_info.mesh.size()}), "
                        "fsdp_pre_all_gather must return all-gather inputs with the padded sharded size "
                        f"{self.padded_sharded_param_size} but got {[t.size() for t in all_gather_inputs]}"
                    )
            self._extensions_data.all_gather_input_sizes = [
                t.size() for t in all_gather_inputs
            ]
            return [t.view(-1) for t in all_gather_inputs]
        sharded_param_data = self._sharded_param_data
        if self.offload_to_cpu:
            sharded_param_data = sharded_param_data.to(self.device, non_blocking=True)
        return [_to_dtype_if_needed(sharded_param_data, self.param_dtype)]
    elif self.sharded_state == ShardedState.SHARDED_POST_FORWARD:
        if not compiled_autograd_enabled() and hasattr(
            self._sharded_local_tensor, "fsdp_pre_all_gather"
        ):
            raise NotImplementedError
        all_gather_input = _to_dtype_if_needed(
            cast(torch.Tensor, self._sharded_post_forward_param_data),
            self.param_dtype,
        )
        return [all_gather_input]
    return [torch.empty(0)]  # mypy


def dispatch_torch_fsdp_param():
    # support cases where param.numel() is not evenly divided by num_gpus
    if digit_version(torch.__version__)[:2] == (2, 6):
        from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam

        logger.info("dispatch_torch_fsdp_param")
        FSDPParam.all_gather_inputs = property(all_gather_inputs)


def chunk_with_empty(
    tensor: torch.Tensor, num_chunks: int, dim: int
) -> List[torch.Tensor]:
    total_size = tensor.size(dim)
    base_size = 128  # fp8 block_size is 128

    ideal_chunk_size = math.ceil(total_size / num_chunks)

    if ideal_chunk_size > base_size:
        # 如果大于base_size，调整到base_size的倍数
        chunk_size = math.ceil(ideal_chunk_size / base_size) * base_size
    else:
        # 如果小于base_size，找到最大的能整除base_size的数
        factors = [1, 2, 4, 8, 16, 32, 64, 128]
        chunk_size = next(size for size in factors if size >= ideal_chunk_size)
    
    chunks = torch.split(tensor, chunk_size, dim=dim)
    chunks = list(chunks)
    while len(chunks) < num_chunks:
        chunks.append(chunks[0].new_empty(0))
    return chunks


def dispatch_torch_fsdp_chunk_with_empty():
    # fsdp 切完参数后，shard dim 的长度如果不是 128 的倍数也不是 128 的因数
    # 很难对其使用 per block fp8 量化
    # 需要保证 fsdp 切完参数后，shard dim 的长度如果是 128 的倍数或 128 的因数
    # 在 pt26 , torch/distributed/fsdp/_fully_shard/_fsdp_common.py 中的 _chunk_with_empty
    # 只在 torch/distributed/fsdp/_fully_shard/_fsdp_param.py 中用过两次：
    # _init_sharded_param 和 _init_sharded_post_forward_param_metadata
    # **在支持其他版本时务必重点检查 _chunk_with_empty 这个函数的使用情况**
    if digit_version(torch.__version__)[:2] == (2, 6):
        logger.info("dispatch_torch_fsdp_chunk_with_empty")
        module = import_modules_from_strings("torch.distributed.fsdp._fully_shard._fsdp_common")
        if hasattr(module, "_chunk_with_empty"):
            module._chunk_with_empty = chunk_with_empty
