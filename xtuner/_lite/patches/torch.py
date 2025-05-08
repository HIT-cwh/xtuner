# Copyright (c) OpenMMLab. All rights reserved.
import inspect
from functools import wraps
from typing import List, cast, Optional, Union, Tuple
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
        module = import_modules_from_strings("torch.distributed.fsdp._fully_shard._fsdp_param")
        logger.info("dispatch_torch_fsdp_chunk_with_empty")
        if hasattr(module, "_chunk_with_empty"):
            module._chunk_with_empty = chunk_with_empty


from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam
import torch.distributed as dist
from torch.distributed.fsdp._fully_shard._fsdp_common import _raise_assert_with_print, _to_dtype_if_needed, compiled_autograd_enabled
from torch.distributed.fsdp._fully_shard._fsdp_collectives import _get_gradient_divide_factors, _div_if_needed
from torch.distributed.device_mesh import _get_device_handle
import math
from torch.distributed.distributed_c10d import ReduceOp
from torch.distributed.tensor import DTensor


def get_chunk_size(tensor_size, num_chunks):
    total_size = tensor_size[0]
    base_size = 128
    ideal_chunk_size = math.ceil(total_size / num_chunks)
    if ideal_chunk_size > base_size:
        # 如果大于base_size，调整到base_size的倍数
        chunk_size = math.ceil(ideal_chunk_size / base_size) * base_size
    else:
        # 如果小于base_size，找到最大的能整除base_size的数
        factors = [1, 2, 4, 8, 16, 32, 64, 128]
        chunk_size = next(size for size in factors if size >= ideal_chunk_size)
    return chunk_size


# modified
def get_dim0_padded_size(tensor_size: torch.Size, dim0_factor: int) -> torch.Size:
    chunk_size = get_chunk_size(tensor_size, dim0_factor)
    padded_dim0 = chunk_size * dim0_factor
    return cast(torch.Size, torch.Size([padded_dim0]) + tensor_size[1:])


@torch.compile(fullgraph=True)
def foreach_reduce_scatter_copy_in(tensors, num_chunks):
    outs = []
    for tensor in tensors:
        chunk_size = get_chunk_size(tensor.size(), num_chunks)
        trailing_numel = tensor.size(1) if tensor.ndim > 1 else 1
        out_cur = tensors[0].new_empty((num_chunks, chunk_size * trailing_numel))
        # breakpoint()
        out_cur.view(-1)[:tensor.numel()] = tensor.view(-1)
        outs.append(out_cur.view(num_chunks, chunk_size * trailing_numel))
    return torch.cat(outs, dim=1)


@torch.no_grad()
def foreach_reduce(
    fsdp_params: List[FSDPParam],
    unsharded_grads: List[torch.Tensor],
    reduce_scatter_group: dist.ProcessGroup,
    reduce_scatter_stream: torch.Stream,
    orig_dtype: torch.dtype,
    reduce_dtype: Optional[torch.dtype],
    device: torch.device,
    reduce_scatter_reduce_op: Optional[Union[dist.ReduceOp, dist.ReduceOp.RedOpType]],
    all_reduce_group: Optional[dist.ProcessGroup],  # not `None` iff HSDP
    all_reduce_stream: torch.Stream,
    all_reduce_grads: bool,
    partial_reduce_output: Optional[torch.Tensor],  # only used for HSDP
) -> Tuple[
    torch.Tensor,
    torch.Event,
    torch.Event,
    Optional[torch.Tensor],
    Optional[torch.Event],
    Optional[torch.Tensor],
]:
    """
    ``unsharded_grads`` owns the references to the gradients computed by
    autograd, so clearing the list frees the gradients.
    """
    grad_dtypes = {grad.dtype for grad in unsharded_grads}
    if len(grad_dtypes) != 1:
        # Check this at runtime since it could be a real runtime error if e.g.
        # fp8 weights do not produce the correct higher precision gradients
        _raise_assert_with_print(
            f"FSDP reduce-scatter expects uniform gradient dtype but got {grad_dtypes}"
        )
    grad_dtype = unsharded_grads[0].dtype
    reduce_dtype = reduce_dtype or grad_dtype
    predivide_factor, postdivide_factor = _get_gradient_divide_factors(
        reduce_scatter_group, all_reduce_group, reduce_dtype
    )
    world_size = reduce_scatter_group.size()
    for i, (fsdp_param, unsharded_grad) in enumerate(zip(fsdp_params, unsharded_grads)):
        if (shard_dim := fsdp_param.fsdp_placement.dim) == 0:
            continue
        assert (
            unsharded_grad.size(shard_dim) % world_size == 0
        ), f"Shard({shard_dim}) requires even sharding: {unsharded_grad.size()=} {world_size=}"
        chunks = torch.chunk(unsharded_grad, world_size, dim=shard_dim)
        unsharded_grads[i] = torch.cat(chunks, dim=0)
    # modified
    # padded_unsharded_sizes = tuple(
    #     _get_dim0_padded_size(grad.size(), world_size) for grad in unsharded_grads
    # )
    padded_unsharded_sizes = tuple(
        get_dim0_padded_size(grad.size(), world_size) for grad in unsharded_grads
    )
    # print(f'rank {dist.get_rank()} {padded_unsharded_sizes}')
    reduce_scatter_input_numel = sum(s.numel() for s in padded_unsharded_sizes)
    reduce_scatter_output_numel = reduce_scatter_input_numel // world_size
    # modified
    # reduce_scatter_input = torch.empty(
    #     (reduce_scatter_input_numel,), dtype=reduce_dtype, device=device
    # )
    device_handle = _get_device_handle(device.type)
    # modified
    # foreach_reduce_scatter_copy_in(unsharded_grads, reduce_scatter_input, world_size)
    reduce_scatter_input = foreach_reduce_scatter_copy_in(unsharded_grads, world_size)
    current_stream = device_handle.current_stream()
    # Only after the copy-in finishes can we free the gradients
    unsharded_grads.clear()
    reduce_scatter_stream.wait_stream(current_stream)
    all_reduce_input = None
    all_reduce_event = None
    with device_handle.stream(reduce_scatter_stream):
        reduce_output = reduce_scatter_input.new_empty((reduce_scatter_output_numel,))
        _div_if_needed(reduce_scatter_input, predivide_factor)
        if reduce_scatter_reduce_op is None:
            if predivide_factor is None:
                reduce_scatter_reduce_op = ReduceOp.AVG
            else:
                reduce_scatter_reduce_op = ReduceOp.SUM
        dist.reduce_scatter_tensor(
            output=reduce_output,
            input=reduce_scatter_input,
            group=reduce_scatter_group,
            op=reduce_scatter_reduce_op,
        )
        reduce_scatter_event = reduce_scatter_stream.record_event()
        post_reduce_stream = reduce_scatter_stream
        if all_reduce_group is not None:  # HSDP
            # Accumulations must run in the reduce-scatter stream
            if not all_reduce_grads:
                if partial_reduce_output is not None:
                    partial_reduce_output += reduce_output
                else:
                    partial_reduce_output = reduce_output
                return (
                    reduce_scatter_input,
                    reduce_scatter_event,
                    post_reduce_stream.record_event(),
                    all_reduce_input,
                    all_reduce_event,
                    partial_reduce_output,
                )
            if partial_reduce_output is not None:
                reduce_output += partial_reduce_output
            post_reduce_stream = all_reduce_stream
            all_reduce_stream.wait_stream(reduce_scatter_stream)
            with device_handle.stream(all_reduce_stream):
                dist.all_reduce(
                    reduce_output,
                    group=all_reduce_group,
                    op=ReduceOp.AVG if predivide_factor is None else ReduceOp.SUM,
                )
                all_reduce_input = reduce_output
                all_reduce_event = all_reduce_stream.record_event()
    with device_handle.stream(post_reduce_stream):
        _div_if_needed(reduce_output, postdivide_factor)
        reduce_output = _to_dtype_if_needed(reduce_output, orig_dtype)
        # View out and accumulate sharded gradients
        flat_grad_offset = 0  # [0, reduce_scatter_output_numel - 1]
        for padded_unsharded_size, fsdp_param in zip(
            padded_unsharded_sizes, fsdp_params
        ):
            # Assume even sharding for Shard(i), i > 0; otherwise would require
            # copy-out for contiguous strides
            new_sharded_grad = torch.as_strided(
                reduce_output,
                size=fsdp_param.sharded_size,
                stride=fsdp_param.contiguous_sharded_stride,
                storage_offset=flat_grad_offset,
            )
            to_accumulate_grad = fsdp_param.sharded_param.grad is not None
            if fsdp_param.offload_to_cpu:
                # Only overlap the D2H copy (copying to pinned memory) if not
                # accumulating gradients since the CPU add kernel depends on
                # the copy result and we cannot run the add as a callback
                non_blocking = fsdp_param.pin_memory and not to_accumulate_grad
                # Since the GPU sharded gradient is allocated in the RS stream,
                # we can free it here by not keeping a ref without waiting for
                # the D2H copy since future RS-stream ops run after the copy
                new_sharded_grad = new_sharded_grad.to(
                    torch.device("cpu"), non_blocking=non_blocking
                )
                if non_blocking:
                    # Record an event on which to block the CPU thread to
                    # ensure that the D2H copy finishes before the optimizer
                    fsdp_param.grad_offload_event = reduce_scatter_stream.record_event()
            if to_accumulate_grad:
                assert isinstance(fsdp_param.sharded_param.grad, DTensor)
                fsdp_param.sharded_param.grad._local_tensor += new_sharded_grad
            else:
                new_sharded_dtensor_grad = fsdp_param.to_sharded_dtensor(
                    new_sharded_grad
                )
                fsdp_param.sharded_param.grad = new_sharded_dtensor_grad
            if not compiled_autograd_enabled():
                for hook in (
                    getattr(fsdp_param.sharded_param, "_post_accumulate_grad_hooks", {})
                    or {}
                ).values():
                    hook(fsdp_param.sharded_param)
            padded_sharded_numel = padded_unsharded_size.numel() // world_size
            flat_grad_offset += padded_sharded_numel
        post_reduce_event = post_reduce_stream.record_event()
    # The RS output is allocated in the RS stream and used in the default
    # stream (for optimizer). To ensure its memory is not reused for later
    # RSs, we do not need extra synchronization since the sharded parameters
    # hold refs through the end of backward.
    return (
        reduce_scatter_input,
        reduce_scatter_event,
        post_reduce_event,
        all_reduce_input,
        all_reduce_event,
        None,
    )


def dispatch_torch_fsdp_foreach_reduce():
    # 修改了 chunk_with_empty 之后，fsdp 不再按照原有的 torch.chunk 的方式切参数
    # reduce_scatter fsdp copy_in 的时候也就不能再用 torch._chunk_cat 
    if digit_version(torch.__version__)[:2] == (2, 6):
        module = import_modules_from_strings("torch.distributed.fsdp._fully_shard._fsdp_param_group")
        if hasattr(module, "foreach_reduce"):
            logger.info("dispatch_torch_fsdp_foreach_reduce")
            module.foreach_reduce = foreach_reduce
