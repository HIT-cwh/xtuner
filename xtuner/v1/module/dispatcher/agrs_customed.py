from typing import Literal, TypeAlias, cast
import os

import torch
from torch import Tensor
import torch.distributed as dist
from torch.autograd.function import Function
from torch.distributed._functional_collectives import (
    AsyncCollectiveTensor,
    all_gather_tensor,
    all_gather_tensor_autograd,
    reduce_scatter_tensor,
    reduce_scatter_tensor_autograd,
)
from typing_extensions import override

# from xtuner.v1.ops import _permute, _unpermute, _unpermute_inplace, _unpermute_bwd
from xtuner.v1.ops import unpermute, permute
from xtuner.v1.utils import copy_method_signature, get_device, get_logger

from . import XTUNER_DISPATCHER_DEBUG
from .base import (
    CombineResult,
    DispatchResult,
    GenericDispatcher,
    PostCombineResult,
    PostDispatchResult,
    PreCombineResult,
    PreDispatchResult,
)

import ib_wrapper
from ib_wrapper import ibReduceScatter
from ib_wrapper import ibgdaAllgather
from ib_wrapper import copy_no_cache

@torch._dynamo.disable
class SymmBufferManager:
    """
    A manager class for symmetric buffer allocation and lifecycle management.
    Optimizes buffer reuse and handles dynamic resizing based on communication requirements.
    Implements n buffering for concurrent operations with contiguous memory.
    """
    @torch._dynamo.disable
    def __init__(self, default_size=0, alignment=128, num_buffers=2):
        """
        Initialize the symmetric buffer manager with n buffering in contiguous memory.
        
        Args:
            default_size (int): Default buffer size in bytes
            alignment (int): Memory alignment requirement for the buffer
            num_buffers (int): Number of buffers for n-buffering
        """
        self.symm_buf_contiguous = None      # Contiguous memory block for all buffers
        self.symm_buf_ptrs = [None] * num_buffers  # Pointers to individual buffers within contiguous block
        self.current_buffer_idx = 0           # Index of current active buffer
        self.symm_buf_size = default_size     # Current configured buffer size
        self.alignment = alignment            # Memory alignment for optimal performance
        self._creation_count = 0              # Track how many times buffers have been created
        self.num_buffers = num_buffers        # Number of buffers for n-buffering
    @torch._dynamo.disable
    def get_buffer(self, bytes, device):
        """
        Get or create a symmetric buffer of appropriate size for the communication.
        Implements n buffering by cycling through multiple buffers in contiguous memory.
        
        Args:
            bytes (int): Number of bytes required for the current operation
            device: The device (GPU) where the buffer should be allocated
            
        Returns:
            The symmetric buffer object ready for use
        """
        # Calculate required size - use the larger of configured size or actual need
        required_size = max(self.symm_buf_size, bytes * self.num_buffers)
        
        # Case 1: No contiguous buffer exists - create initial contiguous buffer
        if self.symm_buf_contiguous is None:
            self._create_contiguous_buffer(required_size, device, "initial creation")
        
        # Case 2: Existing contiguous buffer is too small - recreate with larger size
        elif self.symm_buf_contiguous.numel() < required_size:
            print(f"Buffer resize required: {self.symm_buf_contiguous.numel()} -> {required_size}, {self.num_buffers = }")
            del self.symm_buf_contiguous
            self.symm_buf_ptrs = [None] * self.num_buffers
            self._create_contiguous_buffer(required_size, device, "resize due to insufficient size")
        
        # Case 3: Buffer exists and is large enough - reuse existing buffer
        # No action needed
        
        # Get the current buffer pointer
        current_buffer_ptr = self.symm_buf_ptrs[self.current_buffer_idx]
        
        # Shift to the next buffer for next call
        self._shift_buffer()
        
        return current_buffer_ptr
    @torch._dynamo.disable
    def _shift_buffer(self):
        """
        Shift to the next buffer in the n buffer system using round-robin.
        """
        self.current_buffer_idx = (self.current_buffer_idx + 1) % self.num_buffers
    @torch._dynamo.disable
    def _create_contiguous_buffer(self, size, device, reason):
        """
        Internal method to create a contiguous memory block for all buffers.
        
        Args:
            size (int): Total size of the contiguous buffer
            device: Target device for buffer allocation
            reason (str): Description of why the buffer is being created (for debugging)
        """
        print(f"{reason = }, {size/2**30 :.5f} GB")
        torch.cuda.synchronize()
        self.symm_buf_contiguous = ib_wrapper.create_symm_buffer(
            size, alignment=self.alignment, local_rank=device.index
        )

        self.symm_buf_contiguous.requires_grad_(False)
        
        # Calculate size per buffer with alignment
        size_per_buf = ((size // self.num_buffers) + self.alignment - 1) // self.alignment * self.alignment
        
        # Create pointers to each individual buffer
        for i in range(self.num_buffers):
            start_idx = i * size_per_buf
            end_idx = (i + 1) * size_per_buf
            # Ensure we don't exceed the total buffer size
            if end_idx > size:
                end_idx = size
            self.symm_buf_ptrs[i] = self.symm_buf_contiguous[start_idx:end_idx]
        
        self._creation_count += 1
        torch.cuda.synchronize()
    @torch._dynamo.disable
    def resize(self, new_size, device):
        """
        Explicitly resize all buffers to a new size.
        
        Args:
            new_size (int): New buffer size in bytes
            device: Target device for the resized buffer
        """
        if self.symm_buf_contiguous is not None:
            del self.symm_buf_contiguous
            self.symm_buf_contiguous = None
            self.symm_buf_ptrs = [None] * self.num_buffers
        
        self.symm_buf_size = new_size
        
        # Create new contiguous buffer with new size
        self._create_contiguous_buffer(new_size, device, "explicit resize")
    @torch._dynamo.disable
    def release(self):
        """Explicitly release all buffer resources."""
        if hasattr(self, 'symm_buf_contiguous') and self.symm_buf_contiguous is not None:
            del self.symm_buf_contiguous
            self.symm_buf_contiguous = None
            self.symm_buf_ptrs = [None] * self.num_buffers

    @torch._dynamo.disable
    def get_current_buffer_index(self):
        """
        Get the index of the current active buffer.
        
        Returns:
            int: Current buffer index (0 to num_buffers-1)
        """
        return self.current_buffer_idx
    @torch._dynamo.disable
    def get_stats(self):
        """
        Get statistics about buffer usage.
        
        Returns:
            dict: Buffer statistics including current sizes and creation count
        """
        if self.symm_buf_contiguous is not None:
            # Calculate individual buffer sizes
            size_per_buf = ((self.symm_buf_contiguous.numel() // self.num_buffers) + self.alignment - 1) // self.alignment * self.alignment
            current_sizes = [size_per_buf] * self.num_buffers
            # Adjust the last buffer size if needed
            total_allocated = size_per_buf * self.num_buffers
            if total_allocated > self.symm_buf_contiguous.numel():
                current_sizes[-1] = self.symm_buf_contiguous.numel() - size_per_buf * (self.num_buffers - 1)
        else:
            current_sizes = [0] * self.num_buffers
        
        return {
            'current_sizes': current_sizes,
            'contiguous_size': self.symm_buf_contiguous.numel() if self.symm_buf_contiguous is not None else 0,
            'current_buffer_index': self.current_buffer_idx,
            'configured_size': self.symm_buf_size,
            'creation_count': self._creation_count,
            'alignment': self.alignment,
            'num_buffers': self.num_buffers,
            'is_contiguous': self.symm_buf_contiguous is not None
        }
    @torch._dynamo.disable
    def __del__(self):
        """Destructor to ensure proper resource cleanup."""
        self.release()



class AllGatherIBManager:
    """
    Manager for ibgdaAllgather objects with double buffering support.
    Handles creation, caching, and rotation of communication buffers.
    """
    def __init__(self, comm_buf_size: int = 2, use_custom_ag: bool = False):
        self.comm_buf_size = comm_buf_size
        self.comm_buf_iter = 0
        self.use_custom_ag = use_custom_ag
        self.ag_ib_dict: dict[int, list] = {}

        # Distributed configuration
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_world_size = 8
        # assert torch.cuda.device_count() == 8

        self.local_rank = self.rank % self.local_world_size
        self.num_nodes = self.world_size // self.local_world_size
        self.node_id = self.rank // self.local_world_size

        self.vertical_ranks = [self.local_rank + nn * self.local_world_size for nn in range(self.num_nodes)]
        self.horizon_ranks = [lr + self.node_id * self.local_world_size for lr in range(self.local_world_size)]
        self.all_ranks = [wr for wr in range(self.world_size)]

        self.ranks = []
        self.fall_back2nccl = None

    @torch._dynamo.disable
    def get_allgather_objects(self, send_bytes, group, all_gather_stream):
        if self.ranks == []:
            self.ranks = dist.get_process_group_ranks(group)
        
        
        if self.fall_back2nccl == None:
            sorted_ranks = sorted(self.ranks)
            # Consolidate fallback conditions
            self.fall_back2nccl = (
                not self.use_custom_ag or
                (
                sorted_ranks != sorted(self.vertical_ranks) and 
                sorted_ranks != sorted(self.horizon_ranks) and 
                sorted_ranks != sorted(self.all_ranks))
            )

        if send_bytes not in self.ag_ib_dict and not self.fall_back2nccl:
            torch.cuda.synchronize()
                        
            # Create double buffered all-gather objects
            AGs = []
            if sorted(self.vertical_ranks) == sorted(self.ranks):
                for _ in range(self.comm_buf_size):
                    AGs.append(ibgdaAllgather(
                        send_bytes, group, all_gather_stream,
                        mode=1, barrier_all=True, vertical_group_ag=True
                    ))

            elif sorted(self.horizon_ranks) == sorted(self.ranks):
                for _ in range(self.comm_buf_size):
                    AGs.append(ibgdaAllgather(
                        send_bytes, group, all_gather_stream,
                        mode=0, barrier_all=False
                    ))
            elif sorted(self.all_ranks) == sorted(self.ranks):
                for _ in range(self.comm_buf_size):
                    AGs.append(ibgdaAllgather(
                        send_bytes, group, all_gather_stream,
                        mode=0, barrier_all=False
                    ))

            torch.cuda.synchronize()   
            self.ag_ib_dict[send_bytes] = AGs
    @torch._dynamo.disable
    def execute_allgather(self, send_bytes: int,
                         all_gather_output, all_gather_input, group):
        """
        Execute all-gather operation using cached ibgdaAllgather objects.
        
        Args:
            send_bytes: Size of the tensor in bytes
            group_size: Size of the process group
            world_size: Total world size
            all_gather_output: Output tensor for all-gather
            all_gather_input: Input tensor for all-gather
            group: Process group for the operation
        """
        if self.fall_back2nccl:
            dist.all_gather_into_tensor(
                all_gather_output, all_gather_input, group = group
            )
        else:
            # Use cached ibgdaAllgather objects
            self.ag_ib_dict[send_bytes][self.comm_buf_iter].all_gather_into_tensor(
                all_gather_output, all_gather_input, group = group
            )
            # Rotate buffer iterator
            self.comm_buf_iter = (self.comm_buf_iter + 1) % self.comm_buf_size

    @torch._dynamo.disable
    def clear_cache(self):
        """Clear all cached all-gather objects."""
        self.ag_ib_dict.clear()
        self.comm_buf_iter = 0

class AllGatherIBTensorAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ag_manager, send_bytes, all_gather_output, all_gather_input, group):
        
        ag_manager.execute_allgather(
                    send_bytes=send_bytes,
                    all_gather_output=all_gather_output,
                    all_gather_input=all_gather_input,
                    group=group
                )
        
        return all_gather_output

class ReduceScatterIBManager:
    """
    Manager for ibReduceScatter objects with double buffering support.
    Handles creation, caching, and execution of reduce-scatter operations.
    """
    @torch._dynamo.disable
    def __init__(self, comm_buf_size: int = 2, use_custom_rs: bool = False):
        self.comm_buf_size = comm_buf_size
        self.comm_buf_iter = 0
        self.use_custom_rs = use_custom_rs
        self.rs_ib_dict: dict[int, list] = {}
        self.rdc_scale: dict[int, torch.Tensor] = {}
        self.copy_stream = torch.cuda.Stream()
        self.copy_event_prev : torch.cuda.Event = None
        self.copy_event : torch.cuda.Event = None

        # Distributed configuration
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_world_size = 8
        # assert torch.cuda.device_count() == 8
        self.local_rank = self.rank % self.local_world_size
        self.num_nodes = self.world_size // self.local_world_size
        self.node_id = self.rank // self.local_world_size

        self.vertical_ranks = [self.local_rank + nn * self.local_world_size for nn in range(self.num_nodes)]
        self.horizon_ranks = [lr + self.node_id * self.local_world_size for lr in range(self.local_world_size)]
        self.all_ranks = [wr for wr in range(self.world_size)]

        self.ranks = []
        self.fall_back2nccl = None

    @torch._dynamo.disable
    def get_reducescatter_objects(self, 
                                    recv_bytes_aligned, 
                                    group,
                                    reduce_scatter_stream):
        if self.ranks == []:
            self.ranks = dist.get_process_group_ranks(group)
        
        
        if self.fall_back2nccl == None:
            sorted_ranks = sorted(self.ranks)
            # Consolidate fallback conditions
            self.fall_back2nccl = (
                not self.use_custom_rs or
                (
                sorted_ranks != sorted(self.vertical_ranks) and 
                sorted_ranks != sorted(self.horizon_ranks) and 
                sorted_ranks != sorted(self.all_ranks))
            )
            

        if recv_bytes_aligned not in self.rs_ib_dict and not self.fall_back2nccl:
            torch.cuda.synchronize()

            RSs = []
            if sorted(self.vertical_ranks) == sorted(self.ranks):
                for _ in range(self.comm_buf_size):
                    RSs.append(ibReduceScatter(
                        recv_bytes_aligned, group, reduce_scatter_stream,
                        mode=1, barrier_all=True, vertical_group_rs=True
                    ))
            elif sorted(self.horizon_ranks) == sorted(self.ranks):
                for _ in range(self.comm_buf_size):
                    RSs.append(ibReduceScatter(
                        recv_bytes_aligned, group, reduce_scatter_stream,
                        mode=0, barrier_all=False, vertical_group_rs=False
                    ))
            elif sorted(self.all_ranks) == sorted(self.ranks):
                RSs.append(ibReduceScatter(
                        recv_bytes_aligned, group, reduce_scatter_stream,
                        mode=0, barrier_all=False, vertical_group_rs=False
                    ))
              
            torch.cuda.synchronize()
            self.rs_ib_dict[recv_bytes_aligned] = RSs
            self.rdc_scale[recv_bytes_aligned] = None
    @torch._dynamo.disable
    def execute_reducescatter(self, 
        recv_bytes,
        reduce_scatter_input,
        reduce_scatter_group, 
        reduce_scatter_stream, 
        reduce_scatter_reduce_op,
    ):
        with torch.cuda.stream(reduce_scatter_stream):
            reduce_scatter_output_numel = recv_bytes // reduce_scatter_input.element_size()
            reduce_output = reduce_scatter_input.new_empty(
                (reduce_scatter_output_numel,)
            )

            if self.fall_back2nccl:
                dist.reduce_scatter_tensor(
                    output=reduce_output,
                    input=reduce_scatter_input,
                    group=reduce_scatter_group,
                    op=reduce_scatter_reduce_op,
                )
            else:
                # Execute reduce-scatter
                self.rs_ib_dict[recv_bytes][self.comm_buf_iter].reduce_scatter_tensor(
                    output=reduce_output,
                    input=reduce_scatter_input,
                    group=reduce_scatter_group,
                    op=reduce_scatter_reduce_op,
                )
                self.comm_buf_iter = (self.comm_buf_iter + 1) % self.comm_buf_size
                
        return reduce_output
    @torch._dynamo.disable
    def clear_cache(self):
        """Clear all cached reduce-scatter objects."""
        self.rs_ib_dict.clear()
        self.comm_buf_iter = 0

use_custom_rs = int(os.getenv("USE_CUSTOMED_RS", 0))
use_custom_ag = int(os.getenv("USE_CUSTOMED_AG", 0))

ag_symm = None
rs_symm = None
ag_manager = None
rs_manager = None


def copy_tensor_in_chunks(src_tensor, dst_tensor, chunk_size_gb=0.1):
    """
    分块拷贝张量数据，避免内存峰值
    
    Args:
        src_tensor: 源张量
        dst_tensor: 目标张量
        chunk_size_gb: 每个分块的大小(GB), 默认为0.1GB
    """
    if src_tensor.numel() != dst_tensor.numel():
        raise ValueError(f"张量元素数量不匹配: src={src_tensor.numel()}, dst={dst_tensor.numel()}")
    
    bytes_per_element = src_tensor.element_size()
    elements_per_chunk = int(chunk_size_gb * 1024 * 1024 * 1024) // bytes_per_element
    total_elements = src_tensor.numel()
    
    # 展平张量视图
    src_flat = src_tensor.view(-1)
    dst_flat = dst_tensor.view(-1)
    
    for start_idx in range(0, total_elements, elements_per_chunk):
        end_idx = min(start_idx + elements_per_chunk, total_elements)
        
        # 拷贝当前分块
        dst_flat[start_idx:end_idx].copy_(src_flat[start_idx:end_idx])



DEVICE = get_device()
logger = get_logger()


MoEAGRSHandle = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


# MoEAGRS handle include 6 tensor:
# (rank_prefix_matrix, channel_prefix_matrix, recv_channel_prefix_matrix, recv_src_idx, is_token_in_rank, send_head)
class MoEAGRSPreDispatchResult(PreDispatchResult):
    backward_previous_event: torch.cuda.Event | None
    forward_finished_event: torch.cuda.Event | None


class MoEAGRSDispatchResult(DispatchResult):
    topk_ids: torch.Tensor
    forward_finished_event: torch.cuda.Event | None
    backward_previous_event: torch.cuda.Event | None


class MoEAGRSPostDispatchResult(PostDispatchResult):
    row_ids_map: torch.Tensor


class MoEAGRSPreCombineResult(PreCombineResult):
    backward_previous_event: torch.cuda.Event | None
    forward_finished_event: torch.cuda.Event | None


class MoEAGRSCombineResult(CombineResult):
    forward_finished_event: torch.cuda.Event | None
    backward_previous_event: torch.cuda.Event | None


MoEAGRSPostCombineResult = PostCombineResult


HiddenStates: TypeAlias = torch.Tensor


def get_backward_pre_hook(backward_previous_event: torch.cuda.Event, name: str | None = None, debug: bool = False):
    def _backward_pre_hook(*_):
        # if name == "TorchAll2AllDispatcher.dispatch_preprocess":
        #     torch.cuda.synchronize()
        if debug:
            logger.info(f"[{name}] backward pre hook")
        if backward_previous_event is not None:
            torch.cuda.current_stream().wait_event(backward_previous_event)

    return _backward_pre_hook


def get_backward_hook(backward_finished_event: torch.cuda.Event, name: str | None = None, debug: bool = False):
    def _backward_hook(*_):
        if debug:
            logger.info(f"[{name}] backward hook")
        if backward_finished_event is not None:
            backward_finished_event.record()

    return _backward_hook


class _AsyncDispatch(Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        forward_previous_event: torch.cuda.Event,
        forward_finished_event: torch.cuda.Event,
        backward_previous_event: torch.cuda.Event,
        backward_finished_event: torch.cuda.Event,
        comm_stream: torch.cuda.Stream,
        process_group: dist.ProcessGroup,
    ):
        with torch.cuda.stream(comm_stream):
            comm_stream.wait_event(forward_previous_event)
            global ag_manager, use_custom_ag, ag_symm
            if use_custom_ag:
                if ag_symm == None:
                    ag_symm = SymmBufferManager(int(os.getenv("SYMM_BUF_SIZE", 0)), num_buffers=2)
                if ag_manager == None:
                    ag_manager = AllGatherIBManager(comm_buf_size=2, use_custom_ag=use_custom_ag)

                send_bytes = hidden_states.element_size() * hidden_states.numel()
                recv_bytes = send_bytes * process_group.size()
                recv_numel = hidden_states.numel() * process_group.size()

                ag_manager.get_allgather_objects(
                    send_bytes=send_bytes,
                    group=process_group,
                    all_gather_stream=comm_stream
                )
                device = hidden_states.device
                dtype = hidden_states.dtype
                combined_grad_out_symm = ag_symm.get_buffer(bytes=recv_bytes, device=device)
                combined_grad_out_symm = combined_grad_out_symm.view(dtype)[ : recv_numel]
                
                AllGatherIBTensorAutograd.apply(ag_manager, send_bytes, combined_grad_out_symm, hidden_states, process_group)
                # Need to copy from symmetric to private
                # dispatched_hidden_states = torch.empty_like(combined_grad_out_symm)
                # dispatched_hidden_states.copy_(combined_grad_out_symm)
                # copy_tensor_in_chunks(combined_grad_out_symm, dispatched_hidden_states, chunk_size_gb=0.05)

                # copy_no_cache(src = combined_grad_out_symm, dst = dispatched_hidden_states, stream = comm_stream, gridSize = 4, blockSize = 1024)
                dispatched_hidden_states = combined_grad_out_symm
                dispatched_hidden_states = dispatched_hidden_states.view(-1, *hidden_states.shape[1:])

            else:
                dispatched_hidden_states = all_gather_tensor_autograd(hidden_states, gather_dim=0, group=process_group)

            if isinstance(dispatched_hidden_states, AsyncCollectiveTensor):
                dispatched_hidden_states = dispatched_hidden_states.wait()

            # topk_ids (seq, topk)
            topk_ids = topk_ids.T.flatten()
            dispatched_topk_ids = torch.empty_like(topk_ids)
            dist.all_to_all_single(
                dispatched_topk_ids,
                topk_ids,
                group=process_group,
            )
            dispatched_topk_ids = dispatched_topk_ids.view(-1, 1)
            topk_weights = topk_weights.T.flatten()
            dispatched_topk_weights = torch.empty_like(topk_weights)
            dist.all_to_all_single(
                dispatched_topk_weights,
                topk_weights,
                group=process_group,
            )
            dispatched_topk_weights = dispatched_topk_weights.view(-1, 1)

            dispatched_hidden_states.record_stream(comm_stream)
            dispatched_topk_ids.record_stream(comm_stream)
            dispatched_topk_weights.record_stream(comm_stream)
            forward_finished_event.record(comm_stream)

        ctx.backward_previous_event = backward_previous_event
        ctx.backward_finished_event = backward_finished_event
        ctx.process_group = process_group
        ctx.comm_stream = comm_stream
        return dispatched_hidden_states, dispatched_topk_ids, dispatched_topk_weights

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor, *args
    ) -> tuple[torch.Tensor | None, None, None, None, None, None, None, None, None]:
        world_size = dist.get_world_size(group=ctx.process_group)
        if world_size == 1:
            return grad_output, None, None, None, None, None, None, None, None

        with torch.cuda.stream(ctx.comm_stream):
            if ctx.backward_previous_event is not None:
                ctx.comm_stream.wait_event(ctx.backward_previous_event)

            global rs_manager, use_custom_rs, rs_symm

            if use_custom_rs:

                # Initialize global managers if needed
                if rs_symm is None:
                    rs_symm = SymmBufferManager(int(os.getenv("SYMM_BUF_SIZE", 0)), num_buffers=1)
                if rs_manager is None:
                    rs_manager = ReduceScatterIBManager(comm_buf_size=2, use_custom_rs=use_custom_rs)

                process_group = ctx.process_group
                comm_stream = ctx.comm_stream
                
                send_bytes = grad_output.element_size() * grad_output.numel()
                recv_bytes = send_bytes // process_group.size()
                send_numel = grad_output.numel()
                recv_numel = send_numel // process_group.size()

                rs_manager.get_reducescatter_objects(
                    recv_bytes_aligned=recv_bytes,
                    group=process_group,
                    reduce_scatter_stream=comm_stream
                )
                device = grad_output.device
                dtype = grad_output.dtype
                symm_input = rs_symm.get_buffer(bytes=send_bytes, device=device)
                symm_input = symm_input.view(dtype)[ : send_numel]
                symm_input.copy_(grad_output.flatten())
                symm_input = symm_input.view(grad_output.shape)
                # ib_wrapper.barrier_node_on_stream(comm_stream)
                # torch.mul(grad_output.flatten(), 10**14, out = symm_input)

                combined_grad_output = rs_manager.execute_reducescatter(
                    recv_bytes=recv_bytes,
                    reduce_scatter_input=symm_input,
                    reduce_scatter_group=process_group,
                    reduce_scatter_stream=comm_stream,
                    reduce_scatter_reduce_op=dist.ReduceOp.SUM,
                )
                # combined_grad_output /= 10**14
                combined_grad_output = combined_grad_output.view(-1, *grad_output.shape[1:])

                # combined_grad_output_ = reduce_scatter_tensor(
                #     grad_output, reduceOp="sum", scatter_dim=0, group=ctx.process_group
                # )

                # if (combined_grad_output.device.index == 0):
                #     print(f"{combined_grad_output = }, {combined_grad_output_ = }")

                

            else:
                # print(f"{grad_output.mean() = }")
                combined_grad_output = reduce_scatter_tensor(
                    grad_output, reduceOp="sum", scatter_dim=0, group=ctx.process_group
                )
                

            grad_output.record_stream(ctx.comm_stream)
            combined_grad_output.record_stream(ctx.comm_stream)
            if ctx.backward_finished_event is not None:
                ctx.backward_finished_event.record(ctx.comm_stream)
        return combined_grad_output, None, None, None, None, None, None, None, None


_async_dispatch = copy_method_signature(_AsyncDispatch.forward)(_AsyncDispatch.apply)


class _AsyncCombine(Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        forward_previous_event: torch.cuda.Event,
        forward_finished_event: torch.cuda.Event,
        backward_previous_event: torch.cuda.Event,
        backward_finished_event: torch.cuda.Event,
        comm_stream: torch.cuda.Stream,
        process_group: dist.ProcessGroup,
    ):
        with torch.cuda.stream(comm_stream):
            comm_stream.wait_event(forward_previous_event)

            global rs_manager, use_custom_rs, rs_symm

            if use_custom_rs:

                if rs_symm == None:
                    rs_symm = SymmBufferManager(int(os.getenv("SYMM_BUF_SIZE", 0)), num_buffers=1)
                if rs_manager == None:
                    rs_manager = ReduceScatterIBManager(comm_buf_size=2, use_custom_rs=use_custom_rs)

                send_bytes = hidden_states.element_size() * hidden_states.numel()
                recv_bytes = send_bytes // process_group.size()
                send_numel = hidden_states.numel()
                recv_numel = send_numel // process_group.size()

                rs_manager.get_reducescatter_objects(
                    recv_bytes_aligned=recv_bytes,
                    group=process_group,
                    reduce_scatter_stream=comm_stream
                )
                device = hidden_states.device
                dtype = hidden_states.dtype
                symm_input = rs_symm.get_buffer(bytes=send_bytes, device=device)
                symm_input = symm_input.view(dtype)[ : send_numel]
                symm_input.copy_(hidden_states.flatten())
                symm_input = symm_input.view(hidden_states.shape)
                # ib_wrapper.barrier_node_on_stream(comm_stream)
                # torch.mul(hidden_states.flatten(), 10**6, out = symm_input)

                combined_hidden_states = rs_manager.execute_reducescatter(
                    recv_bytes=recv_bytes,
                    reduce_scatter_input=symm_input,
                    reduce_scatter_group=process_group,
                    reduce_scatter_stream=comm_stream,
                    reduce_scatter_reduce_op=dist.ReduceOp.SUM,
                )   
                # combined_hidden_states /= 10**6
                combined_hidden_states = combined_hidden_states.view(-1, *hidden_states.shape[1:])
                
            else:
                # print(f"{hidden_states.mean() = }")
                combined_hidden_states = reduce_scatter_tensor_autograd(
                    hidden_states, reduceOp="sum", scatter_dim=0, group=process_group
                )

            if isinstance(combined_hidden_states, AsyncCollectiveTensor):
                combined_hidden_states = combined_hidden_states.wait()

            forward_finished_event.record(comm_stream)

        ctx.comm_stream = comm_stream
        ctx.process_group = process_group

        ctx.backward_previous_event = backward_previous_event
        ctx.backward_finished_event = backward_finished_event
        return combined_hidden_states

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor, *args
    ) -> tuple[torch.Tensor | None, None, None, None, None, None, None]:
        world_size = dist.get_world_size(group=ctx.process_group)
        if world_size == 1:
            return grad_output, None, None, None, None, None, None

        with torch.cuda.stream(ctx.comm_stream):
            if ctx.backward_previous_event is not None:
                ctx.comm_stream.wait_event(ctx.backward_previous_event)

            global ag_manager, use_custom_ag, ag_symm

            if use_custom_ag:
                if ag_symm == None:
                    ag_symm = SymmBufferManager(int(os.getenv("SYMM_BUF_SIZE", 0)), num_buffers=2)
                if ag_manager == None:
                    ag_manager = AllGatherIBManager(comm_buf_size=2, use_custom_ag=use_custom_ag)

                send_bytes = grad_output.element_size() * grad_output.numel()
                recv_bytes = send_bytes * ctx.process_group.size()
                recv_numel = grad_output.numel() * ctx.process_group.size()
                comm_stream = ctx.comm_stream
                ag_manager.get_allgather_objects(
                    send_bytes=send_bytes,
                    group=ctx.process_group,
                    all_gather_stream=ctx.comm_stream
                )
                device = grad_output.device
                dtype = grad_output.dtype
                combined_grad_out_symm = ag_symm.get_buffer(bytes=recv_bytes, device=device)
                combined_grad_out_symm = combined_grad_out_symm.view(dtype)[ : recv_numel]
                

                ag_manager.execute_allgather(
                    send_bytes=send_bytes,
                    all_gather_output=combined_grad_out_symm,
                    all_gather_input=grad_output,
                    group=ctx.process_group
                )
                # Need to copy from symmetric to private
                # combined_grad_output = torch.empty_like(combined_grad_out_symm)
                # combined_grad_output.copy_(combined_grad_out_symm)
                # copy_tensor_in_chunks(combined_grad_out_symm, combined_grad_output, chunk_size_gb=0.05)

                # copy_no_cache(src = combined_grad_out_symm, dst = combined_grad_output, stream = comm_stream, gridSize = 4, blockSize = 1024)
                combined_grad_output = combined_grad_out_symm
                combined_grad_output = combined_grad_output.view(-1, *grad_output.shape[1:])
                
            else:
                combined_grad_output = all_gather_tensor(grad_output, gather_dim=0, group=ctx.process_group)

            grad_output.record_stream(ctx.comm_stream)
            combined_grad_output.record_stream(ctx.comm_stream)

            if ctx.backward_finished_event is not None:
                ctx.backward_finished_event.record(ctx.comm_stream)
        return combined_grad_output, None, None, None, None, None, None


_async_combine = copy_method_signature(_AsyncCombine.forward)(_AsyncCombine.apply)


class MoEAGRSDispatcher(
    GenericDispatcher[
        MoEAGRSPreDispatchResult,
        MoEAGRSDispatchResult,
        MoEAGRSPostDispatchResult,
        MoEAGRSPreCombineResult,
        MoEAGRSCombineResult,
        MoEAGRSPostCombineResult,
    ]
):
    _comm_stream = None
    _process_group: dist.ProcessGroup

    def __init__(
        self,
        *,
        n_routed_experts: int,
        process_group: torch.distributed.ProcessGroup,
        training_dtype: Literal["fp8", "bf16"] = "bf16",
        generate_dtype: Literal["fp8", "bf16"] = "bf16",
    ):
        super().__init__(
            n_routed_experts=n_routed_experts,
            process_group=process_group,
            training_dtype=training_dtype,
            generate_dtype=generate_dtype,
        )
        assert self._process_group is not None, (
            "Process group must be provided for `DeepEPDispatcher`. "
            "If you are training a MoE model, it means that `expert parallel` is not enabled in the config."
        )
        self._experts_per_rank = self._n_routed_experts // self._process_group.size()
        if MoEAGRSDispatcher._comm_stream is None:
            MoEAGRSDispatcher._comm_stream = cast(torch.cuda.Stream, torch.cuda.Stream(device=DEVICE))

    @override
    def dispatch_preprocess(
        self,
        *,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        async_op: bool = False,
    ) -> MoEAGRSPreDispatchResult:
        if async_op:
            forward_finished_event = cast(torch.cuda.Event, torch.cuda.Event())
            forward_finished_event.record()
            backward_previous_event = cast(torch.cuda.Event, torch.cuda.Event())

            if hidden_states.grad_fn is not None:
                hidden_states.grad_fn.register_prehook(
                    get_backward_pre_hook(
                        backward_previous_event=backward_previous_event,
                        name="AGRSDispatcher.dispatch_preprocess",
                        debug=XTUNER_DISPATCHER_DEBUG,
                    )
                )
        else:
            forward_finished_event = None
            backward_previous_event = None

        return MoEAGRSPreDispatchResult(
            hidden_states=hidden_states,
            topk_ids=topk_ids.to(torch.int64),
            backward_previous_event=backward_previous_event,
            forward_finished_event=forward_finished_event,
        )

    @override
    def dispatch(
        self,
        *,
        pre_dispatched: MoEAGRSPreDispatchResult,
        topk_weights: torch.Tensor,
        async_op: bool = False,
        decoding: bool = False,
    ) -> MoEAGRSDispatchResult:
        if not async_op:
            hidden_states = pre_dispatched["hidden_states"]
            topk_ids = pre_dispatched["topk_ids"]

            dispatched_hidden_states = all_gather_tensor_autograd(
                hidden_states, gather_dim=0, group=self._process_group
            )
            if isinstance(dispatched_hidden_states, AsyncCollectiveTensor):
                dispatched_hidden_states = dispatched_hidden_states.wait()

            # topk_ids (seq, topk)
            topk_ids = topk_ids.T.flatten()
            dispatched_topk_ids = torch.empty_like(topk_ids)
            dist.all_to_all_single(
                dispatched_topk_ids,
                topk_ids,
                group=self._process_group,
            )
            dispatched_topk_ids = dispatched_topk_ids.view(-1, 1)
            topk_weights = topk_weights.T.flatten()
            dispatched_topk_weights = torch.empty_like(topk_weights)
            dist.all_to_all_single(
                dispatched_topk_weights,
                topk_weights,
                group=self._process_group,
            )
            dispatched_topk_weights = dispatched_topk_weights.view(-1, 1)

            return MoEAGRSDispatchResult(
                hidden_states=cast(HiddenStates, dispatched_hidden_states),
                topk_weights=dispatched_topk_weights,
                topk_ids=dispatched_topk_ids,
                forward_finished_event=None,
                backward_previous_event=None,
            )
        else:
            forward_previous_event = pre_dispatched["forward_finished_event"]
            forward_finished_event = cast(torch.cuda.Event, torch.cuda.Event())
            backward_finished_event = pre_dispatched["backward_previous_event"]
            backward_previous_event = cast(torch.cuda.Event, torch.cuda.Event())

            dispatched_hidden_states, dispatched_topk_idx, dispatched_topk_weights = _async_dispatch(
                pre_dispatched["hidden_states"],
                pre_dispatched["topk_ids"],
                topk_weights,
                forward_previous_event,
                forward_finished_event,
                backward_previous_event,
                backward_finished_event,
                self._comm_stream,
                self._process_group,
            )
            return MoEAGRSDispatchResult(
                hidden_states=cast(HiddenStates, dispatched_hidden_states),
                topk_weights=dispatched_topk_weights,
                topk_ids=dispatched_topk_idx,
                forward_finished_event=forward_finished_event,
                backward_previous_event=backward_previous_event,
            )

    @override
    def dispatch_postprocess(
        self,
        *,
        pre_dispatched: MoEAGRSPreDispatchResult,
        dispatched: MoEAGRSDispatchResult,
        async_op: bool = False,
        decoding: bool = False,
    ) -> MoEAGRSPostDispatchResult:
        if async_op:
            assert dispatched["forward_finished_event"] is not None, "Please use `async_op=True` for dispatch!"
            self.wait_comm_stream(dispatched["forward_finished_event"])

        permuted_hidden_states, row_ids_map = permute(
            dispatched["hidden_states"],
            dispatched["topk_ids"].to(torch.int32),
        )

        topk_ids = dispatched["topk_ids"]
        rank = dist.get_rank(group=self._process_group)
        tokens_per_expert = torch.histc(
            topk_ids,
            bins=self._experts_per_rank,
            min=rank * self._experts_per_rank,
            max=(rank + 1) * self._experts_per_rank,
        )

        if async_op:
            assert dispatched["backward_previous_event"] is not None, "Please use `async_op=True` for dispatch!"
            if permuted_hidden_states.grad_fn is not None:
                permuted_hidden_states.grad_fn.register_hook(
                    get_backward_hook(
                        dispatched["backward_previous_event"],
                        name="TorchAll2AllDispatcher.dispatch_posAGRSrocess",
                        debug=XTUNER_DISPATCHER_DEBUG,
                    )
                )

        return MoEAGRSPostDispatchResult(
            hidden_states=permuted_hidden_states,
            row_ids_map=row_ids_map,
            tokens_per_expert=tokens_per_expert,
        )

    @override
    def combine_preprocess(
        self,
        *,
        hidden_states: torch.Tensor,
        pre_dispatched: MoEAGRSPreDispatchResult,
        dispatched: MoEAGRSDispatchResult,
        post_dispatched: MoEAGRSPostDispatchResult,
        async_op: bool = False,
        decoding: bool = False,
    ) -> MoEAGRSPreCombineResult:
        hidden_states = unpermute(
            hidden_states,
            post_dispatched["row_ids_map"],
            probs=dispatched["topk_weights"],
        )

        if async_op:
            backward_previous_event = cast(torch.cuda.Event, torch.cuda.Event())
            forward_finished_event = cast(torch.cuda.Event, torch.cuda.Event())
            forward_finished_event.record()
            if hidden_states.grad_fn is not None:
                hidden_states.grad_fn.register_prehook(
                    get_backward_pre_hook(
                        backward_previous_event=backward_previous_event,
                        name="TorchAll2AllDispatcher.combine_preprocess",
                        debug=XTUNER_DISPATCHER_DEBUG,
                    )
                )
        else:
            forward_finished_event = None
            backward_previous_event = None

        return MoEAGRSPreCombineResult(
            hidden_states=hidden_states,
            forward_finished_event=forward_finished_event,
            backward_previous_event=backward_previous_event,
        )

    @override
    def combine(
        self,
        *,
        pre_dispatched: MoEAGRSPreDispatchResult,
        dispatched: MoEAGRSDispatchResult,
        post_dispatched: MoEAGRSPostDispatchResult,
        pre_combined: MoEAGRSPreCombineResult,
        async_op: bool = False,
        decoding: bool = False,
    ) -> CombineResult:
        if async_op:
            forward_previous_event = pre_combined["forward_finished_event"]
            forward_finished_event = cast(torch.cuda.Event, torch.cuda.Event())
            backward_previous_event = cast(torch.cuda.Event, torch.cuda.Event())
            backward_finished_event = pre_combined["backward_previous_event"]
            assert forward_previous_event is not None, "Please use `async_op=True` for combine_preprocess!"
            assert backward_finished_event is not None, "Please use `async_op=True` for combine_preprocess!"

            combined_hidden_states = _async_combine(
                pre_combined["hidden_states"],
                forward_previous_event,
                forward_finished_event,
                backward_previous_event,
                backward_finished_event,
                self._comm_stream,
                self._process_group,
            )
        else:
            forward_finished_event = None
            backward_previous_event = None
            hidden_states = pre_combined["hidden_states"]  # .float()

            assert 0, "Not supported for non async op"
            combined_hidden_states = reduce_scatter_tensor_autograd(
                hidden_states, reduceOp="sum", scatter_dim=0, group=self._process_group
            )
            if isinstance(combined_hidden_states, AsyncCollectiveTensor):
                combined_hidden_states = combined_hidden_states.wait()
            # combined_hidden_states = combined_hidden_states.bfloat16()

        return MoEAGRSCombineResult(
            hidden_states=combined_hidden_states,
            forward_finished_event=forward_finished_event,
            backward_previous_event=backward_previous_event,
        )

    @override
    def combine_postprocess(
        self,
        *,
        pre_dispatched: MoEAGRSPreDispatchResult,
        dispatched: MoEAGRSDispatchResult,
        post_dispatched: MoEAGRSPostDispatchResult,
        pre_combined: MoEAGRSPreCombineResult,
        combined: MoEAGRSCombineResult,
        async_op: bool = False,
    ) -> PostCombineResult:
        hidden_states = combined["hidden_states"]

        if async_op:
            forward_previous_event = combined["forward_finished_event"]
            backward_finished_event = combined["backward_previous_event"]
            assert forward_previous_event is not None, "Please use `async_op=True` for combine!"
            assert backward_finished_event is not None, "Please use `async_op=True` for combine!"
            self.wait_comm_stream(forward_previous_event)

            hidden_states = hidden_states.view_as(hidden_states)

            if hidden_states.grad_fn is not None:
                hidden_states.grad_fn.register_hook(
                    get_backward_hook(
                        backward_finished_event=cast(torch.cuda.Event, combined["backward_previous_event"]),
                        name="DeeEPDispatcher.combine_postprocess",
                        debug=XTUNER_DISPATCHER_DEBUG,
                    )
                )

        return PostCombineResult(hidden_states=hidden_states)

    def wait_comm_stream(self, event: torch.cuda.Event):
        torch.cuda.current_stream().wait_event(event)
