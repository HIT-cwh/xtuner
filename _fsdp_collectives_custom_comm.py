      
# mypy: allow-untyped-decorators
from typing import cast, List, NamedTuple, Optional, Tuple, Union, Dict
import os
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import _get_device_handle
from torch.distributed.distributed_c10d import ReduceOp
from torch.distributed.tensor import DTensor

from ._fsdp_common import (
    _get_dim0_padded_size,
    _raise_assert_with_print,
    _to_dtype_if_needed,
    compiled_autograd_enabled,
)
from ._fsdp_param import FSDPParam, ShardedState

import ib_wrapper
from ib_wrapper import ibReduceScatter
from ib_wrapper import ibgdaAllgather


ag_event: torch.cuda.Event = None
rs_event: torch.cuda.Event = None

tmp = 0

class SymmBufferManager:
    """
    A manager class for symmetric buffer allocation and lifecycle management.
    Optimizes buffer reuse and handles dynamic resizing based on communication requirements.
    Implements n buffering for concurrent operations with contiguous memory.
    """
    
    def __init__(self, default_size=0, alignment=128, num_buffers=3):
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
            print(f"Buffer resize required: {self.symm_buf_contiguous.numel()} -> {required_size}")
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
    
    def _shift_buffer(self):
        """
        Shift to the next buffer in the n buffer system using round-robin.
        """
        self.current_buffer_idx = (self.current_buffer_idx + 1) % self.num_buffers
    
    def _create_contiguous_buffer(self, size, device, reason):
        """
        Internal method to create a contiguous memory block for all buffers.
        
        Args:
            size (int): Total size of the contiguous buffer
            device: Target device for buffer allocation
            reason (str): Description of why the buffer is being created (for debugging)
        """
        print(f"{reason = }, {size/2**30 :.1f} GB")
        self.symm_buf_contiguous = ib_wrapper.create_symm_buffer(
            size, alignment=self.alignment, local_rank=device.index
        )
        
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
    
    def release(self):
        """Explicitly release all buffer resources."""
        if self.symm_buf_contiguous is not None:
            del self.symm_buf_contiguous
            self.symm_buf_contiguous = None
            self.symm_buf_ptrs = [None] * self.num_buffers
    
    def get_current_buffer_index(self):
        """
        Get the index of the current active buffer.
        
        Returns:
            int: Current buffer index (0 to num_buffers-1)
        """
        return self.current_buffer_idx
    
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
    
    def __del__(self):
        """Destructor to ensure proper resource cleanup."""
        self.release()

class AllGatherIBManager:
    """
    Manager for ibgdaAllgather objects with double buffering support.
    Handles creation, caching, and rotation of communication buffers.
    """
    
    def __init__(self, num_buffers: int = 2, use_custom_ag: bool = False):
        self.num_buffers = num_buffers
        self.comm_buf_iter = 0
        self.use_custom_ag = use_custom_ag
        self.ag_ib_dict: dict[int, list] = {}
    
    def get_allgather_objects(self, send_bytes: int, group_size: int, world_size: int, 
                            device_count: int, all_gather_stream, mode: int = 0):
        """
        Get or create ibgdaAllgather objects for the given byte size and group configuration.
        
        Args:
            send_bytes: Size of the tensor in bytes
            group_size: Size of the process group
            world_size: Total world size
            device_count: Number of CUDA devices per node
            all_gather_stream: CUDA stream for all-gather operations
            mode: Operation mode for ibgdaAllgather
        """
        if send_bytes not in self.ag_ib_dict and self.use_custom_ag:
            torch.cuda.synchronize()
            # Determine if this is a full world or EP group
            is_full_world = group_size == world_size
            is_ep_group = group_size == world_size // device_count
            
            vertical_group_ag = is_ep_group      # Use local rank for EP groups
            
            # Create double buffered all-gather objects
            AGs = []
            if vertical_group_ag:
                for _ in range(self.num_buffers):
                    AGs.append(ibgdaAllgather(
                        send_bytes, torch.distributed.group.WORLD, all_gather_stream,
                        mode=1, barrier_all=True, vertical_group_ag=vertical_group_ag
                    ))

            else:
                for _ in range(self.num_buffers):
                    AGs.append(ibgdaAllgather(
                        send_bytes, torch.distributed.group.WORLD, all_gather_stream,
                        mode=0, barrier_all=True, vertical_group_ag=vertical_group_ag
                    ))
            torch.cuda.synchronize()   
            self.ag_ib_dict[send_bytes] = AGs
    
    def execute_allgather(self, send_bytes: int, group_size: int, world_size: int,
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
        if self.use_custom_ag:
            if group_size == world_size or group_size == world_size // torch.cuda.device_count():
                # Use cached ibgdaAllgather objects
                self.ag_ib_dict[send_bytes][self.comm_buf_iter].all_gather_into_tensor(
                    all_gather_output, all_gather_input, group = group
                )
                # Rotate buffer iterator
                self.comm_buf_iter = (self.comm_buf_iter + 1) % self.num_buffers

    
    def clear_cache(self):
        """Clear all cached all-gather objects."""
        self.ag_ib_dict.clear()
        self.comm_buf_iter = 0

class ReduceScatterIBManager:
    """
    Manager for ibReduceScatter objects with double buffering support.
    Handles creation, caching, and execution of reduce-scatter operations.
    """
    
    def __init__(self, num_buffers: int = 2, use_custom_rs: bool = False):
        self.num_buffers = num_buffers
        self.comm_buf_iter = 0
        self.use_custom_rs = use_custom_rs
        self.rs_ib_dict: dict[int, list] = {}
        self.rdc_scale: dict[int, torch.Tensor] = {}
        self.copy_stream = torch.cuda.Stream()
        self.copy_event_prev : torch.cuda.Event = None
        self.copy_event : torch.cuda.Event = None
    
    def get_reducescatter_objects(self, recv_bytes_aligned: int, group_size: int, 
                                world_size: int, device_count: int, reduce_scatter_stream):
        """
        Get or create ibReduceScatter objects for the given byte size and group configuration.
        
        Args:
            recv_bytes_aligned: Aligned size of the tensor in bytes
            group_size: Size of the process group
            world_size: Total world size
            device_count: Number of CUDA devices per node
            reduce_scatter_stream: CUDA stream for reduce-scatter operations
        """
        if recv_bytes_aligned not in self.rs_ib_dict and self.use_custom_rs:
            torch.cuda.synchronize()
            # Determine group type and configuration
            is_full_world = group_size == world_size
            is_ep_group = group_size == world_size // device_count

            RSs = []
            if is_full_world:
                for _ in range(self.num_buffers):
                    RSs.append(ibReduceScatter(
                        recv_bytes_aligned, torch.distributed.group.WORLD, reduce_scatter_stream,
                        mode=0, barrier_all=True, vertical_group_rs=False
                    ))
             
            elif is_ep_group:
                for _ in range(self.num_buffers):
                    RSs.append(ibReduceScatter(
                        recv_bytes_aligned, torch.distributed.group.WORLD, reduce_scatter_stream,
                        mode=1, barrier_all=True, vertical_group_rs=True
                    ))
              
            torch.cuda.synchronize()
            self.rs_ib_dict[recv_bytes_aligned] = RSs
            self.rdc_scale[recv_bytes_aligned] = None
    
    def execute_reducescatter(self, recv_bytes_aligned: int, group_size: int, world_size: int,
                            reduce_scatter_input, reduce_scatter_input_aligned,
                            reduce_scatter_group, reduce_scatter_stream, reduce_scatter_reduce_op,
                            recv_bytes: int, reduce_scatter_output_numel: int, device):
        """
        Execute reduce-scatter operation using cached ibReduceScatter objects.
        
        Args:
            recv_bytes_aligned: Aligned size in bytes
            group_size: Process group size
            world_size: Total world size
            reduce_scatter_input: Input tensor
            reduce_scatter_input_aligned: Aligned input tensor
            reduce_scatter_group: Process group
            reduce_scatter_stream: CUDA stream
            reduce_scatter_reduce_op: Reduction operation
            recv_bytes: Original receive bytes
            reduce_scatter_output_numel: Output tensor element count
            device: Target device
        """
        with torch.cuda.stream(reduce_scatter_stream):
            if self.use_custom_rs and (group_size == world_size or 
                                        group_size == world_size // torch.cuda.device_count()):
                # Use cached ibReduceScatter objects with buffer swapping
                # reduce_output_aligned = reduce_scatter_input.new_empty(
                #     (recv_bytes_aligned // reduce_scatter_input.element_size(),)
                # )
                reduce_output = reduce_scatter_input.new_empty(
                    (reduce_scatter_output_numel,)
                )
                # Execute reduce-scatter
                self.rs_ib_dict[recv_bytes_aligned][self.comm_buf_iter].reduce_scatter_tensor(
                    output=reduce_output,
                    input=reduce_scatter_input_aligned,
                    group=reduce_scatter_group,
                    op=reduce_scatter_reduce_op,
                )
                self.comm_buf_iter = (self.comm_buf_iter + 1) % self.num_buffers
                
                # Extract the actual output from aligned buffer
                # reduce_output.copy_(reduce_output_aligned[:reduce_scatter_output_numel])
                # reduce_output.copy_(reduce_output_aligned[:reduce_scatter_output_numel])
                   
        return reduce_output

    
    def clear_cache(self):
        """Clear all cached reduce-scatter objects."""
        self.rs_ib_dict.clear()
        self.comm_buf_iter = 0

use_custom_rs = int(os.getenv("USE_CUSTOM_RS_IN_FSDP", 0))
use_custom_ag = int(os.getenv("USE_CUSTOM_AG_IN_FSDP", 0))


# NOTE: runtime-init globals (lazy)
ag_symm: Optional["SymmBufferManager"] = None
rs_symm: Optional["SymmBufferManager"] = None
ag_manager: Optional["AllGatherIBManager"] = None
rs_manager: Optional["ReduceScatterIBManager"] = None

def _get_num_ag_buffers() -> int:
    return 2 if int(os.getenv("USE_CUSTOM_AG_IN_FSDP", 0)) else 0

def _get_num_rs_buffers() -> int:
    return 1 if int(os.getenv("USE_CUSTOM_RS_IN_FSDP", 0)) else 0

def ensure_runtime_managers(device: Optional[torch.device] = None) -> None:
    """
    Lazily initialize global comm managers at runtime (first use).
    Safe to call multiple times.
    """
    global use_custom_ag, use_custom_rs
    global ag_symm, rs_symm, ag_manager, rs_manager

    # Refresh flags from env at runtime (optional but usually desired)
    use_custom_ag = int(os.getenv("USE_CUSTOM_AG_IN_FSDP", 0))
    use_custom_rs = int(os.getenv("USE_CUSTOM_RS_IN_FSDP", 0))

    num_ag_buffers = _get_num_ag_buffers()
    num_rs_buffers = _get_num_rs_buffers()

    symm_default = int(os.getenv("SYMM_BUF_SIZE", 0))

    # Only create SymmBufferManager if it might be used; keep None otherwise
    if ag_symm is None and num_ag_buffers > 0 and (use_custom_ag or use_custom_rs):
        ag_symm = SymmBufferManager(symm_default, num_buffers=num_ag_buffers)
    if rs_symm is None and num_rs_buffers > 0 and (use_custom_rs):
        rs_symm = SymmBufferManager(symm_default, num_buffers=num_rs_buffers)

    if ag_manager is None:
        ag_manager = AllGatherIBManager(num_buffers=max(num_ag_buffers, 1), use_custom_ag=use_custom_ag)
    else:
        ag_manager.use_custom_ag = use_custom_ag
        ag_manager.num_buffers = max(num_ag_buffers, 1)

    if rs_manager is None:
        rs_manager = ReduceScatterIBManager(num_buffers=max(num_rs_buffers, 1), use_custom_rs=use_custom_rs)
    else:
        rs_manager.use_custom_rs = use_custom_rs
        rs_manager.num_buffers = max(num_rs_buffers, 1)


class AllGatherResult(NamedTuple):
    all_gather_output: torch.Tensor
    all_gather_event: Optional[torch.Event]
    all_gather_work: Optional[dist.distributed_c10d.Work]
    # For each parameter, the all-gather input dtype for each input
    param_all_gather_input_dtypes: List[List[torch.dtype]]
    # For each parameter, the all-gather input numel for each input
    param_all_gather_input_numels: List[List[int]]
    # 1D flattened version of `param_all_gather_input_numels` saved to avoid
    # CPU overhead from recomputing
    all_gather_input_split_sizes: List[int]


lib = torch.library.Library("fsdp", "FRAGMENT")  # noqa: TOR901

lib.define(
    """
    all_gather_copy_in(
        Tensor[] all_gather_inputs,
        SymInt[] inp_split_sizes,
        SymInt all_gather_input_numel,
        SymInt world_size,
        SymInt rank,
        ScalarType dtype,
        Device device
    ) -> (Tensor, Tensor)
    """
)


@torch.library.impl(lib, "all_gather_copy_in", "Meta")
def all_gather_copy_in_meta(
    all_gather_inputs: List[torch.Tensor],
    inp_split_sizes: List[int],
    all_gather_input_numel: int,
    world_size: int,
    rank: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    all_gather_output = torch.empty(
        (all_gather_input_numel * world_size,), dtype=dtype, device="meta"
    )
    all_gather_input = all_gather_output.narrow(
        0, all_gather_input_numel * rank, all_gather_input_numel
    )
    return all_gather_input, all_gather_output


@torch.library.impl(lib, "all_gather_copy_in", "CUDA")
@torch.library.impl(lib, "all_gather_copy_in", "CPU")
def all_gather_copy_in_cuda(
    all_gather_inputs: List[torch.Tensor],
    inp_split_sizes: List[int],
    all_gather_input_numel: int,
    world_size: int,
    rank: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    global ag_symm
    ensure_runtime_managers(device)

    if (use_custom_ag or use_custom_rs) and world_size == dist.get_world_size() and ag_symm is not None:
        recv_bytes = all_gather_input_numel * world_size * all_gather_inputs[0].element_size()
        send_bytes = recv_bytes // world_size
        recv_bytes_aligned = (send_bytes + 127) // 128 * 128 * world_size
        symm_buf = ag_symm.get_buffer(bytes=recv_bytes_aligned, device=device)
        all_gather_output = symm_buf.view(dtype)[ : all_gather_input_numel * world_size]
    else:
        all_gather_output = torch.empty(
            (all_gather_input_numel * world_size,), dtype=dtype, device=device
        )
       
    all_gather_input = all_gather_output.narrow(
        0, all_gather_input_numel * rank, all_gather_input_numel
    )
    foreach_copy_dsts = torch.split(all_gather_input, inp_split_sizes)
    with torch.no_grad():
        torch._foreach_copy_(foreach_copy_dsts, all_gather_inputs)
    return all_gather_input, all_gather_output


lib.define(
    "split_with_sizes_copy(Tensor all_gather_output, SymInt[] all_gather_input_split_sizes, int dim=0, *, Tensor(a!)[] out) -> ()"
)


@torch.library.impl(lib, "split_with_sizes_copy", "Meta")
@torch.library.impl(lib, "split_with_sizes_copy", "CUDA")
@torch.library.impl(lib, "split_with_sizes_copy", "CPU")
def split_with_sizes_copy(
    all_gather_output: torch.Tensor,
    all_gather_input_split_sizes: List[int],
    dim: int,
    out: List[torch.Tensor],
) -> None:
    torch.split_with_sizes_copy(
        all_gather_output, all_gather_input_split_sizes, dim=dim, out=out
    )


lib.define(
    "chunk_cat(Tensor[] tensors, int dim, int num_chunks, *, Tensor(a!) out) -> ()"
)


@torch.library.impl(lib, "chunk_cat", "Meta")
@torch.library.impl(lib, "chunk_cat", "CUDA")
@torch.library.impl(lib, "chunk_cat", "CPU")
def chunk_cat(
    tensors: List[torch.Tensor],
    dim: int,
    num_chunks: int,
    out: torch.Tensor,
) -> None:
    torch._chunk_cat(tensors, dim, num_chunks, out=out)


@torch.no_grad()
def foreach_all_gather(
    fsdp_params: List[FSDPParam],
    group: dist.ProcessGroup,
    async_op: bool,
    all_gather_copy_in_stream: torch.Stream,
    all_gather_stream: torch.Stream,
    device: torch.device,
) -> Optional[AllGatherResult]:
    ensure_runtime_managers(device)

    world_size, rank = group.size(), group.rank()
    device_handle = _get_device_handle(device.type)
    
    with device_handle.stream(all_gather_copy_in_stream):
        param_all_gather_inputs = _get_param_all_gather_inputs(fsdp_params)
        (
            param_all_gather_input_dtypes,
            param_all_gather_input_numels,
            dtype,
        ) = _get_all_gather_input_metadatas(param_all_gather_inputs)
        if dtype == torch.uint8:
            all_gather_inputs = [
                t.view(torch.uint8) for ts in param_all_gather_inputs for t in ts
            ]
        else:
            all_gather_inputs = [t for ts in param_all_gather_inputs for t in ts]
        inp_split_sizes = [t.numel() for t in all_gather_inputs]
        all_gather_input_numel = sum(inp_split_sizes)
        send_bytes = all_gather_input_numel * all_gather_inputs[0].element_size()
        
        all_gather_input, all_gather_output = torch.ops.fsdp.all_gather_copy_in(
            all_gather_inputs,
            inp_split_sizes,
            all_gather_input_numel,
            world_size,
            rank,
            dtype,
            device,
        )

        
        del param_all_gather_inputs

    # Use the global dictionary for caching ibgdaAllgather objects
    global ag_manager, use_custom_ag
    if use_custom_ag and ag_manager is not None:
        ag_manager.get_allgather_objects(
            send_bytes=send_bytes,
            group_size=group.size(),
            world_size=dist.get_world_size(),
            device_count=torch.cuda.device_count(),
            all_gather_stream=all_gather_stream
        )
    # all_gather_output_ = torch.empty_like(all_gather_output)
    # Ensure all_gather_stream waits for copy operations to complete
    all_gather_stream.wait_stream(all_gather_copy_in_stream)
    global rs_event
    if rs_event is not None:
        all_gather_stream.wait_event(rs_event)
    # Execute operations on the all_gather_stream
    with device_handle.stream(all_gather_stream):
        if (
            use_custom_ag
            and ag_manager is not None
            and (group.size() == dist.get_world_size() or group.size() == dist.get_world_size() // torch.cuda.device_count())
        ):
            ag_manager.execute_allgather(
                send_bytes=send_bytes,
                group_size=group.size(),
                world_size=dist.get_world_size(),
                all_gather_output=all_gather_output,
                all_gather_input=all_gather_input,
                group=group
            )
        else:
            dist.all_gather_into_tensor(
                output_tensor=all_gather_output,
                input_tensor=all_gather_input,
                group=group,
            )

        # Set work handle to None since we're using custom implementation
        all_gather_work = None
        
        # Record event for synchronization
        all_gather_event = all_gather_stream.record_event()
        global ag_event
        ag_event = all_gather_event
        
        return AllGatherResult(
            all_gather_output,
            all_gather_event,
            all_gather_work,
            param_all_gather_input_dtypes,
            param_all_gather_input_numels,
            inp_split_sizes,
        )


@torch.no_grad()
def _get_param_all_gather_inputs(
    fsdp_params: List[FSDPParam],
) -> List[List[torch.Tensor]]:
    if compiled_autograd_enabled():
        return [fsdp_param.all_gather_inputs for fsdp_param in fsdp_params]

    # Intentionally try to run a fast-path that bypasses abstractions for the
    # common FSDP case of bf16/fp32 mixed precision in order to use foreach
    # copy for lower CPU overhead and more efficient copying in eager
    def use_foreach_copy(fsdp_param: FSDPParam) -> bool:
        return (
            fsdp_param.param_dtype is not None
            and not fsdp_param.offload_to_cpu
            and not hasattr(fsdp_param._sharded_local_tensor, "fsdp_pre_all_gather")
        )

    param_all_gather_inputs: List[List[torch.Tensor]] = [[] for _ in fsdp_params]
    foreach_copy_indices: List[int] = []
    foreach_copy_inputs: List[torch.Tensor] = []
    foreach_copy_input_numels: List[int] = []

    # 1st pass: for foreach-copy parameters, get inputs and metadata for the
    # foreach copy, and for the others, actually get their all-gather inputs
    for i, fsdp_param in enumerate(fsdp_params):
        if use_foreach_copy(fsdp_param):
            foreach_copy_indices.append(i)
            all_gather_input = (
                fsdp_param._sharded_param_data
                if fsdp_param.sharded_state == ShardedState.SHARDED
                else cast(torch.Tensor, fsdp_param._sharded_post_forward_param_data)
            )
            foreach_copy_inputs.append(all_gather_input)
            foreach_copy_input_numels.append(all_gather_input.numel())
        else:
            param_all_gather_inputs[i] = fsdp_param.all_gather_inputs

    # 2nd pass: use foreach copy to compute the remaining all-gather inputs
    if foreach_copy_inputs:
        fsdp_param_0 = fsdp_params[foreach_copy_indices[0]]
        param_dtype, device = fsdp_param_0.param_dtype, fsdp_param_0.device
        flat_foreach_copy_input = torch.empty(
            (sum(foreach_copy_input_numels),), device=device, dtype=param_dtype
        )
        splits = torch.split(flat_foreach_copy_input, foreach_copy_input_numels)
        torch._foreach_copy_(splits, foreach_copy_inputs)
        for i, split in zip(foreach_copy_indices, splits):
            param_all_gather_inputs[i] = [split]

    return param_all_gather_inputs


@torch.no_grad()
def foreach_all_gather_copy_out(
    all_gather_result: AllGatherResult,
    fsdp_params: List[FSDPParam],
    group: dist.ProcessGroup,
) -> None:
    (
        all_gather_output,
        all_gather_event,
        all_gather_work,
        param_all_gather_input_dtypes,
        param_all_gather_input_numels,
        all_gather_input_split_sizes,
    ) = all_gather_result

    _dtype, device = all_gather_output.dtype, all_gather_output.device
    device_handle = _get_device_handle(device.type)
    if all_gather_event is not None:  # sync op
        device_handle.current_stream().wait_event(all_gather_event)
    if isinstance(all_gather_work, dist.distributed_c10d.Work):  # async op
        all_gather_work.wait()
    world_size, device = group.size(), all_gather_output.device
   
    split_with_sizes_out: List[torch.Tensor] = []
    shard_i_copy_infos: List[Tuple[FSDPParam, List[torch.Tensor]]] = []
    for all_gather_input_numels, all_gather_input_dtypes, fsdp_param in zip(
        param_all_gather_input_numels, param_all_gather_input_dtypes, fsdp_params
    ):
        # NOTE: Under compile, make sure we always recreate all_gather_outputs
        # per AllGather. See [Note: Invariants for torch.compile Traceable FSDP2].
        force_recreate = compiled_autograd_enabled()
        fsdp_param.init_all_gather_outputs(
            all_gather_input_numels,
            all_gather_input_dtypes,
            world_size,
            device,
            force_recreate=force_recreate,
        )
        if not force_recreate:
            fsdp_param.alloc_all_gather_outputs()
        param_all_gather_outputs = fsdp_param.all_gather_outputs
        if fsdp_param.fsdp_placement.dim != 0:
            # Copy to a temporary and then chunk-cat into the final all-gather
            # output tensors
            param_all_gather_outputs = [
                torch.empty_like(t) for t in param_all_gather_outputs
            ]
            shard_i_copy_infos.append((fsdp_param, param_all_gather_outputs))
        split_with_sizes_out.extend(param_all_gather_outputs)

    all_gather_output = all_gather_output.view(world_size, -1)
    if all_gather_output.dtype == torch.uint8:
        out = [t.view(world_size, -1).view(torch.uint8) for t in split_with_sizes_out]
    else:
        out = [t.view(world_size, -1) for t in split_with_sizes_out]
    torch.ops.fsdp.split_with_sizes_copy(
        all_gather_output, all_gather_input_split_sizes, dim=1, out=out
    )

    for fsdp_param, param_all_gather_outputs in shard_i_copy_infos:
        # Chunk-cat from the temporary to the final all-gather output tensors
        shard_dim = fsdp_param.fsdp_placement.dim
        for param_all_gather_output, target_all_gather_output in zip(
            param_all_gather_outputs, fsdp_param.all_gather_outputs
        ):
            padded_sharded_size = (
                fsdp_param.padded_sharded_param_size
                if fsdp_param.sharded_state == ShardedState.SHARDED
                else cast(
                    torch.Tensor, fsdp_param._sharded_post_forward_param_data
                ).size()
            )
            pre_param_size = list(padded_sharded_size)
            pre_param_size[0] *= world_size
            chunks = torch.chunk(
                param_all_gather_output.view(pre_param_size), world_size, dim=0
            )
            post_param_size = list(padded_sharded_size)
            post_param_size[shard_dim] *= world_size
            cat_out = target_all_gather_output.view(post_param_size)
            torch.cat(chunks, dim=shard_dim, out=cat_out)
            torch._C._autograd._unsafe_set_version_counter(
                target_all_gather_output, target_all_gather_output._version - 1
            )


def rescale_max(tensor, scale, group_size):
    tensor /= scale
    max_val = tensor.abs().max().to(torch.float)
    return tensor, max_val


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
    layer_num: int,
    reduce_scale: Dict
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
    ensure_runtime_managers(device)

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
    padded_unsharded_sizes = tuple(
        _get_dim0_padded_size(grad.size(), world_size) for grad in unsharded_grads
    )
    reduce_scatter_input_numel = sum(s.numel() for s in padded_unsharded_sizes)
    reduce_scatter_output_numel = reduce_scatter_input_numel // world_size

    # global symm_buf
    send_bytes = reduce_scatter_input_numel * unsharded_grads[0].element_size()
    recv_bytes = send_bytes // world_size
    recv_bytes_aligned = (recv_bytes + 127) // 128 * 128
    send_bytes_aligned = recv_bytes_aligned * world_size

    global rs_manager, use_custom_rs, rs_symm
    # Get reduce-scatter objects
    if rs_manager is not None:
        rs_manager.get_reducescatter_objects(
            recv_bytes_aligned=recv_bytes,
            group_size=reduce_scatter_group.size(),
            world_size=dist.get_world_size(),
            device_count=torch.cuda.device_count(),
            reduce_scatter_stream=reduce_scatter_stream
        )


    # Custom op only support when group is of world size
    # reduce_scatter_input = torch.empty(
    #         (reduce_scatter_input_numel,), dtype=reduce_dtype, device=device
    #     )

    is_world_group = reduce_scatter_group.size() == dist.get_world_size()
    is_vertical_group = reduce_scatter_group.size() == dist.get_world_size()// torch.cuda.device_count()
    if is_world_group and use_custom_rs and rs_symm is not None:
        symm_buf = rs_symm.get_buffer(bytes=send_bytes_aligned, device=device)
        reduce_scatter_input = symm_buf.view(reduce_dtype)[ : reduce_scatter_input_numel]
        reduce_scatter_input_aligned = reduce_scatter_input
    else:
        reduce_scatter_input = torch.empty(
            (reduce_scatter_input_numel,), dtype=reduce_dtype, device=device
        )
        reduce_scatter_input_aligned = reduce_scatter_input


    device_handle = _get_device_handle(device.type)

    # Perform barrier
    if int(os.getenv("BARRIER_FSDP_ON_COMP", 0)):
        with torch.cuda.stream(device_handle.current_stream()):
            dist.all_reduce(torch.ones(1, device=reduce_scatter_input.device, dtype=torch.float), group = reduce_scatter_group, op=dist.ReduceOp.MIN)

    # Initialize sliding window for gradient scaling if not exists
    if (layer_num, reduce_scatter_input_numel) not in reduce_scale:
        # s_window = 1  # Fixed window size for moving average
        # scales = torch.ones(s_window, device=reduce_scatter_input.device, dtype=torch.float)
        # reduce_scale[(layer_num, reduce_scatter_input_numel)] = {
        #     "scales": scales,  # Array storing scaling factors in the sliding window
        #     "s_i": 0,          # Current index in the sliding window
        #     "s_window": s_window
        # }

        reduce_scale[(layer_num, reduce_scatter_input_numel)] = torch.ones(1, device = reduce_scatter_input.device, dtype = reduce_dtype)
    
    # Retrieve sliding window data for this layer and tensor size
    # scale_data = reduce_scale[(layer_num, reduce_scatter_input_numel)]
    # scales = scale_data["scales"]
    # s_i = scale_data["s_i"]
    # s_window = scale_data["s_window"]
    scale = reduce_scale[(layer_num, reduce_scatter_input_numel)]


    # Perform scale
    if int(os.getenv("SCALE_RS_IN_FSDP", 0)):
        # Compute current moving average of scaling factors
        # current_scale = scales.min()
        foreach_reduce_scatter_copy_in_scale(unsharded_grads, reduce_scatter_input, scale, world_size)
    else:
        foreach_reduce_scatter_copy_in(unsharded_grads, reduce_scatter_input, world_size)


    current_stream = device_handle.current_stream()
    # Only after the copy-in finishes can we free the gradients
    unsharded_grads.clear()
    reduce_scatter_stream.wait_stream(current_stream)

    all_reduce_input = None
    all_reduce_event = None
    global ag_event, rs_event
    if ag_event is not None:
        reduce_scatter_stream.wait_event(ag_event)

    with device_handle.stream(reduce_scatter_stream):
        _div_if_needed(reduce_scatter_input, predivide_factor)
        if reduce_scatter_reduce_op is None:
            if predivide_factor is None:
                reduce_scatter_reduce_op = ReduceOp.AVG
            else:
                reduce_scatter_reduce_op = ReduceOp.SUM
            
        if is_world_group and use_custom_rs and rs_manager is not None:
            reduce_output = rs_manager.execute_reducescatter(
                recv_bytes_aligned=recv_bytes,
                group_size=reduce_scatter_group.size(),
                world_size=dist.get_world_size(),
                reduce_scatter_input=reduce_scatter_input,
                reduce_scatter_input_aligned=reduce_scatter_input_aligned,
                reduce_scatter_group=reduce_scatter_group,
                reduce_scatter_stream=reduce_scatter_stream,
                reduce_scatter_reduce_op=reduce_scatter_reduce_op,
                recv_bytes=recv_bytes,
                reduce_scatter_output_numel=reduce_scatter_output_numel,
                device=device
            )
        elif is_vertical_group and use_custom_rs and rs_manager is not None:
            reduce_output = rs_manager.execute_reducescatter(
                recv_bytes_aligned=recv_bytes,
                group_size=reduce_scatter_group.size(),
                world_size=dist.get_world_size(),
                reduce_scatter_input=reduce_scatter_input,
                reduce_scatter_input_aligned=reduce_scatter_input_aligned,
                reduce_scatter_group=reduce_scatter_group,
                reduce_scatter_stream=reduce_scatter_stream,
                reduce_scatter_reduce_op=reduce_scatter_reduce_op,
                recv_bytes=recv_bytes,
                reduce_scatter_output_numel=reduce_scatter_output_numel,
                device=device
            )
        else:
            reduce_output = reduce_scatter_input.new_empty((reduce_scatter_output_numel,))
            dist.reduce_scatter_tensor(
                output=reduce_output,
                input=reduce_scatter_input,
                group=reduce_scatter_group,
                op=reduce_scatter_reduce_op,
            )

        # global tmp 
        # tmp += 1

        if int(os.getenv("SCALE_RS_IN_FSDP", 0)):
            # Apply inverse scaling to the reduced output
            reduce_output /= scale
            # Compute new scaling factor based on maximum absolute value
            max_val = reduce_output.abs().max().to(torch.float)
            dist.all_reduce(max_val, group=reduce_scatter_group, op=ReduceOp.MAX)
            new_scale = (10**5) / max_val / reduce_scatter_group.size()
            # # Update sliding window with new scaling factor
            # s_i_new = (s_i + 1) % s_window  # Circular buffer index
            # # scales[s_i_new] = new_scale
            # scales = new_scale
            # # Store updated sliding window data
            # reduce_scale[(layer_num, reduce_scatter_input_numel)] = {
            #     "scales": scales,
            #     "s_i": s_i_new,
            #     "s_window": s_window
            # }
            reduce_scale[(layer_num, reduce_scatter_input_numel)] = new_scale
            # dist.all_reduce(scales, group=reduce_scatter_group, op=ReduceOp.MIN)
            # print(f"{layer_num = }, {current_scale = }, {new_scale = }, {scales = }, {s_i_new = }, {max_val = }, {torch.isnan(reduce_output).any() = }")
            

        reduce_scatter_event = reduce_scatter_stream.record_event()
        post_reduce_stream = reduce_scatter_stream
        rs_event = reduce_scatter_event
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
        rs_event = post_reduce_event
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


def foreach_reduce_scatter_copy_in(
    unsharded_grads: List[torch.Tensor],
    reduce_scatter_input: torch.Tensor,
    world_size: int,
) -> None:
    reduce_scatter_input = reduce_scatter_input.view(world_size, -1)
    torch.ops.fsdp.chunk_cat(
        unsharded_grads, dim=0, num_chunks=world_size, out=reduce_scatter_input
    )

def foreach_reduce_scatter_copy_in_scale(
    unsharded_grads: List[torch.Tensor],
    reduce_scatter_input: torch.Tensor,
    scale: torch.Tensor,
    world_size: int,
) -> None:
    reduce_scatter_input = reduce_scatter_input.view(world_size, -1)
    ib_wrapper._chunk_cat_scale_out(
        srcs = unsharded_grads,
        dim = 0,
        num_chunks = world_size,
        scale = scale,
        out = reduce_scatter_input
    )


def foreach_reduce_scatter_copy_in_aligned(
    unsharded_grads: List[torch.Tensor],
    reduce_scatter_input_aligned: torch.Tensor,
    world_size: int,
    recv_bytes: int,
    recv_bytes_aligned: int
) -> None:
    reduce_scatter_input_aligned = reduce_scatter_input_aligned.view(n, -1)
    dtype = unsharded_grads[0].dtype
    device = unsharded_grads[0].device
    if recv_bytes_aligned != recv_bytes:
        grad_pad = torch.empty((world_size, (recv_bytes_aligned - recv_bytes) // reduce_scatter_input_aligned.element_size()), dtype = dtype, device = device)
        unsharded_grads_pad = unsharded_grads + [grad_pad]
        torch.ops.fsdp.chunk_cat(
            unsharded_grads_pad, dim=0, num_chunks=world_size, out=reduce_scatter_input_aligned
        )

    else:
        torch.ops.fsdp.chunk_cat(
            unsharded_grads, dim=0, num_chunks=world_size, out=reduce_scatter_input_aligned
        )

def _get_all_gather_input_metadatas(
    param_all_gather_inputs: List[List[torch.Tensor]],
) -> Tuple[List[List[torch.dtype]], List[List[int]], torch.dtype]:
    param_all_gather_input_dtypes: List[List[torch.dtype]] = []
    param_all_gather_input_numels: List[List[int]] = []
    all_gather_dtype = param_all_gather_inputs[0][0].dtype
    for all_gather_inputs in param_all_gather_inputs:
        input_dtypes: List[torch.dtype] = []
        input_numels: List[int] = []
        for all_gather_input in all_gather_inputs:
            if all_gather_input.dtype != all_gather_dtype:
                all_gather_dtype = torch.uint8
            input_dtypes.append(all_gather_input.dtype)
            input_numels.append(all_gather_input.numel())
        param_all_gather_input_dtypes.append(input_dtypes)
        param_all_gather_input_numels.append(input_numels)
    return (
        param_all_gather_input_dtypes,
        param_all_gather_input_numels,
        all_gather_dtype,
    )


def _get_gradient_divide_factors(
    reduce_scatter_group: dist.ProcessGroup,
    all_reduce_group: Optional[dist.ProcessGroup],
    reduce_dtype: torch.dtype,
) -> Union[Tuple[None, None], Tuple[float, float]]:
    # For fp32/bf16, we do not need to worry about overflow/underflow, so we
    # use NCCL's built-in division to avoid separate div kernels
    if reduce_dtype in (torch.float32, torch.bfloat16):
        return None, None
    data_parallel_size = reduce_scatter_group.size()
    if all_reduce_group is not None:
        data_parallel_size *= all_reduce_group.size()
    # Since fp16 has smaller dynamic range than fp32/bf16, we want to avoid
    # overflow/underflow. For N data parallel workers, each worker computes
    # g_i, and they collectively reduce (g_1 + ... + g_N) / N. To avoid
    # overflow/underflow, we divide by ~sqrt(N) before/after the reduction.
    factor: int = 1
    while data_parallel_size % factor == 0 and data_parallel_size / factor > factor:
        factor *= 2
    factor = float(factor)
    return (factor, data_parallel_size / factor)


def _div_if_needed(tensor: torch.Tensor, div_factor: Optional[float]) -> None:
    if div_factor is not None and div_factor > 1:
        tensor.div_(div_factor)

    