from xtuner.engine._strategy.deepspeed import (get_sequence_parallel_group,
                                               get_sequence_parallel_world_size,
                                               get_sequence_parallel_rank
                                               )
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from typing import Any, Tuple

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
except ImportError:
    pass


def all_to_all_scatter_nhead(input):
    # bs, seq, nhead, dim ==> 
    # bs, seq * sp_world_size, nhead / sp_world_size, dim
    sp_world_size = get_sequence_parallel_world_size()
    sp_group = get_sequence_parallel_group()
    bs, seq, nhead, dim = input.shape
    input_t = input.reshape(bs, seq, sp_world_size, nhead // sp_world_size, dim)
    input_t = input_t.permute(2, 0, 1, 3, 4).contiguous()
    output = torch.empty_like(input_t)
    dist.all_to_all_single(output, input_t, group=sp_group)
    output = output.transpose(0, 1)
    return output.reshape(bs, seq * sp_world_size, nhead // sp_world_size, dim)


def all_to_all_scatter_seq(input):
    # bs, seq * sp_world_size, nhead / sp_world_size, dim ==> 
    # bs, seq, nhead, dim
    sp_world_size = get_sequence_parallel_world_size()
    sp_group = get_sequence_parallel_group()
    bs, seq, nhead, dim = input.shape
    input_t = input.reshape(bs, sp_world_size, seq // sp_world_size, nhead, dim)
    input_t = input_t.transpose(0, 1).contiguous()
    output = torch.empty_like(input_t)
    dist.all_to_all_single(output, input_t, group=sp_group)
    output = output.permute(1, 2, 0, 3, 4)
    return output.reshape(bs, seq // sp_world_size, nhead * sp_world_size, dim)


class _SeqAllToAll(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, input: Tensor, scatter_seq) -> Tensor:
        ctx.scatter_seq = scatter_seq
        ctx.input_shape = input.shape
        if scatter_seq:
            return all_to_all_scatter_seq(input)
        return all_to_all_scatter_nhead(input)

    @staticmethod
    def backward(ctx: Any,
                 *grad_output: Tensor) -> Tuple[Tensor, None]:
        grad = _SeqAllToAll.apply(*grad_output, not ctx.scatter_seq)
        return (grad, None)


def pre_process_for_sequence_parallel_attn(query_states, key_states, value_states):
    sequence_parallel_world_size = get_sequence_parallel_world_size()
    n_head = query_states.shape[2]
    assert n_head % sequence_parallel_world_size == 0, \
        ('The number of attention heads should be divisible by '
        f'sequence_parallel_world_size. But got n_head = {n_head} and '
        f'sequence_parallel_world_size = {sequence_parallel_world_size}.')

    # (b, s // sp_world_size, nd, dim) -> (b, s, nd // sp_world_size, dim)
    query_states = _SeqAllToAll.apply(query_states, False)
    key_states = _SeqAllToAll.apply(key_states, False)
    value_states = _SeqAllToAll.apply(value_states, False)
    
    return query_states, key_states, value_states


def post_process_for_sequence_parallel_attn(attn_output):
    # (b, s, nd // sp_world_size, dim) -> (b, s // sp_world_size, nd, dim)
    output = _SeqAllToAll.apply(attn_output, True)
    return output


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def upad_qkv(query_layer, key_layer, value_layer, attention_mask, query_length):
    indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
    batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

    key_layer = index_first_axis(
        key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
    )
    value_layer = index_first_axis(
        value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
    )
    if query_length == kv_seq_len:
        # Different from the origin version as sequence parallel change 
        # the number of attention heads.
        query_layer = index_first_axis(
            query_layer.reshape(batch_size * kv_seq_len, -1, head_dim), indices_k
        )
        cu_seqlens_q = cu_seqlens_k
        max_seqlen_in_batch_q = max_seqlen_in_batch_k
        indices_q = indices_k
    elif query_length == 1:
        max_seqlen_in_batch_q = 1
        cu_seqlens_q = torch.arange(
            batch_size + 1, dtype=torch.int32, device=query_layer.device
        )  # There is a memcpy here, that is very bad.
        indices_q = cu_seqlens_q[:-1]
        query_layer = query_layer.squeeze(1)
    else:
        # The -q_len: slice assumes left padding.
        attention_mask = attention_mask[:, -query_length:]
        query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

    return (
        query_layer,
        key_layer,
        value_layer,
        indices_q,
        (cu_seqlens_q, cu_seqlens_k),
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
    )
