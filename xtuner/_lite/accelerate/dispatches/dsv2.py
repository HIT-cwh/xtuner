# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from mmengine import MessageHub
from transformers.cache_utils import Cache
from transformers.utils.import_utils import is_flash_attn_2_available

from xtuner._lite.parallel import (get_sp_world_size,
                                   post_process_for_sequence_parallel_attn,
                                   pre_process_for_sequence_parallel_attn)
from xtuner.model.transformers_models.deepseek_v2.modeling_deepseek import \
    apply_rotary_pos_emb

SUPPORT_FLASH2 = is_flash_attn_2_available()

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except ImportError:
    pass


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_intern(q,
                                k,
                                cos,
                                sin,
                                position_ids=None,
                                unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def deepseek_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
):
    # DeepseekV2FlashAttention2 attention does not support output_attentions
    if 'padding_mask' in kwargs:
        warnings.warn(
            'Passing `padding_mask` is deprecated and will be removed in '
            'v4.37. Please make sure use `attention_mask` instead.`')

        # overwrite attention_mask with padding_mask
        attention_mask = kwargs.pop('padding_mask')

    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    if self.q_lora_rank is None:
        q = self.q_proj(hidden_states)
    else:
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
    q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
    q_nope, q_pe = torch.split(
        q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    compressed_kv, k_pe = torch.split(
        compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
    kv = (
        self.kv_b_proj(self.kv_a_layernorm(compressed_kv)).view(
            bsz, q_len, self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2))

    k_nope, value_states = torch.split(
        kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
    kv_seq_len = value_states.shape[-2]

    kv_seq_len = value_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len,
                                                       self.layer_idx)

    assert position_ids is not None, '`position_ids` should not be None.'
    if type(self.rotary_emb).__name__ in (
            'InternLM2RotaryEmbedding',
            'InternLM2DynamicNTKScalingRotaryEmbedding'):
        cos, sin = self.rotary_emb(value_states, position_ids)
        q_pe, k_pe = apply_rotary_pos_emb_intern(q_pe, k_pe, cos, sin)
    else:
        if self.training:
            cos, sin = self.rotary_emb(
                value_states, seq_len=position_ids.max() + 1)
        else:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

    query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
    query_states[:, :, :, :self.qk_nope_head_dim] = q_nope
    query_states[:, :, :, self.qk_nope_head_dim:] = q_pe

    key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
    key_states[:, :, :, :self.qk_nope_head_dim] = k_nope
    key_states[:, :, :, self.qk_nope_head_dim:] = k_pe

    if self.q_head_dim != self.v_head_dim:
        value_states = F.pad(value_states,
                             [0, self.q_head_dim - self.v_head_dim])

    if past_key_value is not None:
        cache_kwargs = {'sin': sin, 'cos': cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs)

    # Reashape to the expected shape for Flash Attention
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    dropout_rate = self.attention_dropout if self.training else 0.0

    # In PEFT, usually we cast the layer norms in float32 for training
    # stability reasons therefore the input hidden states gets silently
    # casted in float32. Hence, we need cast them back in the correct dtype
    # just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not
    # cast the LayerNorms in fp32. (DeepseekV2RMSNorm handles it correctly)

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        # Handle the case where the model is quantized
        if hasattr(self.config, '_pre_quantization_dtype'):
            target_dtype = self.config._pre_quantization_dtype
        elif torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        else:
            target_dtype = self.q_a_proj.weight.dtype

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    enable_sequence_parallel = (
        dist.is_initialized() and get_sp_world_size() > 1 and self.training)
    if enable_sequence_parallel:
        query_states, key_states, value_states = \
            pre_process_for_sequence_parallel_attn(
                query_states, key_states, value_states)
        # self.num_heads is used in self._upad_input method
        # num_heads has been changed because of sequence parallel
        ori_num_head = self.num_heads
        self.num_heads = query_states.shape[-2]

    attn_output = self._flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_states.shape[1],
        dropout=dropout_rate,
        softmax_scale=self.softmax_scale,
    )

    if enable_sequence_parallel:
        attn_output = post_process_for_sequence_parallel_attn(attn_output)
        self.num_heads = ori_num_head

    if self.q_head_dim != self.v_head_dim:
        attn_output = attn_output[:, :, :, :self.v_head_dim]

    attn_output = attn_output.reshape(bsz, q_len, self.num_heads *
                                      self.v_head_dim).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def deepseek_varlen_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
):

    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    attn_context = MessageHub.get_instance('packed_sequence')
    position_ids = attn_context.get_info('position_ids')
    assert position_ids.size(1) == q_len, f'{position_ids.size(1)} {q_len}'

    if self.q_lora_rank is None:
        q = self.q_proj(hidden_states)
    else:
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
    q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
    q_nope, q_pe = torch.split(
        q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    compressed_kv, k_pe = torch.split(
        compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
    kv = (
        self.kv_b_proj(self.kv_a_layernorm(compressed_kv)).view(
            bsz, q_len, self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2))

    k_nope, value_states = torch.split(
        kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
    kv_seq_len = value_states.shape[-2]

    kv_seq_len = value_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len,
                                                       self.layer_idx)

    assert position_ids is not None, '`position_ids` should not be None.'
    if type(self.rotary_emb).__name__ in (
            'InternLM2RotaryEmbedding',
            'InternLM2DynamicNTKScalingRotaryEmbedding'):
        cos, sin = self.rotary_emb(value_states, position_ids)
        q_pe, k_pe = apply_rotary_pos_emb_intern(q_pe, k_pe, cos, sin)
    else:
        if self.training:
            cos, sin = self.rotary_emb(
                value_states, seq_len=position_ids.max() + 1)
        else:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

    query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
    query_states[:, :, :, :self.qk_nope_head_dim] = q_nope
    query_states[:, :, :, self.qk_nope_head_dim:] = q_pe

    key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
    key_states[:, :, :, :self.qk_nope_head_dim] = k_nope
    key_states[:, :, :, self.qk_nope_head_dim:] = k_pe

    if self.q_head_dim != self.v_head_dim:
        value_states = F.pad(value_states,
                             [0, self.q_head_dim - self.v_head_dim])

    if past_key_value is not None:
        cache_kwargs = {'sin': sin, 'cos': cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs)

    # In PEFT, usually we cast the layer norms in float32 for training
    # stability reasons therefore the input hidden states gets silently
    # casted in float32. Hence, we need cast them back in the correct dtype
    # just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not
    # cast the LayerNorms in fp32. (DeepseekV2RMSNorm handles it correctly)

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        # Handle the case where the model is quantized
        if hasattr(self.config, '_pre_quantization_dtype'):
            target_dtype = self.config._pre_quantization_dtype
        elif torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        else:
            target_dtype = self.q_a_proj.weight.dtype

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    # Reashape to the expected shape for Flash Attention
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    # ----------------- varlen flash attention forward ----------------------#
    dropout_rate = self.attention_dropout if self.training else 0.0

    if not self._flash_attn_uses_top_left_mask:
        causal = self.is_causal
    else:
        causal = self.is_causal and q_len != 1

    # attn_context = MessageHub.get_instance('varlen_attention_context')

    enable_sequence_parallel = (
        dist.is_initialized() and get_sp_world_size() > 1 and self.training)
    if enable_sequence_parallel:
        query_states, key_states, value_states = \
            pre_process_for_sequence_parallel_attn(
                query_states, key_states, value_states)

    assert SUPPORT_FLASH2
    cumulative_lengths = attn_context.get_info('cumulative_lengths')
    if cumulative_lengths is not None and bsz == 1:
        max_seqlen = attn_context.get_info('max_seqlen')

        q_unpad, k_unpad, v_unpad = query_states.flatten(
            0, 1), key_states.flatten(0, 1), value_states.flatten(0, 1)
        attn_output = flash_attn_varlen_func(
            q_unpad,
            k_unpad,
            v_unpad,
            cumulative_lengths,
            cumulative_lengths,
            max_seqlen,
            max_seqlen,
            softmax_scale=self.softmax_scale,
            dropout_p=dropout_rate,
            return_attn_probs=False,
            causal=causal)
        attn_output = attn_output.unsqueeze(0)
    else:
        attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            dropout_p=dropout_rate,
            softmax_scale=self.softmax_scale,
            causal=causal)

    if enable_sequence_parallel:
        attn_output = post_process_for_sequence_parallel_attn(attn_output)

    # ---------------- varlen flash attention forward end ------------------ #

    if self.q_head_dim != self.v_head_dim:
        attn_output = attn_output[:, :, :, :self.v_head_dim]

    attn_output = attn_output.reshape(bsz, q_len,
                                      self.num_heads * self.v_head_dim)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value
