# Copyright (c) OpenMMLab. All rights reserved.
import inspect
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from mmengine import MessageHub
from transformers.cache_utils import Cache
from transformers.models.qwen2.modeling_qwen2 import (apply_rotary_pos_emb,
                                                      repeat_kv)
from transformers.modeling_outputs import CausalLMOutputWithPast
from xtuner._lite.accelerate import lmdeploy_is_available, liger_kernel_is_available
from .._attention import flash_attn_wo_mask, varlen_flash_attn

SUPPORT_FLASH2 = False

try:
    from flash_attn import flash_attn_func
    _flash_supports_window_size = 'window_size' in list(
        inspect.signature(flash_attn_func).parameters)
    SUPPORT_FLASH2 = True
except ImportError:
    pass


def _qwen2_attn_varlen_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
):
    is_training = self.training
   
    # assert is_training == (past_key_value is None)

    if 'padding_mask' in kwargs:
        warnings.warn(
            'Passing `padding_mask` is deprecated and will be removed in v4.37'
            ' Please make sure use `attention_mask` instead.`')

        # overwrite attention_mask with padding_mask
        attention_mask = kwargs.pop('padding_mask')
    bsz, q_len, _ = hidden_states.size()

    attn_context = MessageHub.get_instance('packed_sequence')
    position_ids = attn_context.get_info('position_ids')
    assert position_ids.size(1) == q_len, f'{position_ids.size(1)} {q_len}'
    sp_mesh = attn_context.get_info('sp_mesh')

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads,
                                     self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
                                 self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads,
                                     self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                'The cache structure has changed since version v4.36. '
                f'If you are using {self.__class__.__name__} '
                'for auto-regressive decoding with k/v caching, '
                'please make sure to initialize the attention class '
                'with a layer index.')
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len,
                                                       self.layer_idx)

    cos, sin = self.rotary_emb(value_states, position_ids)

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                    cos, sin, position_ids)

    if past_key_value is not None:
        # Activate slicing cache only if the config has a value
        # `sliding_windows` attribute
        cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0
        if (getattr(self.config, 'sliding_window', None) is not None
                and kv_seq_len > self.config.sliding_window
                and cache_has_contents):
            slicing_tokens = 1 - self.config.sliding_window

            past_key = past_key_value[self.layer_idx][0]
            past_value = past_key_value[self.layer_idx][1]

            past_key = past_key[:, :, slicing_tokens:, :].contiguous()
            past_value = past_value[:, :, slicing_tokens:, :].contiguous()

            if past_key.shape[-2] != self.config.sliding_window - 1:
                raise ValueError(
                    'past key must have a shape of (`batch_size, num_heads, '
                    'self.config.sliding_window-1, head_dim`), got'
                    f' {past_key.shape}')

            if attention_mask is not None:
                attention_mask = attention_mask[:, slicing_tokens:]
                attention_mask = torch.cat(
                    [attention_mask,
                     torch.ones_like(attention_mask[:, -1:])],
                    dim=-1)

        cache_kwargs = {'sin': sin, 'cos': cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs)

    # repeat k/v heads if n_kv_heads < n_heads for sequence parallel
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    dropout_rate = 0.0 if not self.training else self.attention_dropout

    # In PEFT, usually we cast the layer norms in float32 for
    # training stability reasons, therefore the input hidden states gets
    # silently casted in float32. Hence, we need
    # cast them back in float16 just to be sure everything works as expected.
    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, '_pre_quantization_dtype'):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    # Reashape to the expected shape for Flash Attention
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    # ----------------- flash attention forward ------------------------#

    if not self._flash_attn_uses_top_left_mask:
        causal = self.is_causal
    else:
        causal = self.is_causal and q_len != 1

    use_sliding_windows = (
        _flash_supports_window_size
        and getattr(self.config, 'sliding_window', None) is not None
        and kv_seq_len > self.config.sliding_window
        and self.layer_idx < self.config.max_window_layers
        and self.config.use_sliding_window)

    window_size = (self.config.sliding_window,
                   self.config.sliding_window) if use_sliding_windows else (-1,
                                                                            -1)

    assert SUPPORT_FLASH2
    cumulative_lengths = attn_context.get_info('cumulative_lengths')
    if cumulative_lengths is not None and SUPPORT_FLASH2 and bsz == 1:
        max_seqlen = attn_context.get_info('max_seqlen')
        attn_output = varlen_flash_attn(
            query_states,
            key_states,
            value_states,
            cumulative_lengths,
            max_seqlen,
            causal=causal,
            dropout_p=dropout_rate,
            window_size=window_size,
            training=self.training,
            sp_mesh=sp_mesh)
    else:
        attn_output = flash_attn_wo_mask(
            query_states,
            key_states,
            value_states,
            causal=causal,
            dropout_p=dropout_rate,
            window_size=window_size,
            training=self.training,
            sp_mesh=sp_mesh)

    # ---------------- flash attention forward end ------------------- #

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


from lmdeploy.pytorch.kernels import \
    apply_rotary_pos_emb as apply_rotary_pos_emb_lmdeploy
from lmdeploy.pytorch.kernels import fill_kv_cache, paged_attention_fwd
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from torch import Tensor


@torch.library.custom_op("attn::fill_kv_cache_op", mutates_args=('k_caches', 'v_caches'))
def fill_kv_cache_op(
    key_states: Tensor,
    value_states: Tensor,
    k_caches: Tensor,
    v_caches: Tensor,
    q_start_loc: Tensor,
    q_seq_length: Tensor,
    kv_seq_length: Tensor,
    max_q_seq_length: int,
    block_offsets: Tensor,
) -> None:
    # max_q_seq_length = torch.tensor(max_q_seq_length).cuda()
    fill_kv_cache(
        key_states,
        value_states,
        k_caches,
        v_caches,
        q_start_loc,
        q_seq_length,
        kv_seq_length=kv_seq_length,
        max_q_seq_length=max_q_seq_length,
        block_offsets=block_offsets,
    )
    return 


@fill_kv_cache_op.register_fake
def _(
    key_states: Tensor,
    value_states: Tensor,
    k_caches: Tensor,
    v_caches: Tensor,
    q_start_loc: Tensor,
    q_seq_length: Tensor,
    kv_seq_length: Tensor,
    max_q_seq_length: int,
    block_offsets: Tensor,
) -> None:
    return 


import flash_attn_2_cuda as flash_attn_cuda
@torch.library.custom_op("attn::fwd_kvcache_op", mutates_args=())
def fwd_kvcache_op(
    q: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    k: Optional[Tensor],
    v: Optional[Tensor],
    cache_seqlens: Tensor,
    rotary_cos: Optional[Tensor],
    rotary_sin: Optional[Tensor],
    cache_batch_idx: Optional[Tensor],
    cache_leftpad: Optional[Tensor],
    block_table: Optional[Tensor],
    alibi_slopes: Optional[Tensor],
    softmax_scale: float,
    causal: bool,
    window_size0: int,
    window_size1: int,
    softcap: float,
    rotary_interleaved: bool,
    num_splits: int,
) -> Tensor:
    out, softmax_lse = flash_attn_cuda.fwd_kvcache(
        q,
        k_cache,
        v_cache,
        k,
        v,
        cache_seqlens,
        rotary_cos,
        rotary_sin,
        cache_batch_idx,
        cache_leftpad,
        block_table,
        alibi_slopes,
        None,
        softmax_scale,
        causal,
        window_size0,
        window_size1,
        softcap,
        rotary_interleaved,
        num_splits,
    )
    return out


@fwd_kvcache_op.register_fake
def _(
    q: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    k: Optional[Tensor],
    v: Optional[Tensor],
    cache_seqlens: Tensor,
    rotary_cos: Optional[Tensor],
    rotary_sin: Optional[Tensor],
    cache_batch_idx: Optional[Tensor],
    cache_leftpad: Optional[Tensor],
    block_table: Optional[Tensor],
    alibi_slopes: Optional[Tensor],
    softmax_scale: float,
    causal: bool,
    window_size0: int,
    window_size1: int,
    softcap: float,
    rotary_interleaved: bool,
    num_splits: int,
) -> Tensor:
    return torch.empty_like(q)


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def flash_attn_with_kvcache_op(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens: Optional[Union[(int, torch.Tensor)]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0, # 0.0 means deactivated
    rotary_interleaved=True,
    alibi_slopes=None,
    num_splits=0,
    return_softmax_lse=False,
):
    assert not return_softmax_lse
    assert k_cache.stride(-1) == 1, "k_cache must have contiguous last dimension"
    assert v_cache.stride(-1) == 1, "v_cache must have contiguous last dimension"
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    if cache_seqlens is not None and isinstance(cache_seqlens, int):
        cache_seqlens = torch.full(
            (k_cache.shape[0],), cache_seqlens, dtype=torch.int32, device=k_cache.device
        )
        cache_seqlens = maybe_contiguous(cache_seqlens)
    cache_batch_idx = maybe_contiguous(cache_batch_idx)
    block_table = maybe_contiguous(block_table)
    out = fwd_kvcache_op(
        q,
        k_cache,
        v_cache,
        k,
        v,
        cache_seqlens,
        rotary_cos,
        rotary_sin,
        cache_batch_idx,
        cache_leftpad,
        block_table,
        alibi_slopes,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        softcap,
        rotary_interleaved,
        num_splits,
    )
    return out


@torch.library.custom_op("attn::apply_rotary_pos_emb_op", mutates_args=())
def apply_rotary_pos_emb_op(
    query_states: Tensor,
    key_states: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> Tuple[Tensor, Tensor]:
    return apply_rotary_pos_emb_lmdeploy(
        query_states,
        key_states,
        cos,
        sin)


@apply_rotary_pos_emb_op.register_fake
def _(
    query_states: Tensor,
    key_states: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> Tuple[Tensor, Tensor]:
    return torch.empty_like(query_states), torch.empty_like(key_states)


@torch.library.custom_op("attn::paged_attention_fwd_op", mutates_args=('attn_output', ))
def paged_attention_fwd_op(
    query_states: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    attn_output: Tensor,
    block_offsets: Tensor,
    q_start_loc: Tensor,
    q_seqlens: Tensor,
    kv_seqlens: Tensor,
    max_seqlen: int
) -> None:
    paged_attention_fwd(
        query_states,
        k_cache,
        v_cache,
        attn_output,
        block_offsets,
        q_start_loc=q_start_loc,
        q_seqlens=q_seqlens,
        kv_seqlens=kv_seqlens,
        max_seqlen=max_seqlen,
    )


@paged_attention_fwd_op.register_fake
def _(
    query_states: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    attn_output: Tensor,
    block_offsets: Tensor,
    q_start_loc: Tensor,
    q_seqlens: Tensor,
    kv_seqlens: Tensor,
    max_seqlen: int
) -> None:
    return None


from lmdeploy.pytorch.kernels import fill_kv_cache, paged_attention_fwd

def qwen2_attn_flash_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    **kwargs,
):
    kv_seq_length = kwargs['kv_seq_length']
    q_seq_length = kwargs['q_seq_length']
    q_start_loc = kwargs['q_start_loc']
    block_offsets = kwargs['block_offsets']
    max_q_seq_length = kwargs['max_q_seq_length']
    max_kv_seq_length = kwargs['max_kv_seq_length']
    cumulative_length = kwargs['cumulative_length']
    is_prefilling = kwargs['is_prefilling']
    
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    
    query_states = query_states.view(bsz * q_len, self.num_heads, self.head_dim)
    key_states = key_states.view(bsz * q_len, self.num_key_value_heads, self.head_dim)
    value_states = value_states.view(bsz * q_len, self.num_key_value_heads, self.head_dim)

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    
    query_states, key_states = apply_rotary_pos_emb_op(
        query_states,
        key_states,
        cos,
        sin,)
    
    fill_kv_cache_op(
        key_states,
        value_states,
        past_key_value[self.layer_idx][0],
        past_key_value[self.layer_idx][1],
        q_start_loc,
        q_seq_length,
        kv_seq_length=kv_seq_length,
        max_q_seq_length=max_q_seq_length,
        block_offsets=block_offsets,
    )

    attn_output = query_states
    paged_attention_fwd_op(
        query_states,
        past_key_value[self.layer_idx][0],
        past_key_value[self.layer_idx][1],
        attn_output,
        block_offsets,
        q_start_loc=q_start_loc,
        q_seqlens=q_seq_length,
        kv_seqlens=kv_seq_length,
        max_seqlen=max_q_seq_length,
    )
    attn_output = attn_output.reshape(*hidden_states.shape[:-1], -1)
    attn_output = self.o_proj(attn_output)
    return attn_output, None, past_key_value


def qwen2_attn_flash_forward1(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    **kwargs,
):
    
    kv_seq_length = kwargs['kv_seq_length']
    q_seq_length = kwargs['q_seq_length']
    q_start_loc = kwargs['q_start_loc']
    block_offsets = kwargs['block_offsets']
    max_q_seq_length = kwargs['max_q_seq_length']
    max_kv_seq_length = kwargs['max_kv_seq_length']
    cumulative_length = kwargs['cumulative_length']
    is_prefilling = kwargs['is_prefilling']
    
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads,
                                     self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
                                 self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads,
                                     self.head_dim).transpose(1, 2)
    
    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                    cos, sin, position_ids)
    
    # fill_kv_cache(
    fill_kv_cache_op(
        key_states.transpose(1, 2),  # torch.Size([1, 3, 2, 64])
        value_states.transpose(1, 2),
        past_key_value[self.layer_idx][0],  # torch.Size([8, 256, 2, 64])
        past_key_value[self.layer_idx][1],
        q_start_loc,  # tensor([0]
        q_seq_length,  # tensor([3]
        kv_seq_length=kv_seq_length,  # tensor([3
        max_q_seq_length=max_q_seq_length,  # tensor(3
        block_offsets=block_offsets,  # tensor([[0, 1, 2, 3, 4, 5, 6, 7]]
    )

    # ----------------- flash attention forward ------------------------#

    if not self._flash_attn_uses_top_left_mask:
        causal = self.is_causal
    else:
        causal = self.is_causal and q_len != 1

    use_sliding_windows = False

    window_size = (self.config.sliding_window,
                   self.config.sliding_window) if use_sliding_windows else (-1,
                                                                            -1)
    # TODO support sliding window attention
    if is_prefilling:
    
        key_states = repeat_kv(
            key_states, self.num_key_value_groups)
        value_states = repeat_kv(
            value_states, self.num_key_value_groups)
        
        attn_output = flash_attn_varlen_func(
            query_states.transpose(1,2).squeeze(0),
            key_states.transpose(1,2).squeeze(0),
            value_states.transpose(1,2).squeeze(0),
            cumulative_length,
            cumulative_length,
            max_q_seq_length,
            max_kv_seq_length,
            causal=True)
    else:
        query_states = query_states.transpose(1,2).transpose(0,1)

        attn_output = flash_attn_with_kvcache_op(
            query_states,
            past_key_value[self.layer_idx][0],
            past_key_value[self.layer_idx][1],
            cache_seqlens=kv_seq_length,
            block_table=block_offsets,
            causal=True)
        attn_output = attn_output.squeeze(1)

    # ---------------- flash attention forward end ------------------- #

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)
    # attn_output = o_proj_op(attn_output, self.o_proj.weight)

    attn_weights = None

    return attn_output, attn_weights, past_key_value



# def qwen2_attn_flash_forward(
#     self,
#     hidden_states: torch.Tensor,
#     attention_mask: Optional[torch.Tensor] = None,
#     position_ids: Optional[torch.LongTensor] = None,
#     past_key_value: Optional[Cache] = None,
#     output_attentions: bool = False,
#     use_cache: bool = False,
#     **kwargs,
# ):

#     lmdeploy_ctx = MessageHub.get_instance('paged_attention')
    
    # if lmdeploy_is_available() and len(lmdeploy_ctx.runtime_info) > 0:

    #     return _qwen2_attn_contiguous_batching_forward(self, hidden_states,attention_mask, position_ids,
    #                             past_key_value, use_cache)
    # else:
    #     return _qwen2_attn_varlen_forward(self, hidden_states,
    #                                           attention_mask, position_ids,
    #                                           past_key_value,
    #                                           output_attentions, use_cache)




def qwen2_casual_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    num_logits_to_keep: int = 0,
    label_shifted = False,
    **loss_kwargs,
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        num_logits_to_keep (`int`, *optional*):
            Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
            `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
            token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, Qwen2ForCausalLM

    >>> model = Qwen2ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
    >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    ```"""

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    hidden_states = outputs[0]

    if labels is None:
        loss = None
        logits = self.lm_head(hidden_states)
    else:
        if liger_kernel_is_available():
            # unable to return logits when using Liger Kernel
            logits = None

            if label_shifted:
                shift_hidden_states = hidden_states
                shift_labels = labels
            else:
                shift_hidden_states = hidden_states[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

            shift_hidden_states = shift_hidden_states.view(-1, self.config.hidden_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_hidden_states.device)

            from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss

            loss_fct = LigerFusedLinearCrossEntropyLoss()
            loss = loss_fct(self.lm_head.weight, shift_hidden_states, shift_labels, self.lm_head.bias)

        else:
            logits = self.lm_head(hidden_states)

            if label_shifted:
                shift_logits = logits
                shift_labels = labels
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)

            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
