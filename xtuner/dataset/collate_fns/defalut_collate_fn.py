# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence

from xtuner.engine._strategy.deepspeed import (get_sequence_parallel_rank,
                                               get_sequence_parallel_world_size
                                               )
from xtuner.utils import DEFAULT_PAD_TOKEN_INDEX, IGNORE_INDEX


def pad_for_sequence_parallel(tensor,
                              seq_parallel_world_size,
                              pad_index=DEFAULT_PAD_TOKEN_INDEX):
    bs, seq_len = tensor.shape
    if seq_len % seq_parallel_world_size == 0:
        return tensor
    
    pad_num = seq_parallel_world_size - (seq_len % seq_parallel_world_size)
    pad = torch.full((bs, pad_num), pad_index, dtype=tensor.dtype, device=tensor.device)
    tensor = torch.cat([tensor, pad], dim=1)
    return tensor


def split_for_sequence_parallel(input_ids, labels, position_ids):
    seq_parallel_world_size = get_sequence_parallel_world_size()
    seq_parallel_world_rank = get_sequence_parallel_rank()
    seq_len = input_ids.size(1)
    assert seq_len % seq_parallel_world_size == 0
    sub_seq_len = seq_len // seq_parallel_world_size
    sub_seq_start = seq_parallel_world_rank * sub_seq_len
    sub_seq_end = (seq_parallel_world_rank + 1) * sub_seq_len
    input_ids = input_ids[:, sub_seq_start:sub_seq_end]
    labels = labels[:, sub_seq_start:sub_seq_end]
    position_ids = position_ids[:, sub_seq_start:sub_seq_end]
    return input_ids, labels, position_ids


def default_collate_fn(instances: Sequence[Dict],
                       pad_index: int = DEFAULT_PAD_TOKEN_INDEX,
                       return_hf_format: bool = False,
                       use_varlen_attn: bool = False):

    seq_parallel_world_size = get_sequence_parallel_world_size()

    input_ids, labels, position_ids = [], [], []
    has_image = any(inst.get('pixel_values') is not None for inst in instances)
    if use_varlen_attn:
        cumulative_len= []
        assert len(instances) == 1, (
            f'If utilizing varlen attention, the batch size should be'
            f' set to 1, but got {len(instances)}')
        assert not has_image, 'Currently, it is not configured to '
        'accommodate the use of varlen Attention in multimodal training'

    if has_image:
        pixel_values = []
    
    for example in instances:
        input_ids.append(torch.LongTensor(example['input_ids']))
        labels.append(torch.LongTensor(example['labels']))
        if use_varlen_attn:
            cumulative_len.append(torch.IntTensor(example['cumulative_len']))
            position_ids.append(torch.LongTensor(example['indexes']))

        if has_image:
            pixel_values.append(example['pixel_values'])

    if len(instances) > 1:
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=pad_index)
        labels = pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX)
    else:
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
    
    if use_varlen_attn:
        assert input_ids.size(1) % seq_parallel_world_size == 0
        position_ids = torch.stack(position_ids, dim=0)
    else:
        input_ids = pad_for_sequence_parallel(input_ids, seq_parallel_world_size, pad_index)
        labels = pad_for_sequence_parallel(labels, seq_parallel_world_size, IGNORE_INDEX)
        attention_mask = input_ids.ne(pad_index)
        position_ids = attention_mask.long().cumsum(-1) - 1

    input_ids, labels, position_ids = split_for_sequence_parallel(input_ids, labels, position_ids)

    if use_varlen_attn:
        max_seqlen = (
            cumulative_len[0][1:] -  # noqa: W504
            cumulative_len[0][:-1]).max().item()
        data_dict = {
            'input_ids': input_ids,
            'cumulative_len': cumulative_len,
            'indexes': position_ids,
            'labels': labels,
            'max_seqlen': max_seqlen
        }
    else:
        data_dict = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'labels': labels
        }

    if has_image:
        pixel_values = torch.stack(pixel_values)
        data_dict['pixel_values'] = pixel_values

    if return_hf_format:
        return data_dict
    else:
        return {'data': data_dict, 'data_samples': None}
