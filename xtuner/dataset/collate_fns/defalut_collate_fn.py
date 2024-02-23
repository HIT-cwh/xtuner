# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence

from xtuner.engine._strategy.deepspeed import (get_sequence_parallel_rank,
                                               get_sequence_parallel_world_size
                                               )
from xtuner.utils import DEFAULT_PAD_TOKEN_INDEX, IGNORE_INDEX


def pad_for_sequence_parallel(input_ids,
                              labels,
                              seq_parallel_world_size,
                              pad_index=DEFAULT_PAD_TOKEN_INDEX):
    seq_length = len(input_ids)
    if seq_length % seq_parallel_world_size == 0:
        return input_ids, labels

    pad_num = seq_parallel_world_size - (seq_length % seq_parallel_world_size)
    input_ids += [pad_index] * pad_num
    labels += [IGNORE_INDEX] * pad_num

    return input_ids, labels


def default_collate_fn(instances: Sequence[Dict],
                       pad_index: int = DEFAULT_PAD_TOKEN_INDEX,
                       return_hf_format: bool = False,
                       use_varlen_attn: bool = False):

    seq_parallel_world_size = get_sequence_parallel_world_size()
    seq_parallel_world_rank = get_sequence_parallel_rank()

    input_ids, labels = [], []
    has_image = any(inst.get('pixel_values') is not None for inst in instances)
    if use_varlen_attn:
        cumulative_len, indexes = [], []
        assert len(instances) == 1, (
            f'If utilizing varlen attention, the batch size should be'
            f' set to 1, but got {len(instances)}')
        assert not has_image, 'Currently, it is not configured to '
        'accommodate the use of varlen Attention in multimodal training'

    if has_image:
        pixel_values = []

    for example in instances:
        cur_input_ids = example['input_ids']
        cur_labels = example['labels']
        cur_input_ids, cur_labels = pad_for_sequence_parallel(
            cur_input_ids, cur_labels, seq_parallel_world_size, pad_index)

        seq_length = len(cur_input_ids)
        assert seq_length % seq_parallel_world_size == 0
        sub_seq_length = seq_length // seq_parallel_world_size
        sub_seq_start = seq_parallel_world_rank * sub_seq_length
        sub_seq_end = (seq_parallel_world_rank + 1) * sub_seq_length

        input_ids.append(
            torch.LongTensor(cur_input_ids[sub_seq_start:sub_seq_end]))
        labels.append(torch.LongTensor(cur_labels[sub_seq_start:sub_seq_end]))
        if use_varlen_attn:
            cumulative_len.append(torch.IntTensor(example['cumulative_len']))
            indexes.append(
                torch.LongTensor(
                    example['indexes'][sub_seq_start:sub_seq_end]))

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
        indexes = torch.stack(indexes, dim=0)
        max_seqlen = (
            cumulative_len[0][1:] -  # noqa: W504
            cumulative_len[0][:-1]).max().item()
        data_dict = {
            'input_ids': input_ids,
            'cumulative_len': cumulative_len,
            'indexes': indexes,
            'labels': labels,
            'max_seqlen': max_seqlen
        }
    else:
        data_dict = {
            'input_ids': input_ids,
            'attention_mask': input_ids.ne(pad_index),
            'labels': labels
        }

    if has_image:
        pixel_values = torch.stack(pixel_values)
        data_dict['pixel_values'] = pixel_values

    if return_hf_format:
        return data_dict
    else:
        return {'data': data_dict, 'data_samples': None}
