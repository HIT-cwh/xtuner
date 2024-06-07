# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence

from xtuner.parallel.sequence import (get_sequence_parallel_world_size,
                                      pad_cumulative_len_for_sequence_parallel,
                                      pad_for_sequence_parallel)
from xtuner.utils import DEFAULT_PAD_TOKEN_INDEX, IGNORE_INDEX


def preference_collate_fn(instances: Sequence[Dict],
                          pad_index: int = DEFAULT_PAD_TOKEN_INDEX,
                          return_hf_format: bool = False,
                          use_varlen_attn: bool = False):
    seq_parallel_world_size = get_sequence_parallel_world_size()
    ds_names = []
    if not use_varlen_attn:
        # split chosen and rejected into two instances
        splited_instances = []
        for d in instances:
            splited_instances.append({
                'input_ids': d['chosen_ids'],
                'labels': d['chosen_labels']
            })
            splited_instances.append({
                'input_ids': d['rejected_ids'],
                'labels': d['rejected_labels']
            })
            ds_names.append(d.get('ds_name', None))
        instances = splited_instances

    input_ids, labels = [], []
    if use_varlen_attn:
        position_ids, cumulative_len = [], []
        assert len(instances) == 1, (
            f'If utilizing varlen attention, the batch size should be'
            f' set to 1, but got {len(instances)}')

    for example in instances:
        input_ids.append(torch.LongTensor(example['input_ids']))
        labels.append(torch.LongTensor(example['labels']))
        if use_varlen_attn:
            cumulative_len.append(torch.IntTensor(example['cumulative_len']))
            position_ids.append(torch.LongTensor(example['position_ids']))
            num_samples = (len(example['cumulative_len']) - 1) // 2
            ds_names.extend(example.get('ds_names', [None] * num_samples))

    ori_length = [len(ids) for ids in input_ids]
    if len(instances) > 1:
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=pad_index)
        labels = pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX)
    else:
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)

    if use_varlen_attn:
        attention_mask = None
        position_ids = torch.stack(position_ids, dim=0)
    else:
        # Some tokenizers have the same eos token and pad token, so input_ids
        # cannot be masked directly based on the pad token id.
        attention_mask = torch.zeros_like(input_ids).bool()
        for i in ori_length:
            attention_mask[:i] = True

        bs, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len).unsqueeze(0).long().repeat(bs, 1)

    if seq_parallel_world_size > 1:
        input_ids = pad_for_sequence_parallel(input_ids, pad_index)
        labels = pad_for_sequence_parallel(labels, IGNORE_INDEX)
        position_ids = pad_for_sequence_parallel(position_ids, 0)
        if attention_mask is not None:
            attention_mask = pad_for_sequence_parallel(attention_mask, 0)
        if use_varlen_attn:
            cumulative_len, attention_mask = pad_cumulative_len_for_sequence_parallel(
                cumulative_len)

    if use_varlen_attn:
        max_seqlen = (
            cumulative_len[0][1:] -  # noqa: W504
            cumulative_len[0][:-1]).max().item()
        data_dict = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'cumulative_len': cumulative_len,
            'position_ids': position_ids,
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

    if return_hf_format:
        return data_dict
    else:
        return {'data': data_dict, 'data_samples': {'ds_names': ds_names}}
