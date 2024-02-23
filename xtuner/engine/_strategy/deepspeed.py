# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
from mmengine._strategy import DeepSpeedStrategy as MMEngineDeepSpeedStrategy

from xtuner import DS_CEPH_DIR
from xtuner.utils.fileio import patch_fileio

# For DeepSpeed's sequence parallel
_SEQUENCE_PARALLEL_GROUP = None
_SEQUENCE_PARALLEL_WORLD_SIZE = None
_SEQUENCE_PARALLEL_RANK = None

_DATA_PARALLEL_GROUP = None
_DATA_PARALLEL_WORLD_SIZE = None
_DATA_PARALLEL_RANK = None


def init_seq_parallel(sequence_parallel_size: int = 1):
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()

    # enable_ds_sequence_parallel = sequence_parallel_size > 1
    # if enable_ds_sequence_parallel:
    if world_size % sequence_parallel_size != 0:
        raise RuntimeError(
            f'world_size ({world_size}) is not divisible by sequence_parallel_size {sequence_parallel_size}'
        )

    num_sequence_parallel_groups: int = world_size // sequence_parallel_size

    rank = torch.distributed.get_rank()

    # Build the sequence parallel groups.
    global _SEQUENCE_PARALLEL_GROUP
    assert _SEQUENCE_PARALLEL_GROUP is None, \
        'sequence parallel group is already initialized'
    for i in range(num_sequence_parallel_groups):
        ranks = range(i * sequence_parallel_size,
                      (i + 1) * sequence_parallel_size)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _SEQUENCE_PARALLEL_GROUP = group

    global _DATA_PARALLEL_GROUP
    assert _DATA_PARALLEL_GROUP is None, 'data parallel group is already initialized'
    all_data_parallel_group_ranks = []
    start_rank = 0
    end_rank = world_size
    for j in range(sequence_parallel_size):
        ranks = range(start_rank + j, end_rank, sequence_parallel_size)
        all_data_parallel_group_ranks.append(list(ranks))
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _DATA_PARALLEL_GROUP = group


def get_sequence_parallel_group():
    """Get the sequence parallel group the caller rank belongs to."""
    return _SEQUENCE_PARALLEL_GROUP


def get_sequence_parallel_world_size():
    """Return world size for the sequence parallel group."""
    global _SEQUENCE_PARALLEL_WORLD_SIZE
    if _SEQUENCE_PARALLEL_WORLD_SIZE is not None:
        return _SEQUENCE_PARALLEL_WORLD_SIZE
    _SEQUENCE_PARALLEL_WORLD_SIZE = torch.distributed.get_world_size(
        group=get_sequence_parallel_group())
    return _SEQUENCE_PARALLEL_WORLD_SIZE


def get_sequence_parallel_rank():
    """Return my rank for the sequence parallel group."""
    global _SEQUENCE_PARALLEL_RANK
    if _SEQUENCE_PARALLEL_RANK is not None:
        return _SEQUENCE_PARALLEL_RANK
    _SEQUENCE_PARALLEL_RANK = torch.distributed.get_rank(
        group=get_sequence_parallel_group())
    return _SEQUENCE_PARALLEL_RANK


def get_data_parallel_group():
    """Get the data parallel group the caller rank belongs to."""
    assert _DATA_PARALLEL_GROUP is not None, 'data parallel group is not initialized'
    return _DATA_PARALLEL_GROUP


def get_data_parallel_world_size():
    """Return world size for the data parallel group."""
    global _DATA_PARALLEL_WORLD_SIZE
    if _DATA_PARALLEL_WORLD_SIZE is not None:
        return _DATA_PARALLEL_WORLD_SIZE
    _DATA_PARALLEL_WORLD_SIZE = torch.distributed.get_world_size(
        group=get_data_parallel_group())
    return _DATA_PARALLEL_WORLD_SIZE


def get_data_parallel_rank():
    """Return my rank for the data parallel group."""
    global _DATA_PARALLEL_RANK
    if _DATA_PARALLEL_RANK is not None:
        return _DATA_PARALLEL_RANK
    _DATA_PARALLEL_RANK = torch.distributed.get_rank(
        group=get_data_parallel_group())
    return _DATA_PARALLEL_RANK


class DeepSpeedStrategy(MMEngineDeepSpeedStrategy):

    def __init__(self, *args, **kwargs):
        sp_size = kwargs.pop('sp_size')
        self.sp_size = sp_size
        super().__init__(*args, **kwargs)

        from transformers.integrations.deepspeed import HfDeepSpeedConfig

        # hf_deepspeed_config has to be saved as an attribute.
        self.hf_deepspeed_config = HfDeepSpeedConfig(self.config)

    def _wrap_model(self, model):
        wrapper = super()._wrap_model(model)
        # hard code for deepspeed zero3
        # When utilizing Zero3, the model isn't allocated to CUDA within the
        # `deepspeed.initialize` process.
        assert hasattr(wrapper.model, 'data_preprocessor')
        wrapper.model.data_preprocessor.cuda()
        return wrapper

    def save_checkpoint(self, *args, **kwargs) -> None:
        if DS_CEPH_DIR:
            from os import path as osp
            work_dir_prefix = osp.split(self.work_dir)[0]

            filename = kwargs['filename'].replace(work_dir_prefix, DS_CEPH_DIR)
            kwargs['filename'] = filename
            with patch_fileio():
                super().save_checkpoint(*args, **kwargs)
        else:
            super().save_checkpoint(*args, **kwargs)

    def load_checkpoint(self, *args, **kwargs) -> None:
        if DS_CEPH_DIR:

            with patch_fileio():
                checkpoint = super().load_checkpoint(*args, **kwargs)
        else:
            checkpoint = super().load_checkpoint(*args, **kwargs)
        return checkpoint

    def resume(self, *args, **kwargs) -> None:
        if DS_CEPH_DIR:

            with patch_fileio():
                checkpoint = super().resume(*args, **kwargs)
        else:
            checkpoint = super().resume(*args, **kwargs)
        return checkpoint

    def _setup_distributed(  # type: ignore
        self,
        launcher: Optional[str] = None,
        backend: str = 'nccl',
        **kwargs,
    ):
        super()._setup_distributed(launcher, backend, **kwargs)
        init_seq_parallel(self.sp_size)
