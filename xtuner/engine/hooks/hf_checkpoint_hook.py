# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from pathlib import Path
from typing import Optional, Union

import torch.distributed as dist
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

DATA_BATCH = Optional[Union[dict, tuple, list]]


class HFCheckpointHook(Hook):

    priority = 95  # lower than CheckpointHook in MMEngine

    def __init__(self, out_dir: Optional[Union[str, Path]] = None) -> None:
        self.out_dir = out_dir

    @staticmethod
    def _get_deepspeed_z3_state_dict(model):
        import deepspeed

        def get_state_dict(module, state_dict, prefix=''):
            for name, param in module.named_parameters(
                    prefix=prefix[:-1], recurse=False):
                gather_param = None
                with deepspeed.zero.GatheredParameters(param, modifier_rank=0):
                    if dist.get_rank() == 0:
                        gather_param = param.to(device='cpu')
                    else:
                        gather_param = param.to(device='meta')
                state_dict[name] = gather_param
            for name, buffer in module.named_buffers(
                    prefix=prefix[:-1], recurse=False):
                if dist.get_rank() == 0:
                    buffer = buffer.to(device='cpu')
                else:
                    buffer = buffer.to(device='meta')
                state_dict[name] = buffer

            for name, child in module._modules.items():
                if child is not None:
                    get_state_dict(child, state_dict, prefix + name + '.')

        state_dict = {}
        get_state_dict(model, state_dict)
        return state_dict

    def after_run(self, runner) -> None:
        if self.out_dir is None:
            self.out_dir = osp.join(runner.work_dir, 'hf_model')
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        llm = model.llm
        state_dict = self._get_deepspeed_z3_state_dict(llm)
        if dist.get_rank() == 0:
            llm.save_pretrained(self.out_dir, state_dict=state_dict)
