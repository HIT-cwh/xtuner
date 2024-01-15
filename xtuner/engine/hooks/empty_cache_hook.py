# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks import Hook

from typing import Dict, Optional, Sequence, Union
from deepspeed import get_accelerator

DATA_BATCH = Optional[Union[dict, tuple, list]]

class EmptyCacheHook(Hook):

    priority = 'VERY_LOW'

    def __init__(self, every_n_iters=1):
        self.every_n_iters = every_n_iters

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:
        get_accelerator().empty_cache()
