# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch.distributed._tensor import DTensor

try:
    from flash_attn.ops.triton.layernorm import rms_norm_fn
except ImportError:
    try:
        from flash_attn.ops.triton.layer_norm import rms_norm_fn
    except ImportError:
        import flash_attn
        raise ImportError(f'flash_attn version {flash_attn.__version__}')


def rms_norm_forward(self, hidden_states):

    from torch.distributed._functional_collectives import AsyncCollectiveTensor
    if isinstance(hidden_states, AsyncCollectiveTensor):
        hidden_states = hidden_states.wait()
    if (hidden_states.device == torch.device('cpu')
            or self.weight.device == torch.device('cpu')):
        raise RuntimeError(
            'Can not use triton kernels on cpu. Please set `USE_TRITON_KERNEL`'
            ' environment variable to 0 before training.')

    if isinstance(self.weight, DTensor):
        w = self.weight.to_local()
    else:
        w = self.weight
    ret = rms_norm_fn(hidden_states, w, None, eps=self.variance_epsilon)
    return ret
