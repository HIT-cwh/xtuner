from typing import Protocol

import torch


class RMSNormProtocol(Protocol):
    def __call__(self, x: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor: ...


def native_rms_norm(x: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    # from torch.nn import functional as F

    # return F.rms_norm(x, weight.shape, weight, epsilon)
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + epsilon)
    return weight * x.to(input_dtype)


def npu_rms_norm(x: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    import torch_npu

    return torch_npu.npu_rms_norm(x, weight, epsilon=epsilon)[0]


def get_rms_norm() -> RMSNormProtocol:
    from xtuner.v1.utils import get_device

    device = get_device()
    if device == "cpu":
        return native_rms_norm
    elif device == "npu":
        return npu_rms_norm
    else:
        return native_rms_norm


rms_norm = get_rms_norm()
