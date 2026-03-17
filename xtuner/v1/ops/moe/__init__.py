import os
import traceback

import torch
from mmengine import digit_version

from .protocol import (
    GroupGemmProtocol,
    MoePermuteProtocol,
    MoeUnpermuteProtocol,
    cpu_group_gemm,
    cpu_permute,
    cpu_unpermute,
)


def get_group_gemm() -> GroupGemmProtocol:
    from xtuner.v1.utils import get_device

    device = get_device()
    if device == "cpu":
        return cpu_group_gemm
    elif device == "cuda":
        if os.environ.get("XTUNER_USE_CUTLASS_GROUP_GEMM", "0") == "1":
            from .cuda import cutlass_group_gemm as cuda_group_gemm

            print("---------------------------Using cutlass group gemm-------------------------")
        else:
            from .cuda import triton_group_gemm as cuda_group_gemm

        return cuda_group_gemm

    elif device == "npu":
        from .npu import npu_group_gemm

        return npu_group_gemm
    else:
        raise NotImplementedError


use_torch_permute = os.environ.get("USE_TORCH_PERMUTE", "0") == "1"


def get_token_permute() -> MoePermuteProtocol:
    from xtuner.v1.utils import get_device

    device = get_device()
    if device == "cpu":
        return cpu_permute

    elif device == "cuda":
        from .cuda import cuda_token_permute
        from .cuda.permute_unpermute import cuda_token_permute_torch

        return cuda_token_permute_torch if use_torch_permute else cuda_token_permute
    elif device == "npu":
        from .npu import npu_token_permute

        return npu_token_permute
    else:
        raise NotImplementedError


def get_token_unpermute() -> MoeUnpermuteProtocol:
    from xtuner.v1.utils import get_device

    device = get_device()
    if device == "cpu":
        return cpu_unpermute
    elif device == "cuda":
        from .cuda import cuda_token_unpermute
        from .cuda.permute_unpermute import cuda_token_unpermute_torch

        return cuda_token_unpermute_torch if use_torch_permute else cuda_token_unpermute
    elif device == "npu":
        from .npu import npu_token_unpermute

        return npu_token_unpermute
    else:
        raise NotImplementedError


group_gemm = get_group_gemm()
permute = get_token_permute()
unpermute = get_token_unpermute()
