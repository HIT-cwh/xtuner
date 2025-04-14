# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributed._tensor import DTensor

from xtuner._lite.accelerate.float8_gmm.distributed_utils import (
    tensor_already_casted_to_fp8,
)
from xtuner._lite.accelerate.float8_gmm.float8_tensor import (
    Float8Tensor,
    GemmInputRole,
    LinearMMConfig,
    ScaledMMConfig,
)
from xtuner._lite.accelerate.float8_gmm.float8_utils import to_fp8_saturated
from xtuner._lite.accelerate.float8_gmm.fsdp_utils import (
    WeightWithDynamicChannelwiseFloat8CastTensorGMM,
)
from xtuner._lite.accelerate.float8_gmm.triton_kernels import (
    gmm_dw_fp8_act_per_channel_w_per_expert,
    gmm_fp8_act_per_channel_w_per_expert,
    trans_quant_expand_128x,
)
from xtuner._lite.accelerate.float8_gmm.fsdp_utils import (
    WeightWithDynamicFloat8CastTensor,
)


@torch.no_grad()
def per_channel_quant_fp8(x, eps=1e-12, quant_dtype=torch.float8_e4m3fn):
    x_amax = x.abs().amax(-1, True).to(torch.float64)
    x_scales = torch.clamp(x_amax, min=eps) / torch.finfo(quant_dtype).max
    x_scales = x_scales.to(torch.float32)
    x_quanted = to_fp8_saturated(x.float() / x_scales, quant_dtype)
    return x_quanted, x_scales


@torch.no_grad()
def per_channel_trans_quant_fp8(x, eps=1e-12, quant_dtype=torch.float8_e4m3fn):
    x_trans = x.transpose(0, 1).contiguous()
    return per_channel_quant_fp8(x_trans, eps, quant_dtype)


class weight_to_float8_dynamic(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        w: torch.Tensor,
        float8_dtype: torch.dtype,
        linear_mm_config,
        gemm_input_role=GemmInputRole.WEIGHT,
    ):
        amax = torch.max(torch.abs(w))
        scale = amax.float() / torch.finfo(float8_dtype).max
        w_scaled = w.float() / scale
        w_bits_fp8 = to_fp8_saturated(w_scaled, float8_dtype)
        return Float8Tensor(
            w_bits_fp8,
            scale,
            w.dtype,
            linear_mm_config=linear_mm_config,
            gemm_input_role=gemm_input_role,
        )

    @staticmethod
    def backward(ctx, g):
        return g, None, None, None


@torch.library.custom_op("moe::dw_backward", mutates_args=("tokens_per_expert",))
def dw_backward(
    x_trans_fp8: Tensor,
    x_trans_scale: Tensor,
    grad_output_trans_fp8: Tensor,
    grad_output_trans_scale: Tensor,
    tokens_per_expert: Tensor,
) -> Tensor:
    grad_output_trans_scale = (
        grad_output_trans_scale.contiguous()
        .transpose(0, 1)
        .contiguous()
        .transpose(0, 1)
    )
    x_trans_scale = (
        x_trans_scale.contiguous().transpose(0, 1).contiguous().transpose(0, 1)
    )
    out = gmm_dw_fp8_act_per_channel_w_per_expert(
        grad_output_trans_fp8,
        grad_output_trans_scale,
        x_trans_fp8,
        x_trans_scale,
        tokens_per_expert,
    )

    dout = grad_output_trans_fp8.shape[0]
    din = x_trans_fp8.shape[0]
    ne = x_trans_scale.shape[-1]
    out = out.view(ne, dout, din)
    return out


@dw_backward.register_fake
def _(
    x_trans_fp8: Tensor,
    x_trans_scale: Tensor,
    grad_output_trans_fp8: Tensor,
    grad_output_trans_scale: Tensor,
    tokens_per_expert: Tensor,
) -> Tensor:
    ne = x_trans_scale.shape[-1]
    dout = grad_output_trans_fp8.shape[0]
    din = x_trans_fp8.shape[0]
    out = torch.empty((ne, dout, din), dtype=torch.bfloat16, device="cuda")
    return out


class fp8_matmul_weight_per_expert_act_per_channel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_fp8):

        assert x.shape[0] == 1
        x = x.squeeze(0)

        x_fp8, x_scale = per_channel_quant_fp8(x)
        x_trans_fp8, x_trans_scale = per_channel_trans_quant_fp8(x, quant_dtype=torch.float8_e4m3fn)

        dout, din = w_fp8.shape
        w_fp8_data = w_fp8._data.unsqueeze(0)
        w_fp8_scale = w_fp8._scale.view(1, 1).repeat(1, dout)
        tokens_per_expert = torch.tensor([x.shape[0]], device=x.device, dtype=torch.int)
        # import torch.distributed as dist
        # dist.breakpoint()

        out = gmm_fp8_act_per_channel_w_per_expert(
            x_fp8, x_scale, w_fp8_data, w_fp8_scale, tokens_per_expert, torch.bfloat16
        )

        ctx.save_for_backward(
            x_trans_fp8,
            x_trans_scale,
            w_fp8,
        )
        return out.unsqueeze(0)

    @staticmethod
    def backward(ctx, grad_output_hp):
        assert grad_output_hp.shape[0] == 1
        grad_output_hp = grad_output_hp.squeeze(0)
        (
            x_trans_fp8,
            x_trans_scale,
            w_fp8,
        ) = ctx.saved_tensors
        tokens_per_expert = torch.tensor([grad_output_hp.shape[0]], device=grad_output_hp.device, dtype=torch.int)

        grad_output_trans_fp8, grad_output_trans_scale = per_channel_trans_quant_fp8(grad_output_hp, quant_dtype=torch.float8_e5m2)

        grad_output_fp8, grad_output_scale = per_channel_quant_fp8(
            grad_output_hp, quant_dtype=torch.float8_e5m2
        )

        dout, din = w_fp8.shape

        w_fp8_data = w_fp8._data.transpose(0, 1).contiguous().unsqueeze(0)
        w_fp8_scale = w_fp8._scale.view(1, 1).repeat(1, din)
        dx = gmm_fp8_act_per_channel_w_per_expert(
            grad_output_fp8,
            grad_output_scale,
            w_fp8_data,
            w_fp8_scale,
            tokens_per_expert,
            torch.bfloat16,
        )
        dx = dx.unsqueeze(0)

        dw = dw_backward(
            x_trans_fp8,
            x_trans_scale,
            grad_output_trans_fp8,
            grad_output_trans_scale,
            tokens_per_expert,
        )

        return dx, dw, None, None, None


class ChannelWiseFloat8Linear(torch.nn.Linear):
    def __init__(self, *args, **kwargs):
        """Additional arguments on top of `torch.nn.Linear`'s arguments:

        * `config`: Float8LinearConfig
        """

        super().__init__(*args, **kwargs)

        self.linear_mm_config = LinearMMConfig(
            # output
            ScaledMMConfig(
                emulate=False,
                use_fast_accum=True,
                fp8_output=False,
                pad_inner_dim=False,
            ),
            # grad_input
            ScaledMMConfig(
                emulate=False,
                use_fast_accum=False,
                fp8_output=False,
                pad_inner_dim=False,
            ),
            # grad_weight
            ScaledMMConfig(
                emulate=False,
                use_fast_accum=False,
                fp8_output=False,
                pad_inner_dim=False,
            ),
        )

    def cast_weight_to_float8(self, weight):
        if tensor_already_casted_to_fp8(weight):
            return weight
        return weight_to_float8_dynamic.apply(
            weight, torch.float8_e4m3fn, self.linear_mm_config
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight_fp8 = self.cast_weight_to_float8(
            self.weight.to_local() if isinstance(self.weight, DTensor) else self.weight
        )
        out = fp8_matmul_weight_per_expert_act_per_channel.apply(
            input, weight_fp8
        )
        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}"
        )

    @classmethod
    def from_float(cls, mod):
        with torch.device("meta"):
            new_mod = cls(
                mod.in_features,
                mod.out_features,
                bias=False,
            )
        new_mod.bias = mod.bias
        new_mod.weight = torch.nn.Parameter(
            WeightWithDynamicFloat8CastTensor(
                mod.weight,
                new_mod.linear_mm_config,
                torch.float8_e4m3fn,
            )
        )
        return new_mod
