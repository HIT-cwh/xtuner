# Copyright (c) OpenMMLab. All rights reserved.

from typing import Any, Dict, Tuple
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
    per_block_quant_gemm,
    per_tile_quant_gemm,
    trans_per_block_quant_gemm,
    trans_per_tile_quant_gemm,
)
from xtuner._lite.accelerate.float8_gmm.fsdp_utils import (
    WeightWithDynamicFloat8CastTensor,
)
from xtuner._lite.accelerate.float8_gmm.fsdp_utils import (
    WeightWithDynamicTilewiseFloat8CastTensor,
)
import torch.distributed as dist



DEEPGEMM_INSTALLED = False

try:
    from deep_gemm import (
        k_grouped_gemm_dw_fp8_fp8_bf16_tn_contiguous,
        m_grouped_varlen_gemm_fp8_fp8_bf16_nt_contiguous,
        gemm_fp8_fp8_bf16_nt,
    )

    DEEPGEMM_INSTALLED = True
except ImportError:
    deep_gemm = None


def _get_min_alignment(size: int, alignment_value: int) -> int:
    return (1 + ((size - 1) // alignment_value)) * alignment_value


@torch.no_grad()
def per_tile_quant(x, eps=1e-12, quant_dtype=torch.float8_e4m3fn):
    seq, dim = x.shape
    x = x.view(-1, 128)
    x_amax = x.abs().amax(-1, True).to(torch.float64)
    x_scales = torch.clamp(x_amax, min=eps) / torch.finfo(quant_dtype).max
    x_scales = x_scales.to(torch.float32)
    x_quanted = to_fp8_saturated(x.float() / x_scales, quant_dtype)
    x_quanted = x_quanted.view(seq, dim)
    x_scales = x_scales.view(seq, -1)
    return x_quanted, x_scales


@torch.no_grad()
def per_tile_trans_quant(x, eps=1e-12, quant_dtype=torch.float8_e4m3fn):
    x_trans = x.transpose(0, 1).contiguous()
    return per_tile_quant(x_trans, eps, quant_dtype)


@torch.no_grad()
def per_block_quant(x, eps=1e-12, quant_dtype=torch.float8_e4m3fn):
    dout, din = x.shape
    block_size = 128
    x = (
        x.view(dout // block_size, block_size, din // block_size, block_size)
        .transpose(1, 2)
        .reshape(-1, block_size * block_size)
    )
    x_amax = x.abs().amax(-1, True).to(torch.float64)
    x_scales = torch.clamp(x_amax, min=eps) / torch.finfo(quant_dtype).max
    x_scales = x_scales.to(torch.float32)
    x_quanted = to_fp8_saturated(x.float() / x_scales, quant_dtype)

    x_quanted = (
        x_quanted.view(dout // block_size, din // block_size, block_size, block_size)
        .transpose(1, 2)
        .reshape(dout, din)
    )
    x_scales = x_scales.view(dout // block_size, din // block_size)
    return x_quanted, x_scales


@torch.no_grad()
def per_block_trans_quant(x, eps=1e-12, quant_dtype=torch.float8_e4m3fn):
    x_trans = x.transpose(0, 1).contiguous()
    return per_block_quant(x_trans, eps, quant_dtype)


class weight_to_per_block_float8_dynamic(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        w: torch.Tensor,  # ne, dout, din
        float8_dtype: torch.dtype,
        linear_mm_config,
        gemm_input_role=GemmInputRole.WEIGHT,
    ):
        ne, dout, din = w.shape
        block_size = 128
        w = (
            w.view(ne, dout // block_size, block_size, din // block_size, block_size)
            .transpose(2, 3)
            .reshape(-1, block_size * block_size)
        )
        w_amax = w.abs().amax(-1, True)
        w_scale = w_amax.float() / torch.finfo(float8_dtype).max
        w_scaled = w.float() / w_scale
        w_bits_fp8 = to_fp8_saturated(w_scaled, float8_dtype)
        w_bits_fp8 = (
            w_bits_fp8.view(
                ne, dout // block_size, din // block_size, block_size, block_size
            )
            .transpose(2, 3)
            .reshape(ne, dout, din)
        )
        w_scale = w_scale.view(ne, dout // block_size, din // block_size)

        return Float8Tensor(
            w_bits_fp8,
            w_scale,
            w.dtype,
            linear_mm_config=linear_mm_config,
            gemm_input_role=gemm_input_role,
        )

    @staticmethod
    def backward(ctx, g):
        return g, None, None, None


@torch.library.custom_op("moe::save", mutates_args=('grad_out_trans_fp8', 'dw', 'grad_out_trans_scale', 'x_trans_fp8', 'x_trans_scale'))
def save(
        dw: torch.Tensor, 
        grad_output_hp: torch.Tensor,
        grad_out_trans_fp8: torch.Tensor,
        grad_out_trans_scale: torch.Tensor,
        x_trans_fp8: torch.Tensor,
        x_trans_scale: torch.Tensor,
    ) -> None:
    # print(grad_output_hp)
    if torch.isnan(dw).any():
        torch.save(grad_output_hp, f'/cpfs01/shared/llm_razor/caoweihan/projects/xtuner_xpuyu_new/xpuyu_temp/pths/rank{dist.get_rank()}_grad_output_hp.pth')
        torch.save(grad_out_trans_fp8, f'/cpfs01/shared/llm_razor/caoweihan/projects/xtuner_xpuyu_new/xpuyu_temp/pths/rank{dist.get_rank()}_grad_out_trans_fp8.pth')
        torch.save(grad_out_trans_scale, f'/cpfs01/shared/llm_razor/caoweihan/projects/xtuner_xpuyu_new/xpuyu_temp/pths/rank{dist.get_rank()}_grad_out_trans_scale.pth')
        torch.save(x_trans_fp8, f'/cpfs01/shared/llm_razor/caoweihan/projects/xtuner_xpuyu_new/xpuyu_temp/pths/rank{dist.get_rank()}_x_trans_fp8.pth')
        torch.save(x_trans_scale.contiguous(), f'/cpfs01/shared/llm_razor/caoweihan/projects/xtuner_xpuyu_new/xpuyu_temp/pths/rank{dist.get_rank()}_x_trans_scale.pth')
        torch.save(dw, f'/cpfs01/shared/llm_razor/caoweihan/projects/xtuner_xpuyu_new/xpuyu_temp/pths/rank{dist.get_rank()}_dw.pth')
    
    assert not torch.isnan(dw).any()
    

@save.register_fake
def _(
        dw: torch.Tensor, 
        grad_output_hp: torch.Tensor,
        grad_out_trans_fp8: torch.Tensor,
        grad_out_trans_scale: torch.Tensor,
        x_trans_fp8: torch.Tensor,
        x_trans_scale: torch.Tensor,
) -> None:
    return 


class fp8_matmul_weight_per_block_act_per_tile(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_fp8):

        assert x.shape[0] == 1
        x = x.squeeze(0)
        seq, din = x.shape

        x_fp8, x_scale = per_tile_quant_gemm(x)
        x_trans_fp8, x_trans_scale = trans_per_block_quant_gemm(x)

        dout = w_fp8._data.shape[0]
        out = x.new_empty((seq, dout))
        gemm_fp8_fp8_bf16_nt(
            x_fp8, x_scale,
            w_fp8._data, w_fp8._scale,
            out
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
        
        dout, din = w_fp8.shape
        seq, dout = grad_output_hp.shape

        if dout % 128 != 0:
            dout_pad = _get_min_alignment(dout, 128)
            grad_output_hp_pad = torch.nn.functional.pad(grad_output_hp, (0, dout_pad - dout, 0, 0))
            w_fp8_data = torch.nn.functional.pad(w_fp8._data.transpose(0, 1).contiguous(), (0, dout_pad - dout, 0, 0))
        else:
            grad_output_hp_pad = grad_output_hp
            w_fp8_data = w_fp8._data.transpose(0, 1).contiguous()
        w_fp8_scale = w_fp8._scale.transpose(0, 1).contiguous()

        grad_out_pad_fp8, grad_out_pad_scale = per_tile_quant_gemm(grad_output_hp_pad)
        dx = grad_output_hp_pad.new_empty((seq, din))
        gemm_fp8_fp8_bf16_nt(
            grad_out_pad_fp8, grad_out_pad_scale,
            w_fp8_data, w_fp8_scale,
            dx
        )
        dx = dx.unsqueeze(0)

        grad_out_trans_fp8, grad_out_trans_scale = trans_per_tile_quant_gemm(grad_output_hp)
        dw = grad_output_hp.new_empty((dout, din))
        gemm_fp8_fp8_bf16_nt(
            grad_out_trans_fp8, grad_out_trans_scale,
            x_trans_fp8, x_trans_scale.contiguous(),
            dw
        )

        return dx, dw, None, None, None


class TileWiseFloat8Linear(torch.nn.Linear):
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
        raise NotImplementedError
        # return weight_to_float8_dynamic.apply(
        #     weight, torch.float8_e4m3fn, self.linear_mm_config
        # )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight_fp8 = self.cast_weight_to_float8(
            self.weight.to_local() if isinstance(self.weight, DTensor) else self.weight
        )
        out = fp8_matmul_weight_per_block_act_per_tile.apply(
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
            WeightWithDynamicTilewiseFloat8CastTensor(
                mod.weight,
                new_mod.linear_mm_config,
                torch.float8_e4m3fn,
                (new_mod.out_features, new_mod.in_features)
            )
        )
        return new_mod
