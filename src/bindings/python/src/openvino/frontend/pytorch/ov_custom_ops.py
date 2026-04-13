# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Register OpenVINO custom ops via torch.library for use with torch.export.

When converting quantized PyTorch models, module forwards are replaced with
calls to these custom ops.  During ``torch.export``, the ops appear as
``call_function`` nodes in the FX graph (e.g. ``ov_ext.awq_gemm.default``)
and are later translated by the C++ PyTorch frontend.

Import this module to trigger registration::

    import openvino.frontend.pytorch.ov_custom_ops  # noqa: F401
"""

from __future__ import annotations

import logging

import torch

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
#  Library handle – one per process, idempotent on reload
# ──────────────────────────────────────────────────────────────────────
_ov_ext_lib = torch.library.Library("ov_ext", "DEF")


# ──────────────────────────────────────────────────────────────────────
#  ov_ext::awq_gemm  (input, qweight, qzeros, scales, group_size,
#                      w_bit, bias?) → Tensor
# ──────────────────────────────────────────────────────────────────────
_ov_ext_lib.define(
    "awq_gemm(Tensor input, Tensor qweight, Tensor qzeros, "
    "Tensor scales, int group_size, int w_bit, "
    "Tensor? bias) -> Tensor")


@torch.library.impl(_ov_ext_lib, "awq_gemm", "Meta")
def _awq_gemm_meta(
    data: torch.Tensor, qweight: torch.Tensor, _qzeros: torch.Tensor,
    _scales: torch.Tensor, _group_size: int, w_bit: int,
    _bias: torch.Tensor | None,
) -> torch.Tensor:
    pack_num = 32 // w_bit
    out_features = qweight.shape[1] * pack_num
    return torch.empty(
        *data.shape[:-1], out_features,
        dtype=data.dtype, device="meta")


@torch.library.impl(_ov_ext_lib, "awq_gemm", "CPU")
def _awq_gemm_cpu(
    data: torch.Tensor, qweight: torch.Tensor, _qzeros: torch.Tensor,
    _scales: torch.Tensor, _group_size: int, w_bit: int,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    # Placeholder – actual dequantisation happens in the C++ OV translator.
    # This fallback produces the right shape for tracing / testing.
    pack_num = 32 // w_bit
    out_features = qweight.shape[1] * pack_num
    out = torch.zeros(
        *data.shape[:-1], out_features,
        dtype=torch.float32, device=data.device)
    if bias is not None:
        out = out + bias.float()
    return out.to(data.dtype)


# ──────────────────────────────────────────────────────────────────────
#  ov_ext::bit_linear  (input, weight, weight_scale, bias?) → Tensor
# ──────────────────────────────────────────────────────────────────────
_ov_ext_lib.define(
    "bit_linear(Tensor input, Tensor weight, Tensor weight_scale, "
    "Tensor? bias) -> Tensor")


@torch.library.impl(_ov_ext_lib, "bit_linear", "Meta")
def _bit_linear_meta(
    data: torch.Tensor, weight: torch.Tensor, _weight_scale: torch.Tensor,
    _bias: torch.Tensor | None,
) -> torch.Tensor:
    # BitNet weight packing: out_features is weight.shape[0]
    out_features = weight.shape[0]
    return torch.empty(
        *data.shape[:-1], out_features,
        dtype=data.dtype, device="meta")


@torch.library.impl(_ov_ext_lib, "bit_linear", "CPU")
def _bit_linear_cpu(
    data: torch.Tensor, weight: torch.Tensor, _weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    out_features = weight.shape[0]
    out = torch.zeros(
        *data.shape[:-1], out_features,
        dtype=torch.float32, device=data.device)
    if bias is not None:
        out = out + bias.float()
    return out.to(data.dtype)


# ──────────────────────────────────────────────────────────────────────
#  ov_ext::gptq_gemm  (input, qweight, qzeros, scales, group_size,
#                       w_bit, sym, bias?) → Tensor
# ──────────────────────────────────────────────────────────────────────
_ov_ext_lib.define(
    "gptq_gemm(Tensor input, Tensor qweight, Tensor qzeros, "
    "Tensor scales, int group_size, int w_bit, bool sym, "
    "Tensor? bias) -> Tensor")


@torch.library.impl(_ov_ext_lib, "gptq_gemm", "Meta")
def _gptq_gemm_meta(
    data: torch.Tensor, qweight: torch.Tensor, _qzeros: torch.Tensor,
    _scales: torch.Tensor, _group_size: int, w_bit: int, _sym: bool,
    _bias: torch.Tensor | None,
) -> torch.Tensor:
    pack_num = 32 // w_bit
    out_features = qweight.shape[1] * pack_num
    return torch.empty(
        *data.shape[:-1], out_features,
        dtype=data.dtype, device="meta")


@torch.library.impl(_ov_ext_lib, "gptq_gemm", "CPU")
def _gptq_gemm_cpu(
    data: torch.Tensor, qweight: torch.Tensor, _qzeros: torch.Tensor,
    _scales: torch.Tensor, _group_size: int, w_bit: int, _sym: bool,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    # Placeholder – actual dequantisation happens in the C++ OV translator.
    pack_num = 32 // w_bit
    out_features = qweight.shape[1] * pack_num
    out = torch.zeros(
        *data.shape[:-1], out_features,
        dtype=torch.float32, device=data.device)
    if bias is not None:
        out = out + bias.float()
    return out.to(data.dtype)


log.debug("Registered ov_ext custom ops for torch.export")
