# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Register OpenVINO custom ops via torch.library for use with torch.export.

When converting quantized or 16-bit PyTorch models, module forwards are
replaced with calls to these custom ops.  During ``torch.export``, the ops
appear as ``call_function`` nodes in the FX graph (e.g.
``ov_ext.linear.default``) and are later translated by the C++ PyTorch
frontend using the same translators as the TorchScript ``ov_ext::*`` ops.

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
#  ov_ext::linear  (input, weight, bias?) → Tensor
# ──────────────────────────────────────────────────────────────────────
_ov_ext_lib.define(
    "linear(Tensor input, Tensor weight, Tensor? bias) -> Tensor")


@torch.library.impl(_ov_ext_lib, "linear", "Meta")
def _linear_meta(data: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    out_features = weight.shape[0]
    return torch.empty(
        *data.shape[:-1], out_features,
        dtype=torch.float32, device="meta")


@torch.library.impl(_ov_ext_lib, "linear", "CPU")
def _linear_cpu(data: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    out = torch.mm(
        data.reshape(-1, data.shape[-1]).float(),
        weight.float().t())
    if bias is not None:
        out = out + bias.float()
    return out.reshape(*data.shape[:-1], weight.shape[0])


# ──────────────────────────────────────────────────────────────────────
#  ov_ext::embedding  (weight, indices, padding_idx?, scale_grad, sparse) → Tensor
# ──────────────────────────────────────────────────────────────────────
_ov_ext_lib.define(
    "embedding(Tensor weight, Tensor indices, int? padding_idx, "
    "bool scale_grad_by_freq, bool sparse) -> Tensor")


@torch.library.impl(_ov_ext_lib, "embedding", "Meta")
def _embedding_meta(
    weight: torch.Tensor, indices: torch.Tensor, padding_idx: int | None,
    scale_grad_by_freq: bool, sparse: bool,
) -> torch.Tensor:
    return torch.empty(
        *indices.shape, weight.shape[1],
        dtype=torch.float32, device="meta")


@torch.library.impl(_ov_ext_lib, "embedding", "CPU")
def _embedding_cpu(
    weight: torch.Tensor, indices: torch.Tensor, padding_idx: int | None,
    scale_grad_by_freq: bool, sparse: bool,
) -> torch.Tensor:
    return torch.nn.functional.embedding(
        indices, weight.float(),
        padding_idx=padding_idx,
        scale_grad_by_freq=scale_grad_by_freq,
        sparse=sparse)


# ──────────────────────────────────────────────────────────────────────
#  ov_ext::conv1d  (input, weight, bias) → Tensor
#  (this is HuggingFace transformers.pytorch_utils.Conv1D, NOT nn.Conv1d)
# ──────────────────────────────────────────────────────────────────────
_ov_ext_lib.define(
    "conv1d(Tensor input, Tensor weight, Tensor? bias) -> Tensor")


@torch.library.impl(_ov_ext_lib, "conv1d", "Meta")
def _conv1d_meta(data: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    # Output feature count comes from the second dimension of weight.
    out_features = weight.shape[1]
    return torch.empty(
        *data.shape[:-1], out_features,
        dtype=torch.float32, device="meta")


@torch.library.impl(_ov_ext_lib, "conv1d", "CPU")
def _conv1d_cpu(data: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    out = torch.mm(
        data.reshape(-1, data.shape[-1]).float(),
        weight.float())
    if bias is not None:
        out = out + bias.float()
    return out.reshape(*data.shape[:-1], weight.shape[1])


# ──────────────────────────────────────────────────────────────────────
#  ov_ext::bmm  (batch1, batch2) → Tensor
# ──────────────────────────────────────────────────────────────────────
_ov_ext_lib.define("bmm(Tensor batch1, Tensor batch2) -> Tensor")


@torch.library.impl(_ov_ext_lib, "bmm", "Meta")
def _bmm_meta(batch1: torch.Tensor, batch2: torch.Tensor) -> torch.Tensor:
    return torch.empty(
        batch1.shape[0], batch1.shape[1], batch2.shape[2],
        dtype=torch.float32, device="meta")


@torch.library.impl(_ov_ext_lib, "bmm", "CPU")
def _bmm_cpu(batch1: torch.Tensor, batch2: torch.Tensor) -> torch.Tensor:
    return torch.bmm(batch1.float(), batch2.float())


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
    data: torch.Tensor, qweight: torch.Tensor, qzeros: torch.Tensor,
    scales: torch.Tensor, group_size: int, w_bit: int,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    pack_num = 32 // w_bit
    out_features = qweight.shape[1] * pack_num
    return torch.empty(
        *data.shape[:-1], out_features,
        dtype=torch.float32, device="meta")


@torch.library.impl(_ov_ext_lib, "awq_gemm", "CPU")
def _awq_gemm_cpu(
    data: torch.Tensor, qweight: torch.Tensor, qzeros: torch.Tensor,
    scales: torch.Tensor, group_size: int, w_bit: int,
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
    return out


# ──────────────────────────────────────────────────────────────────────
#  ov_ext::bit_linear  (input, weight, weight_scale, bias?) → Tensor
# ──────────────────────────────────────────────────────────────────────
_ov_ext_lib.define(
    "bit_linear(Tensor input, Tensor weight, Tensor weight_scale, "
    "Tensor? bias) -> Tensor")


@torch.library.impl(_ov_ext_lib, "bit_linear", "Meta")
def _bit_linear_meta(
    data: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    # BitNet weight packing: out_features is weight.shape[0]
    out_features = weight.shape[0]
    return torch.empty(
        *data.shape[:-1], out_features,
        dtype=torch.float32, device="meta")


@torch.library.impl(_ov_ext_lib, "bit_linear", "CPU")
def _bit_linear_cpu(
    data: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    out_features = weight.shape[0]
    out = torch.zeros(
        *data.shape[:-1], out_features,
        dtype=torch.float32, device=data.device)
    if bias is not None:
        out = out + bias.float()
    return out


log.debug("Registered ov_ext custom ops for torch.export")
