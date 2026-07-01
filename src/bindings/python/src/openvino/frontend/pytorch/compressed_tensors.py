# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import Any

import torch
from openvino.frontend.pytorch import ModuleExtension

log = logging.getLogger(__name__)


def build_extensions(for_export: bool = False) -> dict[Any, Any]:
    """Build a ``ModuleExtension`` mapping for compressed-tensors ``CompressedLinear``.

    Converts int4 pack-quantized weights directly to ``ov_ext::ct_gemm``, preserving
    weights as OV i4 (symmetric) or u4 (asymmetric) constants without decompressing
    to float first.  All unpacking and dequantization graph construction happens in
    the C++ PyTorch frontend translator.

    Supports symmetric and asymmetric group-quantized int4 (``num_bits=4``,
    ``strategy="group"``). An exception is raised for unsupported configurations
    so that they are never silently skipped.

    Args:
        for_export: When ``True``, scalar arguments (``group_size``, ``sym``)
            are passed as plain Python values, as required by ``torch.export``.
            When ``False`` (TorchScript default), they are wrapped in ``torch.tensor``.

    Returns:
        Dict mapping the ``CompressedLinear`` class to a ``ModuleExtension``, or an
        empty dict if the ``compressed_tensors`` package is not installed.
    """
    try:
        from compressed_tensors.linear.compressed_linear import CompressedLinear
    except ImportError:
        log.warning(
            "compressed_tensors package not found; "
            "CompressedLinear extensions will not be registered."
        )
        return {}

    def _convert(
        module: Any,
        target_op: Any,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        weight_args = getattr(
            getattr(module, "quantization_scheme", None), "weights", None
        )
        if weight_args is None:
            raise ValueError(
                "CompressedLinear module is missing quantization_scheme.weights; "
                "cannot export to OpenVINO int4."
            )

        num_bits = getattr(weight_args, "num_bits", None)
        symmetric = getattr(weight_args, "symmetric", None)
        strategy = getattr(weight_args, "strategy", None)
        group_size = getattr(weight_args, "group_size", None)

        if num_bits != 4:
            raise ValueError(
                f"Unsupported num_bits={num_bits} in compressed-tensors module. "
                "Only num_bits=4 is supported."
            )
        if symmetric is None:
            raise ValueError(
                "CompressedLinear module is missing quantization_scheme.weights.symmetric."
            )
        if strategy != "group":
            raise ValueError(
                f"Unsupported strategy={strategy!r} in compressed-tensors module. "
                "Only strategy='group' is supported."
            )
        if group_size is None:
            raise ValueError(
                "CompressedLinear module is missing quantization_scheme.weights.group_size."
            )

        sym_val = bool(symmetric)
        gs = group_size if for_export else torch.tensor(group_size)
        sym = sym_val if for_export else torch.tensor(sym_val)

        # Pass raw CT buffers directly — all unpacking happens in C++ translate_linear_ct.
        # weight_packed:      [out, in//8]     int32  (8 nibbles/int32, low-nibble first)
        # weight_scale:       [out, n_groups]  float32
        # weight_zero_point:  [out//8, n_groups] int32 (asym only, same nibble packing)
        bias = getattr(module, "bias", None)
        if sym_val:
            return target_op(args[0], module.weight_packed, module.weight_scale,
                             gs, sym, None, bias)
        else:
            return target_op(args[0], module.weight_packed, module.weight_scale,
                             gs, sym, module.weight_zero_point, bias)

    return {
        CompressedLinear: ModuleExtension(
            CompressedLinear,
            "ov_ext::ct_gemm",
            convert=_convert,
            evaluate=lambda module, *args, **kwargs: torch.full(
                (*args[0].shape[:-1], module.out_features), 0.5,
                dtype=torch.float32, device=args[0].device),
        )
    }
