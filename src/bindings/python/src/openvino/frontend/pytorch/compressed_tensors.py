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
    """Build a ``ModuleExtension`` mapping for compressed-tensors pack-quantized weights.

    Converts int4 pack-quantized ``nn.Linear`` and ``nn.Embedding`` weights directly to
    ``ov_ext::ct_gemm`` / ``ov_ext::ct_embedding``, preserving weights as OV i4 (symmetric)
    or u4 (asymmetric) constants without decompressing to float first.  All unpacking and
    dequantization graph construction happens in the C++ PyTorch frontend translator.

    Supports symmetric and asymmetric group-quantized int4 (``num_bits=4``,
    ``strategy="group"``). Any packed module with an unsupported configuration
    (float formats such as nvfp4/mxfp4, other bit widths, non-group strategy) is
    still matched and raises a precise error, so it is never silently exported as dense.

    Handles two compressed-tensors variants:
    - Legacy (< 0.17): ``CompressedLinear`` subclass with ``weight_packed`` buffer.
    - Modern (>= 0.17): plain ``torch.nn.Linear`` / ``torch.nn.Embedding`` with
      ``weight_packed`` parameter and ``quantization_scheme`` set by ``apply_quantization_config``.

    Args:
        for_export: When ``True``, scalar arguments (``group_size``, ``sym``)
            are passed as plain Python values, as required by ``torch.export``.
            When ``False`` (TorchScript default), they are wrapped in ``torch.tensor``.

    Returns:
        Dict mapping module class(es) to ``ModuleExtension`` objects, or an empty
        dict if the ``compressed_tensors`` package is not installed.
    """
    try:
        import compressed_tensors  # noqa: F401
    except ImportError:
        log.warning(
            "compressed_tensors package not found; "
            "pack-quantized extensions will not be registered."
        )
        return {}

    # CompressionFormat is the canonical source of the format string, but fall
    # back to the literal so a stripped-down install (or a test stub) still works.
    try:
        from compressed_tensors.config.base import CompressionFormat
        pack_quantized_fmt = CompressionFormat.pack_quantized.value
    except ImportError:
        pack_quantized_fmt = "pack-quantized"

    def _get_weight_args(module: Any) -> Any:
        return getattr(getattr(module, "quantization_scheme", None), "weights", None)

    def _is_ct_pack_quantized(module: Any) -> bool:
        """Match any compressed-tensors module carrying packed quantized weights.

        Intentionally permissive: modules that carry ``weight_packed`` but use a
        configuration we cannot convert (wrong bit width, float type, ...) are
        still matched here so that ``_validate_weights`` can raise a precise
        error instead of the module being silently exported as dense float.
        """
        return hasattr(module, "weight_packed") and _get_weight_args(module) is not None

    def _validate_weights(module: Any, weight_args: Any) -> tuple[bool, int]:
        """Validate the weight quantization config, returning ``(sym, group_size)``.

        Raises ``ValueError`` with a precise reason for any unsupported config so
        it is never silently skipped. Only int4 group pack-quantized is supported.
        """
        # compressed-tensors stores these as ``str, Enum`` members; unwrap ``.value``
        # to the underlying string (``getattr(x, "value", x)`` also handles plain
        # strings / None and keeps mypy happy about the union type).
        fmt = getattr(getattr(module, "quantization_scheme", None), "format", None)
        w_type = getattr(weight_args, "type", None)
        w_type = str(getattr(w_type, "value", w_type))
        num_bits = getattr(weight_args, "num_bits", None)
        symmetric = getattr(weight_args, "symmetric", None)
        strategy = getattr(weight_args, "strategy", None)
        strategy = getattr(strategy, "value", strategy)
        group_size = getattr(weight_args, "group_size", None)

        name = type(module).__name__
        # ``scheme.format`` is only populated on the compress/decompress path; the
        # plain model-load path (and older CT / legacy CompressedLinear) leave it
        # None while still being valid pack-quantized. So enforce it only when set;
        # the type/num_bits/strategy checks below robustly reject float formats
        # (nvfp4/mxfp4 use type="float") and other unsupported configs regardless.
        fmt_str = getattr(fmt, "value", fmt)
        if fmt_str is not None and fmt_str != pack_quantized_fmt:
            raise ValueError(
                f"Unsupported compressed-tensors format={fmt_str!r} on {name}. "
                f"Only {pack_quantized_fmt!r} is supported by OpenVINO.")
        if w_type != "int":
            raise ValueError(
                f"Unsupported compressed-tensors weight type={w_type!r} on {name}. "
                "Only integer pack-quantized weights are supported (float "
                "formats such as nvfp4/mxfp4 are not).")
        if num_bits != 4:
            raise ValueError(
                f"Unsupported num_bits={num_bits} on {name}. Only num_bits=4 is supported.")
        if symmetric is None:
            raise ValueError(f"{name} is missing quantization_scheme.weights.symmetric.")
        if strategy != "group":
            raise ValueError(
                f"Unsupported strategy={strategy!r} on {name}. Only strategy='group' is supported.")
        if group_size is None:
            raise ValueError(f"{name} is missing quantization_scheme.weights.group_size.")
        return bool(symmetric), group_size

    def _packed_args(module: Any, weight_args: Any) -> tuple[Any, Any, Any, Any, Any]:
        """Common CT buffers → ``(weight_packed, weight_scale, gs, sym, zero_point)``.

        Buffers are passed raw; all unpacking happens in the C++ translator.
          weight_packed:     [rows, cols//8]      int32  (8 nibbles/int32, low-nibble first)
          weight_scale:      [rows, n_groups]     float32
          weight_zero_point: [rows//8, n_groups]  int32  (asymmetric only, same packing)
        """
        sym_val, group_size = _validate_weights(module, weight_args)
        gs = group_size if for_export else torch.tensor(group_size)
        sym = sym_val if for_export else torch.tensor(sym_val)
        weight_scale = getattr(module, "weight_scale", None)
        if weight_scale is None:
            raise ValueError(f"{type(module).__name__} is missing weight_scale buffer.")
        zero_point = None
        if not sym_val:
            zero_point = getattr(module, "weight_zero_point", None)
            if zero_point is None:
                raise ValueError(
                    f"asymmetric {type(module).__name__} is missing weight_zero_point buffer.")
        return module.weight_packed, weight_scale, gs, sym, zero_point

    def _convert_linear(module: Any, target_op: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        weight_args = _get_weight_args(module)
        wp, ws, gs, sym, zp = _packed_args(module, weight_args)
        bias = getattr(module, "bias", None)
        return target_op(args[0], wp, ws, gs, sym, zp, bias)

    def _convert_embedding(module: Any, target_op: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        weight_args = _get_weight_args(module)
        wp, ws, gs, sym, zp = _packed_args(module, weight_args)
        return target_op(wp, ws, gs, sym, args[0], zp)

    def _evaluate_linear(module: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        return torch.full(
            (*args[0].shape[:-1], module.out_features), 0.5,
            dtype=torch.float32, device=args[0].device)

    def _evaluate_embedding(module: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        return torch.full(
            (*args[0].shape, module.embedding_dim), 0.5,
            dtype=torch.float32, device=args[0].device)

    extensions: dict[Any, Any] = {}

    # Modern compressed-tensors (>= 0.17): plain nn.Linear / nn.Embedding, gated by condition.
    extensions[torch.nn.Linear] = ModuleExtension(
        torch.nn.Linear, "ov_ext::ct_gemm",
        convert=_convert_linear, evaluate=_evaluate_linear,
        condition=_is_ct_pack_quantized,
    )
    extensions[torch.nn.Embedding] = ModuleExtension(
        torch.nn.Embedding, "ov_ext::ct_embedding",
        convert=_convert_embedding, evaluate=_evaluate_embedding,
        condition=_is_ct_pack_quantized,
    )

    # Legacy compressed-tensors (< 0.17): CompressedLinear subclass (absent in newer releases).
    try:
        from compressed_tensors.linear.compressed_linear import CompressedLinear
        legacy_linear_cls = CompressedLinear
    except ImportError:
        legacy_linear_cls = None

    # Skip the alias case so it can't clobber the conditioned modern nn.Linear entry.
    if legacy_linear_cls is not None and legacy_linear_cls is not torch.nn.Linear:
        extensions[legacy_linear_cls] = ModuleExtension(
            legacy_linear_cls, "ov_ext::ct_gemm",
            convert=_convert_linear, evaluate=_evaluate_linear,
        )

    return extensions
