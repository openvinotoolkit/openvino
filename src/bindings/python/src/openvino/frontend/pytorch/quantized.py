# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Callable

import torch
from openvino.frontend.pytorch import ModuleExtension, gptq
from openvino.frontend.pytorch.patch_model import (
    patch_model, unpatch_model, patch_model_for_export)


def detect_quantized_model(model: torch.nn.Module) -> str | None:
    """Detects the quantization method used in a given PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model to check for quantization.

    Returns:
        str: The quantization method if available, otherwise None.
    """
    if (model and getattr(model, "config", None)
            and getattr(model.config, "quantization_config", None)):
        return model.config.quantization_config.quant_method  # type: ignore
    if getattr(model, "model", None):
        return detect_quantized_model(model.model)  # type: ignore[arg-type]
    return None


def patch_quantized(model: torch.nn.Module) -> None:
    """Patches a model based on its quantization type ("awq" or "gptq").

    Args:
        model (torch.nn.Module): The model to patch.

    Raises:
        RuntimeError: If the quantization type is unknown.
    """
    quant_type = detect_quantized_model(model)
    if quant_type == "gptq":
        model._openvino_gptq_patched = True  # type: ignore[assignment]
        gptq.patch_model(model)  # type: ignore
        return

    extensions = _build_quantized_extensions(quant_type)
    patch_model(model, extensions,
                "_openvino_quantized_patch_orig_forward")  # type: ignore


def unpatch_quantized(model: torch.nn.Module) -> None:
    """Reverts the patching applied to a quantized PyTorch model.

    Args:
        model (torch.nn.Module): The model to unpatch.
    """
    if getattr(model, "_openvino_gptq_patched", False):
        gptq.unpatch_model(model)  # type: ignore
        del model._openvino_gptq_patched
    else:
        unpatch_model(model,
                      "_openvino_quantized_patch_orig_forward")  # type: ignore


def _build_quantized_extensions(
    quant_type: str | None, for_export: bool = False,
) -> dict[Any, ModuleExtension] | None:
    """Build ModuleExtension dict for the given quantization type.

    Args:
        quant_type: Quantization method string ("awq", "bitnet", etc.).
        for_export: If True, pass group_size/w_bit as plain ints for
            ``torch.export``.  If False (default), wrap them in
            ``torch.tensor`` so they appear as graph inputs in
            TorchScript tracing.

    Returns the extensions dict (may be empty if the required package is
    not installed).  Raises ``RuntimeError`` for unknown quant types.
    """
    def fp32_tensor(*shape: int) -> torch.Tensor:
        return torch.full(shape, 0.5, dtype=torch.float32)

    extensions: dict[Any, ModuleExtension] = {}
    if quant_type == "awq":
        try:
            from awq.modules.linear import WQLinear_GEMM

            def _awq_convert(
                module: Any, target_op: Callable[..., torch.Tensor],
                *args: Any, **kwargs: Any,
            ) -> torch.Tensor:
                gs = module.group_size if for_export else torch.tensor(module.group_size)
                wb = module.w_bit if for_export else torch.tensor(module.w_bit)
                return target_op(
                    args[0], module.qweight, module.qzeros, module.scales,
                    gs, wb, module.bias)

            extensions[WQLinear_GEMM] = ModuleExtension(
                WQLinear_GEMM, "ov_ext::awq_gemm",
                convert=_awq_convert,
                evaluate=lambda module, *args, **kwargs: fp32_tensor(
                    *args[0].shape[:-1], module.out_features))  # type: ignore
        except ImportError:
            pass
    elif quant_type == "bitnet":
        try:
            from transformers.integrations.bitnet import AutoBitLinear
            extensions[AutoBitLinear] = ModuleExtension(
                AutoBitLinear, "ov_ext::bit_linear",
                convert=lambda module, target_op, *args, **kwargs: target_op(
                    module.rms_norm(
                        args[0]) if module.rms_norm is not None else args[0],
                    getattr(module, "original_weight", module.weight),
                    module.weight_scale,
                    module.bias),
                evaluate=lambda module, *args, **kwargs: fp32_tensor(
                    *args[0].shape[:-1], module.out_features))  # type: ignore
        except ImportError:
            pass
    elif quant_type == "gptq":
        # GPTQ is handled separately — not via ModuleExtension
        return None
    else:
        raise RuntimeError(f"Unknown quantization type: {quant_type}.")
    return extensions


# ──────────────────────────────────────────────────────────────────────
#  torch.export–compatible patching
# ──────────────────────────────────────────────────────────────────────

def patch_quantized_for_export(model: torch.nn.Module) -> None:
    """Patch a quantized model for ``torch.export`` using ``torch.library`` custom ops.

    This is the ``torch.export`` counterpart of ``patch_quantized``.
    GPTQ models are not supported in the export path and will raise
    ``RuntimeError``.

    Args:
        model (torch.nn.Module): The model to patch.

    Raises:
        RuntimeError: If the quantization type is unknown or unsupported.
    """
    quant_type = detect_quantized_model(model)
    extensions = _build_quantized_extensions(quant_type, for_export=True)
    if extensions is None:
        raise RuntimeError(
            "GPTQ models are not yet supported with torch.export. "
            "Use the TorchScript path (default) instead.")

    patch_model_for_export(model, extensions,
                           "_openvino_quantized_patch_orig_forward")  # type: ignore


def unpatch_quantized_for_export(model: torch.nn.Module) -> None:
    """Revert patching applied by ``patch_quantized_for_export``.

    Args:
        model (torch.nn.Module): The model to unpatch.
    """
    unpatch_model(model,
                  "_openvino_quantized_patch_orig_forward")  # type: ignore
