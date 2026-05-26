# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations


import functools
import logging
from collections.abc import Callable
from typing import Any

import torch
from openvino.frontend.pytorch import ModuleExtension, gptq
from openvino.frontend.pytorch.patch_model import (
    patch_model, unpatch_model, patch_model_for_export)

log = logging.getLogger(__name__)


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
        # GPTQ is handled separately — not via ModuleExtension for TorchScript,
        # and via attribute-based patching for torch.export (see _patch_gptq_for_export).
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

    Args:
        model (torch.nn.Module): The model to patch.

    Raises:
        RuntimeError: If the quantization type is unknown or unsupported.
    """
    quant_type = detect_quantized_model(model)

    if quant_type == "gptq":
        _patch_gptq_for_export(model)
        return

    extensions = _build_quantized_extensions(quant_type, for_export=True)
    patch_model_for_export(model, extensions,
                           "_openvino_quantized_patch_orig_forward")  # type: ignore


def _detect_gptq_sym(model: torch.nn.Module) -> bool:
    """Detect whether a GPTQ model uses symmetric quantization."""
    config = getattr(model, "config", None)
    if config is None and hasattr(model, "model"):
        config = getattr(model.model, "config", None)
    if config is not None and hasattr(config, "quantization_config"):
        return bool(getattr(config.quantization_config, "sym", False))
    return False


def _patch_gptq_for_export(model: torch.nn.Module) -> None:
    """Patch GPTQ modules for ``torch.export`` using ``ov_ext::gptq_gemm``.

    Unlike AWQ/BitNet which use ``ModuleExtension``, GPTQ modules are
    discovered by attribute (``QUANT_TYPE``) since there is no common base
    class across AutoGPTQ backends.
    """
    import openvino.frontend.pytorch.ov_custom_ops  # noqa: F401

    sym = _detect_gptq_sym(model)
    target_op = torch.ops.ov_ext.gptq_gemm

    for name, module in model.named_modules():
        if hasattr(module, "_openvino_quantized_patch_orig_forward"):
            log.debug("Skipping already-patched GPTQ module %s", name)
            continue
        if not hasattr(module, "QUANT_TYPE"):
            continue
        if module.QUANT_TYPE not in gptq.supported_quant_types:
            raise ValueError(
                f"Unsupported QUANT_TYPE == {module.QUANT_TYPE} in module "
                f"{name}. Supported types: {gptq.supported_quant_types}")
        if module.bits != 4:
            raise ValueError(
                f"Unsupported bits == {module.bits} in module {name}, "
                "only bits == 4 is supported.")

        module._openvino_quantized_patch_orig_forward = module.forward

        def _make_forward(mod: torch.nn.Module) -> Callable[..., torch.Tensor]:
            @functools.wraps(mod.forward)
            def new_forward(*args: Any, **kwargs: Any) -> torch.Tensor:
                return target_op(
                    args[0], mod.qweight, mod.qzeros, mod.scales,
                    mod.group_size, mod.bits, sym,
                    mod.bias)
            return new_forward

        module.forward = _make_forward(module)


def unpatch_quantized_for_export(model: torch.nn.Module) -> None:
    """Revert patching applied by ``patch_quantized_for_export``.

    Args:
        model (torch.nn.Module): The model to unpatch.
    """
    unpatch_model(model,
                  "_openvino_quantized_patch_orig_forward")  # type: ignore
