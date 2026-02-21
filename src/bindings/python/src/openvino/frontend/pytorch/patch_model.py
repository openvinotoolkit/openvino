# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# mypy: ignore-errors

import functools
import logging
import threading
import torch
from openvino.frontend.pytorch import ModuleExtension

log = logging.getLogger(__name__)


def patch_model(model, module_extensions, orig_forward_name):
    def module_patcher(module, name):
        extension = None
        if module in module_extensions:
            extension = module_extensions[module]
        elif module.__class__ in module_extensions:
            extension = module_extensions[module.__class__]
        elif name in module_extensions:
            extension = module_extensions[name]

        if extension and extension.condition(module):
            log.debug("Patching module %s", module)
            # The Trampoline class is instantiated for every module replacement, so we can use
            # class members individually for each module.

            class Trampoline(torch.autograd.Function):
                # required to be saved in class
                target_extension = extension

                @staticmethod
                @torch.jit.ignore
                def forward(ctx, *args, **kwargs):
                    # Temporarily restore the original forward function of `module` to avoid
                    # recursion issues in `evaluate`, then revert it back.
                    patched_forward = module.forward
                    # set original forward for the module
                    module.forward = getattr(module, orig_forward_name)
                    # call user code
                    results = extension.evaluate(module, *args, **kwargs)
                    module.forward = patched_forward  # return patched forward back
                    return results

            def new_forward(*args, **kwargs):
                return extension.convert(module, Trampoline.apply, *args, **kwargs)

            # make signature of new_forward same as of forward
            new_forward = functools.wraps(module.forward)(new_forward)
            setattr(module, orig_forward_name, module.forward)
            module.forward = new_forward

    for name, module in model.named_modules():
        if hasattr(module, orig_forward_name):
            # already patched, skipping. It may happen when patching applied for same module twice
            log.debug("Unexpectedly found already patched module %s while applying "
                      "ModuleExtension during PyTorch model conversion. "
                      "Result of the conversion maybe broken. Depending on the exact issue "
                      "it may lead to broken original model.", name)
            continue

        module_patcher(module, name)


def unpatch_model(model, orig_forward_name):
    # Restore patched torch functions (bmm, baddbmm, etc.)
    _unpatch_torch_functions()

    for _, module in model.named_modules():
        if hasattr(module, orig_forward_name):
            try:
                module.forward = getattr(module, orig_forward_name)
                delattr(module, orig_forward_name)
            except Exception as error:
                log.warning("Exception raised during model unpatching. "
                            "Depending on the exact issue it may lead to broken original model.\n"
                            "Original exception details:\n%s", error)


def _create_function_wrapper(extension):
    """Create a wrapper for a torch function using the same Trampoline pattern as modules."""

    class Trampoline(torch.autograd.Function):
        target_extension = extension

        @staticmethod
        @torch.jit.ignore
        def forward(ctx, *args, **kwargs):
            return extension.evaluate(None, *args, **kwargs)

    def wrapper(*args, **kwargs):
        return extension.convert(None, Trampoline.apply, *args, **kwargs)

    return wrapper


def _fp32_tensor(*shape, device=None):
    """Create a placeholder FP32 tensor with the given shape and device."""
    return torch.full(shape, 0.5, dtype=torch.float32, device=device)


# Extension for torch.bmm: (b, n, m) @ (b, m, p) -> (b, n, p)
_bmm_extension = ModuleExtension(
    None, "ov_ext::bmm",
    convert=lambda module, target_op, *args, **kwargs: target_op(*args),
    evaluate=lambda module, *args, **kwargs: _fp32_tensor(
        args[0].shape[0], args[0].shape[1], args[1].shape[2], device=args[0].device)
)


# Thread-safe, reference-counted storage for patched torch functions.
# Each entry: key -> (orig_fn, ref_count)
_patched_torch_functions = {}
_patch_lock = threading.Lock()


def _patch_torch_functions():
    """Patch torch functions that don't work well with 16-bit types (e.g., bmm for MoE models).

    These patches skip actual computation and create custom ops in the TorchScript graph,
    similar to how ModuleExtension works for modules. This speeds up tracing and avoids
    loading weights from mmap.

    Thread-safe and ref-counted: the wrapper is installed only on the first call and
    restored only when every matching _unpatch_torch_functions() call has been made,
    so concurrent or nested patching is safe.
    """
    functions_to_patch = [
        (torch, "bmm", _bmm_extension),
    ]

    with _patch_lock:
        for module, fn_name, extension in functions_to_patch:
            key = (module, fn_name)
            if key in _patched_torch_functions:
                orig_fn, ref_count = _patched_torch_functions[key]
                _patched_torch_functions[key] = (orig_fn, ref_count + 1)
                log.debug("Already patched torch function: %s.%s (ref_count=%d)",
                          module.__name__, fn_name, ref_count + 1)
            else:
                orig_fn = getattr(module, fn_name)
                _patched_torch_functions[key] = (orig_fn, 1)
                setattr(module, fn_name, _create_function_wrapper(extension))
                log.debug("Patched torch function: %s.%s", module.__name__, fn_name)


def _unpatch_torch_functions():
    """Restore original torch functions.

    Decrements the ref count; the original function is restored only when the
    count reaches zero, so nested/concurrent patch pairs work correctly.
    """
    with _patch_lock:
        to_remove = []
        for (module, fn_name), (orig_fn, ref_count) in _patched_torch_functions.items():
            new_count = ref_count - 1
            if new_count <= 0:
                setattr(module, fn_name, orig_fn)
                to_remove.append((module, fn_name))
                log.debug("Restored torch function: %s.%s", module.__name__, fn_name)
            else:
                _patched_torch_functions[(module, fn_name)] = (orig_fn, new_count)
                log.debug("Decremented ref count for torch function: %s.%s (ref_count=%d)",
                          module.__name__, fn_name, new_count)
        for key in to_remove:
            del _patched_torch_functions[key]


def __make_16bit_traceable(model: torch.nn.Module,
                           orig_forward_name: str = "_openvino_module_extension_patch_orig_forward",
                           patch_condition=None):
    """Prepare a 16-bit PyTorch model for tracing with OpenVINO.

    - Replace known list of modules with ModuleExtension.
    - Patch torch functions (bmm, baddbmm, etc.) for MoE and similar models.
    - Convert other modules with weights to FP32.
    """
    supported = {torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2}

    # Patch torch functions for operations like bmm used in MoE models
    _patch_torch_functions()

    if patch_condition is None:
        def patch_condition(module):
            dtype_to_patch = {torch.float32, *supported}
            weight = getattr(module, "weight", None)
            return weight is not None and weight.dtype in dtype_to_patch

    extensions = {
        torch.nn.Linear: ModuleExtension(
            torch.nn.Linear, "ov_ext::linear",
            convert=lambda module, target_op, *args, **kwargs: target_op(args[0],
                                                                         module.weight,
                                                                         module.bias),
            evaluate=lambda module, *args, **kwargs: _fp32_tensor(
                *args[0].shape[:-1], module.out_features, device=args[0].device),
            condition=patch_condition),
        torch.nn.Embedding: ModuleExtension(
            torch.nn.Embedding, "ov_ext::embedding",
            convert=lambda module, target_op, *args, **kwargs: target_op(module.weight,
                                                                         args[0],
                                                                         module.padding_idx,
                                                                         module.scale_grad_by_freq,
                                                                         module.sparse),
            evaluate=lambda module, *args, **kwargs: _fp32_tensor(
                *args[1].shape, module.embedding_dim, device=args[1].device),
            condition=patch_condition),
    }
    try:
        from transformers.pytorch_utils import Conv1D
        extensions[Conv1D] = ModuleExtension(
            Conv1D, "ov_ext::conv1d",
            convert=lambda module, target_op, *args, **kwargs: target_op(args[0],
                                                                         module.weight,
                                                                         module.bias),
            evaluate=lambda module, *args, **kwargs: _fp32_tensor(
                *args[0].shape[:-1], module.nf, device=args[0].device),
            condition=patch_condition)
    except ImportError:
        pass
    patch_model(model, extensions, orig_forward_name)
    for _, module in model.named_modules():
        if (module.__class__ not in extensions
            and (any(p.dtype in supported for p in module.parameters(False))
                 or any(b.dtype in supported for b in module.buffers(False)))):
            log.debug("Casting module %s to float32", module)
            module.float()
