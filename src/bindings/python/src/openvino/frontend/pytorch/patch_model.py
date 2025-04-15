# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# mypy: ignore-errors

import functools
import logging
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
    for _, module in model.named_modules():
        if hasattr(module, orig_forward_name):
            try:
                module.forward = getattr(module, orig_forward_name)
                delattr(module, orig_forward_name)
            except Exception as error:
                log.warning("Exception raised during model unpatching. "
                            "Depending on the exact issue it may lead to broken original model.\n"
                            "Original exception details:\n%s", error)


def __make_16bit_traceable(model: torch.nn.Module,
                           orig_forward_name: str = "_openvino_module_extension_patch_orig_forward",
                           patch_condition=None):
    """Prepare a 16-bit PyTorch model for tracing with OpenVINO.

    - Replace known list of modules with ModuleExtension.
    - Convert other modules with weights to FP32.
    """
    supported = {torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2}
    if patch_condition is None:
        def patch_condition(module):
            dtype_to_patch = {torch.float32, *supported}
            weight = getattr(module, "weight", None)
            return weight is not None and weight.dtype in dtype_to_patch

    def fp32_tensor(*shape):
        return torch.full(shape, 0.5, dtype=torch.float32)

    extensions = {
        torch.nn.Linear: ModuleExtension(
            torch.nn.Linear, "ov_ext::linear",
            convert=lambda module, target_op, *args, **kwargs: target_op(args[0],
                                                                         module.weight,
                                                                         module.bias),
            evaluate=lambda module, *args, **kwargs: fp32_tensor(*args[0].shape[:-1], module.out_features),
            condition=patch_condition),
        torch.nn.Embedding: ModuleExtension(
            torch.nn.Embedding, "ov_ext::embedding",
            convert=lambda module, target_op, *args, **kwargs: target_op(module.weight,
                                                                         args[0],
                                                                         module.padding_idx,
                                                                         module.scale_grad_by_freq,
                                                                         module.sparse),
            evaluate=lambda module, *args, **kwargs: fp32_tensor(*args[1].shape, module.embedding_dim),
            condition=patch_condition),
    }
    try:
        from transformers.pytorch_utils import Conv1D
        extensions[Conv1D] = ModuleExtension(
            Conv1D, "ov_ext::conv1d",
            convert=lambda module, target_op, *args, **kwargs: target_op(args[0],
                                                                         module.weight,
                                                                         module.bias),
            evaluate=lambda module, *args, **kwargs: fp32_tensor(*args[0].shape[:-1], module.nf),
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
