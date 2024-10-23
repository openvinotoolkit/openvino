# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa
# mypy: ignore-errors

import logging
import torch
from openvino.frontend.pytorch import ModuleExtension

log = logging.getLogger(__name__)


class no_jit_trace:
    def __enter__(self):
        self.state = torch._C._get_tracing_state()
        torch._C._set_tracing_state(None)

    def __exit__(self, *args):
        torch._C._set_tracing_state(self.state)
        self.state = None


def patch_model(model, module_extensions, orig_forward_name, use_meta=False):
    def module_patcher(m, name):
        extension = None
        if m in module_extensions:
            extension = module_extensions[m]
        elif m.__class__ in module_extensions:
            extension = module_extensions[m.__class__]
        elif name in module_extensions:
            extension = module_extensions[name]

        if extension:
            log.debug("Patching module %s", m)
            # The Trampoline class is instantiated for every module replacement, so we can use class members individually for each module.

            class Trampoline(torch.autograd.Function):
                target_extension = extension
                original_module = m
                stashed_args = tuple()
                stashed_kwargs = {}

                @staticmethod
                @torch.jit.ignore
                def forward(*args, **kwargs):
                    with no_jit_trace():
                        # `module` is going to be passed to a user-defined function `evaluate`
                        # `module` is patched: forward function was replaced, and we are actually in this patched function right in this code
                        # if we pass `module` as-is to the user code below, and it happens to call forward it will lead to infinite recursion or fail
                        # so we need to temporary patch the module back to the original forward and then return it back again
                        # stash the current forward to be able to return it back
                        patched_forward = m.forward
                        # set original forward for the module
                        m.forward = getattr(m, orig_forward_name)
                        # call user code
                        results = extension.evaluate(m, *Trampoline.stashed_args,
                                                     **Trampoline.stashed_kwargs)
                        m.forward = patched_forward  # return patched forward back
                        return results

            def new_forward(*args, **kwargs):
                # use meta device to store args, to save memory
                if use_meta:
                    d = torch.device("meta")
                    Trampoline.stashed_args = tuple(a.to(d) for a in args)
                    Trampoline.stashed_kwargs = dict((k, v.to(d)) for k, v in kwargs.items())
                else:
                    Trampoline.stashed_args = args
                    Trampoline.stashed_kwargs = kwargs
                return extension.convert(m, Trampoline.apply, *args, **kwargs)

            setattr(m, orig_forward_name, m.forward)
            m.forward = new_forward

    for name, m in model.named_modules():
        if hasattr(m, orig_forward_name):
            # already patched, skipping. It may happen when patching applied for same module twice
            log.debug("Unexpectedly found already patched module %s while applying "
                      "ModuleExtension during PyTorch model conversion. "
                      "Result of the conversion maybe broken. Depending on the exact issue "
                      "it may lead to broken original model.", name)
            continue

        module_patcher(m, name)


def unpatch_model(model, orig_forward_name):
    for _, m in model.named_modules():
        if hasattr(m, orig_forward_name):
            try:
                m.forward = getattr(m, orig_forward_name)
                delattr(m, orig_forward_name)
            except Exception as error:
                log.warning("Exception raised during model unpatching. "
                            "Depending on the exact issue it may lead to broken original model.\n"
                            "Original exception details:\n%s", error)


def __make_16bit_traceable(model: torch.nn.Module):
    """
    Prepare a 16-bit PyTorch model for tracing with OpenVINO.
     - Replace known list of modules with ModuleExtension.
     - Convert other modules with weights to FP32.
    """
    extensions = {
        torch.nn.Linear: ModuleExtension(
            torch.nn.Linear, "ov_ext::linear",
            evaluate=lambda module, *args, **kwargs: torch.full(
                list(args[0].shape[:-1]) + [module.out_features], 0.5, dtype=torch.float32),
            convert=lambda module, target_op, *args, **kwargs: target_op(args[0],
                                                                         module.weight,
                                                                         module.bias)),
        torch.nn.Embedding: ModuleExtension(
            torch.nn.Embedding, "ov_ext::embedding",
            evaluate=lambda module, *args, **kwargs: torch.full(
                list(args[0].shape) + [module.embedding_dim], 0.5, dtype=torch.float32),
            convert=lambda module, target_op, *args, **kwargs: target_op(module.weight,
                                                                         args[0],
                                                                         module.padding_idx,
                                                                         module.scale_grad_by_freq,
                                                                         module.sparse)),
    }
    try:
        from transformers.pytorch_utils import Conv1D
        extensions[Conv1D] = ModuleExtension(
            Conv1D, "ov_ext::conv1d",
            evaluate=lambda module, *args, **kwargs: torch.full(
                list(args[0].shape[:-1]) + [module.nf], 0.5, dtype=torch.float32),
            convert=lambda module, target_op, *args, **kwargs: target_op(args[0],
                                                                         module.weight,
                                                                         module.bias))
    except:
        pass
    patch_model(model, extensions,
                "_openvino_module_extension_patch_orig_forward", use_meta=True)
    for _, module in model.named_modules():
        if module.__class__ not in extensions and (any(p.dtype in [torch.float16, torch.bfloat16] for p in module.parameters(False))
                                                   or any(b.dtype in [torch.float16, torch.bfloat16] for b in module.buffers(False))):
            log.debug("Casting module %s to float32", module)
            module.float()
