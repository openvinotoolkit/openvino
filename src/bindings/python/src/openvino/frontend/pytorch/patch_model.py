# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa
# mypy: ignore-errors

import torch
from openvino.frontend.pytorch import ModuleExtension


class no_jit_trace:
    def __enter__(self):
        self.state = torch._C._get_tracing_state()
        torch._C._set_tracing_state(None)

    def __exit__(self, *args):
        torch._C._set_tracing_state(self.state)
        self.state = None


def patch_model(model, module_extensions, orig_forward_name):
    def module_patcher(m, name):
        extension = None
        if m in module_extensions:
            extension = module_extensions[m]
        elif m.__class__ in module_extensions:
            extension = module_extensions[m.__class__]
        elif name in module_extensions:
            extension = module_extensions[name]

        if extension:
            # The Trampoline class is instantiated for every module replacement, so we can use class members individually for each module.
            class Trampoline(torch.autograd.Function):
                target_extension = extension
                original_module = m
                stashed_args = None
                stashed_kwargs = None

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
                        results = extension.evaluate(
                            m, *Trampoline.stashed_args, **Trampoline.stashed_kwargs)  # call user code
                        m.forward = patched_forward  # return patched forward back
                        return results

            def new_forward(*args, **kwargs):
                Trampoline.stashed_args = args
                Trampoline.stashed_kwargs = kwargs
                return extension.convert(m, Trampoline.apply, *args, **kwargs)
            setattr(m, orig_forward_name, m.forward)
            m.forward = new_forward

    for name, m in model.named_modules():
        if hasattr(m, orig_forward_name):
            # already patched, skipping with a warning because it is unexpected
            print(f'[ WARNING ] Unexpectedly found already patched module {name} while applying ModuleExtension during PyTorch model conversion. '
                  'Result of the conversion maybe broken. Depending on the exact issue it may lead to broken original model.')
            continue
        module_patcher(m, name)


def unpatch_model(model, orig_forward_name):
    for _, m in model.named_modules():
        if hasattr(m, orig_forward_name):
            try:
                m.forward = getattr(m, orig_forward_name)
                delattr(m, orig_forward_name)
            except Exception as error:
                print('[ WARNING ] Exception raised during model unpatching. Depending on the exact issue it may lead to broken original model.')
                print('Original exception details:')
                print(error)


def __make_16bit_traceable(model: torch.nn.Module):
    # Replace torch.nn.Linear with ModuleExtension and move other modules to fp32
    extensions = {torch.nn.Linear: ModuleExtension(
        torch.nn.Linear,
        "aten::linear",
        evaluate=lambda module, *args, **kwargs: torch.ones(
            list(args[0].shape[:-1]) + [module.out_features], dtype=torch.float32) * 0.5,
        convert=lambda module, target_op, *args, **kwargs: target_op(args[0], module.weight, module.bias))
    }
    patch_model(model, extensions,
                "_openvino_module_extension_patch_orig_forward")
    for _, module in model.named_modules():
        if module.__class__ not in extensions and hasattr(module, "weight") and module.weight.dtype in [torch.float16, torch.bfloat16]:
            module.float()
