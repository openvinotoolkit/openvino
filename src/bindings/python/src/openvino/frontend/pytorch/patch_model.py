
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa
# mypy: ignore-errors


def patch_model(model, module_patcher, orig_forward_name):
    for name, m in model.named_modules():
        # TODO: Use one way to identify a patched module, currently GPTQ model patching uses different name of attribute
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
