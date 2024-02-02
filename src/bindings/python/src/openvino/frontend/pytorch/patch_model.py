
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa
# mypy: ignore-errors


def patch_model(model, module_patcher):
    for name, m in model.named_modules():
        # TODO: Use one ID to mark nodes as patched, now GPTQ models use different ID
        if hasattr(m, '_openvino_patch_orig_forward_v2'):
            # already patched, skipping
            continue
        module_patcher(m, name)


def unpatch_model(model):
    for _, m in model.named_modules():
        if hasattr(m, '_openvino_patch_orig_forward_v2'):
            try:
                m.forward = m._openvino_patch_orig_forward_v2
                del m._openvino_patch_orig_forward_v2
            except Exception as error:
                print('[ WARNING ] Exception raised during model unpatching. Depending on the exact issue it may lead to broken original model')
                print(error)
