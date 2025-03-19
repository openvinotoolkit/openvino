# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino as ov
from snippets import get_model


def main():
    model = get_model()

    core = ov.Core()
    if "NPU" not in core.available_devices:
        return 0

    #! [compile_model_default_npu]
    core = ov.Core()
    compiled_model = core.compile_model(model, "NPU")
    #! [compile_model_default_npu]
