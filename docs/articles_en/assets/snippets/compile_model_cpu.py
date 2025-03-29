# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from snippets import get_model


def main():
    model = get_model()

    #! [compile_model_default]
    import openvino as ov

    core = ov.Core()
    compiled_model = core.compile_model(model, "CPU")
    #! [compile_model_default]

    if "GPU" not in core.available_devices:
        return 0

    #! [compile_model_multi]
    core = ov.Core()
    compiled_model = core.compile_model(model, "MULTI:CPU,GPU.0")
    #! [compile_model_multi]

    #! [compile_model_auto]
    core = ov.Core()
    compiled_model = core.compile_model(model, "AUTO:CPU,GPU.0", {hints.performance_mode: hints.PerformanceMode.CUMULATIVE_THROUGHPUT})
    #! [compile_model_auto]
