# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import openvino as ov
from snippets import get_model


def main():
    model = get_model()

    core = ov.Core()
    if "GPU" not in core.available_devices:
        return 0

    #! [compile_model_default_gpu]
    core = ov.Core()
    compiled_model = core.compile_model(model, "GPU")
    #! [compile_model_default_gpu]

    #! [compile_model_gpu_with_id]
    core = ov.Core()
    compiled_model = core.compile_model(model, "GPU.1")
    #! [compile_model_gpu_with_id]

    #! [compile_model_gpu_with_id_and_tile]
    core = ov.Core()
    compiled_model = core.compile_model(model, "GPU.1.0")
    #! [compile_model_gpu_with_id_and_tile]

    #! [compile_model_multi]
    core = ov.Core()
    compiled_model = core.compile_model(model, "MULTI:GPU.1,GPU.0")
    #! [compile_model_multi]

    #! [compile_model_auto]
    core = ov.Core()
    compiled_model = core.compile_model(model, "AUTO:GPU.1,CPU.0", {hints.performance_mode: hints.PerformanceMode.CUMULATIVE_THROUGHPUT})
    #! [compile_model_auto]

    #! [compile_model_batch_plugin]
    core = ov.Core()
    compiled_model = core.compile_model(model, "BATCH:GPU")
    #! [compile_model_batch_plugin]

    #! [compile_model_auto_batch]
    import openvino.properties.hint as hints

    core = ov.Core()
    compiled_model = core.compile_model(
        model,
        "GPU",
        {
            hints.performance_mode: hints.PerformanceMode.THROUGHPUT,
        },
    )
    #! [compile_model_auto_batch]
