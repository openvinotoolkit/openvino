# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import openvino as ov
from utils import get_model


def main():

    core = ov.Core()
    model = get_model()

    if "GPU" not in core.available_devices:
        return 0

    # [compile_model]
    import openvino.properties as props
    import openvino.properties.hint as hints

    config = {hints.performance_mode: hints.PerformanceMode.THROUGHPUT}
    compiled_model = core.compile_model(model, "GPU", config)
    # [compile_model]

    # [compile_model_no_auto_batching]
    # disabling the automatic batching
    # leaving intact other configurations options that the device selects for the 'throughput' hint 
    config = {hints.performance_mode: hints.PerformanceMode.THROUGHPUT,
              hints.allow_auto_batching: False}
    compiled_model = core.compile_model(model, "GPU", config)
    # [compile_model_no_auto_batching]

    # [query_optimal_num_requests]
    # when the batch size is automatically selected by the implementation
    # it is important to query/create and run the sufficient requests
    config = {hints.performance_mode: hints.PerformanceMode.THROUGHPUT}
    compiled_model = core.compile_model(model, "GPU", config)
    num_requests = compiled_model.get_property(props.optimal_number_of_infer_requests)
    # [query_optimal_num_requests]

    # [hint_num_requests]
    config = {hints.performance_mode: hints.PerformanceMode.THROUGHPUT,
              hints.num_requests: "4"}
    # limiting the available parallel slack for the 'throughput'
    # so that certain parameters (like selected batch size) are automatically accommodated accordingly 
    compiled_model = core.compile_model(model, "GPU", config)
    # [hint_num_requests]

    # [hint_plus_low_level]
    config = {hints.performance_mode: hints.PerformanceMode.THROUGHPUT,
              props.inference_num_threads: "4"}
    # limiting the available parallel slack for the 'throughput'
    # so that certain parameters (like selected batch size) are automatically accommodated accordingly
    compiled_model = core.compile_model(model, "CPU", config)
    # [hint_plus_low_level]]
