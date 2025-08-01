# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(ov_infer_request_SRC
    batched_tensors.cpp
    callback.cpp
    cancellation.cpp
    infer_consistency.cpp
    infer_request_dynamic.cpp
    inference.cpp
    inference_chaining.cpp
    io_tensor.cpp
    iteration_chaining.cpp
    memory_states.cpp
    multithreading.cpp
    perf_counters.cpp
    properties_tests.cpp
    wait.cpp
)

# Convert to full paths
foreach(src_file ${ov_infer_request_SRC})
    list(APPEND ov_infer_request_SRC_FULL "${CMAKE_CURRENT_SOURCE_DIR}/${src_file}")
endforeach()

set_property(GLOBAL APPEND PROPERTY CPU_FUNC_TESTS_SRC ${ov_infer_request_SRC_FULL})
