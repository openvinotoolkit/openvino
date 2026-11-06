# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(OV_API_CONFORMANCE_TEST_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/src/core_config.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/ov_compiled_model/exec_network_base.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/ov_compiled_model/get_metric.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/ov_compiled_model/import_export.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/ov_compiled_model/properties.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/ov_infer_request/callback.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/ov_infer_request/cancellation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/ov_infer_request/infer_request_dynamic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/ov_infer_request/inference_chaining.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/ov_infer_request/io_tensor.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/ov_infer_request/memory_state.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/ov_infer_request/multithreading.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/ov_infer_request/perf_counters.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/ov_infer_request/wait.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/ov_plugin/caching_tests.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/ov_plugin/core_integration.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/ov_plugin/core_threading_tests.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/ov_plugin/life_time.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/ov_plugin/properties.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/ov_plugin/version.cpp
)
