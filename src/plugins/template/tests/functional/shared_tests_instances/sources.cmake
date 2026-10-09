# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(OV_TEMPLATE_TESTS_SHARED_INSTANCES_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/behavior/ov_executable_network/exec_network_base.cpp
    ${CMAKE_CURRENT_LIST_DIR}/behavior/ov_executable_network/get_metric.cpp
    ${CMAKE_CURRENT_LIST_DIR}/behavior/ov_executable_network/ov_exec_net_import_export.cpp
    ${CMAKE_CURRENT_LIST_DIR}/behavior/ov_executable_network/properties.cpp
    ${CMAKE_CURRENT_LIST_DIR}/behavior/ov_infer_request/batched_tensors.cpp
    ${CMAKE_CURRENT_LIST_DIR}/behavior/ov_infer_request/callback.cpp
    ${CMAKE_CURRENT_LIST_DIR}/behavior/ov_infer_request/cancellation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/behavior/ov_infer_request/infer_request_dynamic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/behavior/ov_infer_request/inference.cpp
    ${CMAKE_CURRENT_LIST_DIR}/behavior/ov_infer_request/inference_chaining.cpp
    ${CMAKE_CURRENT_LIST_DIR}/behavior/ov_infer_request/io_tensor.cpp
    ${CMAKE_CURRENT_LIST_DIR}/behavior/ov_infer_request/memory_states.cpp
    ${CMAKE_CURRENT_LIST_DIR}/behavior/ov_infer_request/multithreading.cpp
    ${CMAKE_CURRENT_LIST_DIR}/behavior/ov_infer_request/properties_tests.cpp
    ${CMAKE_CURRENT_LIST_DIR}/behavior/ov_infer_request/wait.cpp
    ${CMAKE_CURRENT_LIST_DIR}/behavior/ov_plugin/caching_tests.cpp
    ${CMAKE_CURRENT_LIST_DIR}/behavior/ov_plugin/core_integration.cpp
    ${CMAKE_CURRENT_LIST_DIR}/behavior/ov_plugin/hetero_synthetic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/behavior/ov_plugin/life_time.cpp
    ${CMAKE_CURRENT_LIST_DIR}/behavior/ov_plugin/properties_tests.cpp
    ${CMAKE_CURRENT_LIST_DIR}/behavior/ov_plugin/remote.cpp
    ${CMAKE_CURRENT_LIST_DIR}/behavior/ov_plugin/version.cpp
    ${CMAKE_CURRENT_LIST_DIR}/single_layer_tests/convolution.cpp
    ${CMAKE_CURRENT_LIST_DIR}/single_layer_tests/eltwise.cpp
    ${CMAKE_CURRENT_LIST_DIR}/single_layer_tests/gather_nd.cpp
    ${CMAKE_CURRENT_LIST_DIR}/single_layer_tests/reshape.cpp
    ${CMAKE_CURRENT_LIST_DIR}/single_layer_tests/softmax.cpp
    ${CMAKE_CURRENT_LIST_DIR}/single_layer_tests/split.cpp
)
