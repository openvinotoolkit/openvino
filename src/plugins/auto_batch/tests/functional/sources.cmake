# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(OV_AUTO_BATCH_FUNC_TESTS_SRC
    ${CMAKE_CURRENT_LIST_DIR}/behavior/ov_executable_network/exec_net_base.cpp
    ${CMAKE_CURRENT_LIST_DIR}/behavior/ov_executable_network/properties.cpp
    ${CMAKE_CURRENT_LIST_DIR}/behavior/ov_infer_request/callback.cpp
    ${CMAKE_CURRENT_LIST_DIR}/behavior/ov_infer_request/cancellation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/behavior/ov_infer_request/io_tensor.cpp
    ${CMAKE_CURRENT_LIST_DIR}/behavior/ov_infer_request/multithreading.cpp
    ${CMAKE_CURRENT_LIST_DIR}/behavior/ov_infer_request/perf_counters.cpp
    ${CMAKE_CURRENT_LIST_DIR}/behavior/ov_infer_request/wait.cpp
    ${CMAKE_CURRENT_LIST_DIR}/behavior/ov_plugin/auto_batching_tests.cpp
    ${CMAKE_CURRENT_LIST_DIR}/behavior/ov_plugin/life_time.cpp
    ${CMAKE_CURRENT_LIST_DIR}/behavior/ov_plugin/properties_tests.cpp
    ${CMAKE_CURRENT_LIST_DIR}/behavior/ov_plugin/remote.cpp
    ${CMAKE_CURRENT_LIST_DIR}/set_device_name.cpp
    ${CMAKE_CURRENT_LIST_DIR}/skip_tests_config.cpp
)
