# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(OV_AUTO_BATCH_UNIT_TESTS_SRC
    ${CMAKE_CURRENT_LIST_DIR}/async_infer_request_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/compile_model_create_infer_request_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/compile_model_get_property_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/compile_model_get_runtime_model_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/compile_model_set_property_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mock_common.hpp
    ${CMAKE_CURRENT_LIST_DIR}/parse_batch_device_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/parse_meta_device_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/plugin_compile_model_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/plugin_get_property_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/plugin_query_model_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/plugin_set_property_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/sync_infer_request_test.cpp
    ${OpenVINO_SOURCE_DIR}/src/plugins/auto_batch/src/async_infer_request.cpp
    ${OpenVINO_SOURCE_DIR}/src/plugins/auto_batch/src/async_infer_request.hpp
    ${OpenVINO_SOURCE_DIR}/src/plugins/auto_batch/src/compiled_model.cpp
    ${OpenVINO_SOURCE_DIR}/src/plugins/auto_batch/src/compiled_model.hpp
    ${OpenVINO_SOURCE_DIR}/src/plugins/auto_batch/src/plugin.cpp
    ${OpenVINO_SOURCE_DIR}/src/plugins/auto_batch/src/plugin.hpp
    ${OpenVINO_SOURCE_DIR}/src/plugins/auto_batch/src/sync_infer_request.cpp
    ${OpenVINO_SOURCE_DIR}/src/plugins/auto_batch/src/sync_infer_request.hpp
)
