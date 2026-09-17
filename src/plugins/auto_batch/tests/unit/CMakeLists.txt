# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_auto_batch_unit_tests)

set(CI_BUILD_NUMBER "unittest")
add_definitions(-DAUTOBATCH_UNITTEST)

set(SHARED_HEADERS_DIR "${OpenVINO_SOURCE_DIR}/src/tests/ie_test_util")

set(OV_AUTO_BATCH_UNIT_TESTS_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/async_infer_request_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compile_model_create_infer_request_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compile_model_get_property_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compile_model_get_runtime_model_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compile_model_set_property_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mock_common.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/parse_batch_device_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/parse_meta_device_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plugin_compile_model_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plugin_get_property_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plugin_query_model_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plugin_set_property_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sync_infer_request_test.cpp
    ${OpenVINO_SOURCE_DIR}/src/plugins/auto_batch/src/async_infer_request.cpp
    ${OpenVINO_SOURCE_DIR}/src/plugins/auto_batch/src/async_infer_request.hpp
    ${OpenVINO_SOURCE_DIR}/src/plugins/auto_batch/src/compiled_model.cpp
    ${OpenVINO_SOURCE_DIR}/src/plugins/auto_batch/src/compiled_model.hpp
    ${OpenVINO_SOURCE_DIR}/src/plugins/auto_batch/src/plugin.cpp
    ${OpenVINO_SOURCE_DIR}/src/plugins/auto_batch/src/plugin.hpp
    ${OpenVINO_SOURCE_DIR}/src/plugins/auto_batch/src/sync_infer_request.cpp
    ${OpenVINO_SOURCE_DIR}/src/plugins/auto_batch/src/sync_infer_request.hpp
)

ov_add_test_target(
        NAME
            ${TARGET_NAME}
        SOURCES
            ${OV_AUTO_BATCH_UNIT_TESTS_SRC}
        INCLUDES
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${OpenVINO_SOURCE_DIR}/src/plugins/auto_batch/src
            ${SHARED_HEADERS_DIR}
        LINK_LIBRARIES
            unit_test_utils
        DEPENDENCIES
            mock_engine
        CHECK_SOURCES_LISTED
        LABELS
            OV UNIT AUTO_BATCH
)

ov_add_version_defines(${OpenVINO_SOURCE_DIR}/src/plugins/auto_batch/src/plugin.cpp ${TARGET_NAME})

ov_set_threading_interface_for(${TARGET_NAME})
