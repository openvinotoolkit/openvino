# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# OpenVINO C API 2.0 test sample
set(TARGET_NAME "ov_capi_test")

set(TEST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/ov_auto_property_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ov_compiled_model_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ov_core_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ov_infer_request_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ov_layout_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ov_model_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ov_partial_shape_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ov_preprocess_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ov_tensor_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ov_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ov_util_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ov_windows_conflict_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_model_repo.cpp
)
set(TEST_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/ov_test.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_model_repo.hpp
)
list(APPEND TEST_DEFINES TEST_CUSTOM_OP_CONFIG_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../../../plugins/intel_gpu/tests/functional/custom_op/custom_op.xml")

set(TEST_CHECK_SOURCES_EXCLUDE_FILES)

if(TARGET OpenCL::OpenCL)
    list(APPEND TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/ov_remote_context_test.cpp)
else()
    list(APPEND TEST_CHECK_SOURCES_EXCLUDE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/ov_remote_context_test.cpp)
endif()

set(TEST_LINK_LIBRARIES openvino_c openvino::util common_test_utils gtest_main)

if(TARGET OpenCL::OpenCL)
    list(APPEND TEST_LINK_LIBRARIES OpenCL::OpenCL)
endif()

ov_add_test_target(
    NAME
        ${TARGET_NAME}
    SOURCES
        ${TEST_SOURCES} ${TEST_HEADERS}
    DEFINES
        ${TEST_DEFINES}
    INCLUDES
        $<BUILD_INTERFACE:${OPENVINO_API_SOURCE_DIR}/include>
    LINK_LIBRARIES
        ${TEST_LINK_LIBRARIES}
    DEPENDENCIES
        mock_engine
    ADD_CLANG_FORMAT
    LABELS
        OV UNIT C_API
    CHECK_SOURCES_LISTED
    CHECK_SOURCES_EXCLUDE_FILES
        ${TEST_CHECK_SOURCES_EXCLUDE_FILES}
)

ov_build_target_faster(${TARGET_NAME} PCH PCH_EXCLUDE "ov_windows_conflict_test.cpp")
