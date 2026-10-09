# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_ir_frontend_tests)

ov_add_test_target(
        NAME ${TARGET_NAME}
        SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/frontend_test.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/frontend_test_basic.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/frontend_test_mmap.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/frontend_test_with_extensions.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/meta_data_tests.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/partial_shape_deserialization.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/pre_processing_deserialization.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/rt_info_custom.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/rt_info_deserialization.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/tensor_iterator_deserialization.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/threading_tests.cpp
        DEPENDENCIES
            openvino_ir_frontend
        LINK_LIBRARIES
            gtest
            gtest_main
            openvino::runtime::dev
            common_test_utils
            frontend_shared_test_classes
        INCLUDES
            "${CMAKE_CURRENT_SOURCE_DIR}/../include"
        ADD_CLANG_FORMAT
        CHECK_SOURCES_LISTED
        LABELS
            OV UNIT IR_FE
)

ov_build_target_faster(${TARGET_NAME} PCH)
