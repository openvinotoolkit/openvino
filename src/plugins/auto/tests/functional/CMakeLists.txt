# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_auto_func_tests)

if(ENABLE_AUTO_BATCH)
    list(APPEND DEPENDENCIES openvino_auto_batch_plugin)
    list(APPEND COMPILE_DEFINITIONS ENABLE_AUTO_BATCH)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    ov_add_compiler_flags(/wd4305)
endif()

include(sources.cmake)

ov_add_test_target(
        NAME ${TARGET_NAME}
        SOURCES
            ${OV_AUTO_FUNC_TESTS_SRC}
        LINK_LIBRARIES
            openvino::runtime::dev
            gtest
            gtest_main
            funcSharedTests
        INCLUDES
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${TEST_COMMON_INCLUDE_DIR}
        ADD_CLANG_FORMAT
        CHECK_SOURCES_LISTED
        LABELS
            Multi
            Auto
)

target_compile_definitions(${TARGET_NAME} PRIVATE ${COMPILE_DEFINITIONS})
ov_set_threading_interface_for(${TARGET_NAME})
