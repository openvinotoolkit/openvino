# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_auto_func_tests)

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    ie_add_compiler_flags(/wd4305)
endif()

ov_add_test_target(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDENCIES
            openvino_auto_plugin
            openvino_intel_cpu_plugin
            openvino_intel_gpu_plugin
        LINK_LIBRARIES
            openvino::funcSharedTests
            openvino::runtime::dev
        INCLUDES
            ${CMAKE_CURRENT_SOURCE_DIR}
            $<TARGET_PROPERTY:openvino_auto_plugin,SOURCE_DIR>/include/
            ${TEST_COMMON_INCLUDE_DIR}
        # ADD_CPPLINT
        LABELS
            Multi
            Auto
)

set_ie_threading_interface_for(${TARGET_NAME})