# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# [cmake:functional_tests]
set(TARGET_NAME ov_template_func_tests)

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    ov_add_compiler_flags(/wd4305)
endif()

ov_add_test_target(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDENCIES
            openvino_template_plugin
        LINK_LIBRARIES
            openvino::funcSharedTests
            openvino::runtime::dev
        INCLUDES
            "${OpenVINOTemplatePlugin_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/op_reference"
        ADD_CLANG_FORMAT
        LABELS
            OV UNIT TEMPLATE
)

find_package(OpenCV QUIET COMPONENTS core imgproc)

if(OpenCV_FOUND AND OpenCV_VERSION VERSION_GREATER_EQUAL 3.4)
    message(STATUS "Reference preprocessing: OpenCV tests are enabled")
    target_compile_definitions(${TARGET_NAME} PRIVATE OPENCV_TEMPLATE_TESTS)
    target_link_libraries(${TARGET_NAME} PRIVATE opencv_imgproc opencv_core)
else()
    message(WARNING "Reference preprocessing: OpenCV tests are disabled, because OpenCV ver. 3.4+ is not found")
endif()

if (ENABLE_INTEL_CPU)
    set_source_files_properties(
        "${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/behavior/executable_network/get_metric.cpp"
        PROPERTIES COMPILE_DEFINITIONS ENABLE_INTEL_CPU=1)
endif()
# [cmake:functional_tests]
