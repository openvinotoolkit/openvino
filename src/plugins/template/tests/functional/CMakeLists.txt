# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# [cmake:functional_tests]
set(TARGET_NAME ov_template_func_tests)

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    ov_add_compiler_flags(/wd4305)
endif()

set(OV_TEMPLATE_TESTS_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/core_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/set_device_name.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/skip_tests_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/specific_properties_test.cpp
)

include(op_reference/sources.cmake)
include(preprocessing/sources.cmake)
include(shared_tests_instances/sources.cmake)
include(subgraph_reference/sources.cmake)
include(transformations/sources.cmake)

ov_add_test_target(
        NAME ${TARGET_NAME}
        SOURCES
            ${OV_TEMPLATE_TESTS_SRCS}
            ${OV_TEMPLATE_TESTS_OP_REFERENCE_SRCS}
            ${OV_TEMPLATE_TESTS_PREPROCESSING_SRCS}
            ${OV_TEMPLATE_TESTS_SHARED_INSTANCES_SRCS}
            ${OV_TEMPLATE_TESTS_SUBGRAPH_REFERENCE_SRCS}
            ${OV_TEMPLATE_TESTS_TRANSFORMATIONS_SRCS}
        DEPENDENCIES
            openvino_template_plugin
        LINK_LIBRARIES
            openvino::funcSharedTests
            openvino::runtime::dev
        INCLUDES
            "${OpenVINOTemplatePlugin_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/op_reference"
        DEFINES
            IN_OV_COMPONENT
        ADD_CLANG_FORMAT
        LABELS
            OV UNIT TEMPLATE
        CHECK_SOURCES_LISTED
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
