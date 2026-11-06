# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_tp_func_tests)

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_executable(${TARGET_NAME} ${SOURCES})

target_include_directories(${TARGET_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include)

target_link_libraries(${TARGET_NAME} PRIVATE
    openvino::runtime
    openvino::runtime::dev  # the TP plugin's internal properties
    common_test_utils
    gtest
    gtest_main)

if(ENABLE_TP_DEBUG_CAPS)
    target_compile_definitions(${TARGET_NAME} PRIVATE ENABLE_TP_DEBUG_CAPS)
endif()

# The plugins are loaded at run time, but must exist before the test runs. The
# rank devices are of type OV_TP_TEST_DEVICE (GPU by default).
add_dependencies(${TARGET_NAME} openvino_tp_plugin)
if(ENABLE_INTEL_GPU)
    add_dependencies(${TARGET_NAME} openvino_intel_gpu_plugin)
endif()

set_target_properties(${TARGET_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${OpenVINO_SOURCE_DIR}/bin/intel64/$<CONFIG>")

add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})
set_tests_properties(${TARGET_NAME} PROPERTIES LABELS "TP")
