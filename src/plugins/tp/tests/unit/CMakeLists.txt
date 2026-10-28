# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_tp_unit_tests)

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# The plugin is built as a MODULE and exports no symbols, so the sources under
# test are compiled into the test binary. The collectives are tested with the
# GPU plugin that runs them (ov_gpu_unit_tests, TPDeviceCoordinatorTest).
list(APPEND SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../../src/graph_rewriter.cpp)

add_executable(${TARGET_NAME} ${SOURCES})

target_include_directories(${TARGET_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src)

if(ENABLE_TP_DEBUG_CAPS)
    target_compile_definitions(${TARGET_NAME} PRIVATE TP_DEBUG_CONFIG=1)
endif()

target_link_libraries(${TARGET_NAME} PRIVATE
    openvino::runtime::dev
    common_test_utils
    gtest_main)

set_target_properties(${TARGET_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${OpenVINO_SOURCE_DIR}/bin/intel64/$<CONFIG>")

add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})
set_tests_properties(${TARGET_NAME} PROPERTIES LABELS "TP")

ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})
