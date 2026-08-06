# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_tp_gpu_unit_tests)

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# The plugin is built as a MODULE and exports no symbols, so the sources under
# test are compiled into the test binary.  The collective ops already live in a
# static library, which is linked instead.
list(APPEND SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../../src/graph_rewriter.cpp)

# The coordinator tests talk to Level Zero and OpenCL directly; without them
# they cannot be built, and the rest of the suite still can.
find_package(OpenCL)
set(ENABLE_COORDINATOR_TESTS OFF)
if(OpenCL_FOUND AND TARGET LevelZero::LevelZero)
    set(ENABLE_COORDINATOR_TESTS ON)
    list(APPEND SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../../src/tp_device_coordinator.cpp)
else()
    message(STATUS "${TARGET_NAME}: Level Zero or OpenCL not available; coordinator tests disabled")
    list(REMOVE_ITEM SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/tp_device_coordinator_test.cpp)
endif()

add_executable(${TARGET_NAME} ${SOURCES})

target_include_directories(${TARGET_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src)

target_link_libraries(${TARGET_NAME} PRIVATE
    openvino::tp_gpu_ops
    openvino::runtime
    openvino::runtime::dev
    common_test_utils
    gtest
    gtest_main)

if(ENABLE_COORDINATOR_TESTS)
    target_include_directories(${TARGET_NAME} PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}/../../generated)

    # Link the L0 loader directly (as DT_NEEDED) so it is loaded globally at
    # process startup. Without this, openvino::zero_loader's dlopen uses
    # RTLD_LOCAL and the L0 GPU driver fails to resolve its loader callbacks.
    target_link_libraries(${TARGET_NAME} PRIVATE
        openvino::zero_loader
        LevelZero::LevelZero
        OpenCL::OpenCL
        Threads::Threads)
endif()

set_target_properties(${TARGET_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${OpenVINO_SOURCE_DIR}/bin/intel64/$<CONFIG>")

add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})
set_tests_properties(${TARGET_NAME} PROPERTIES LABELS "TP_GPU")

ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})
