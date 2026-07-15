# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME tp_test)

add_executable(${TARGET_NAME} tp_test.cpp)

target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime)

set_target_properties(${TARGET_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${OpenVINO_SOURCE_DIR}/bin/intel64/$<CONFIG>")

# ---------------------------------------------------------------------------
# tp_coordinator_test: standalone unit-test for TPDeviceCoordinator.
#
# The plugin is built as a MODULE (no exported symbols), so we recompile the
# coordinator source directly into the test executable.
# ---------------------------------------------------------------------------
find_package(OpenCL REQUIRED)
find_library(LEVEL_ZERO_LOADER_LIB ze_loader)
find_path(LEVEL_ZERO_INCLUDE_DIR_TEST level_zero/ze_api.h)

if(NOT LEVEL_ZERO_LOADER_LIB OR NOT LEVEL_ZERO_INCLUDE_DIR_TEST)
    message(STATUS "tp_coordinator_test: Level Zero loader/headers not found; test disabled")
    return()
endif()

set(COORD_TARGET tp_coordinator_test)
add_executable(${COORD_TARGET}
    tp_coordinator_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/tp_device_coordinator.cpp
)

target_include_directories(${COORD_TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${CMAKE_CURRENT_BINARY_DIR}/../generated
    ${LEVEL_ZERO_INCLUDE_DIR_TEST}
)

# Link the system libze_loader directly (as DT_NEEDED) so ze_loader is loaded
# globally at process startup. Without this, openvino::zero_loader's dlopen
# uses RTLD_LOCAL and the L0 GPU driver fails to resolve its loader callbacks.
target_link_libraries(${COORD_TARGET} PRIVATE
    openvino::runtime
    openvino::zero_loader
    ${LEVEL_ZERO_LOADER_LIB}
    OpenCL::OpenCL
    Threads::Threads
)

set_target_properties(${COORD_TARGET} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${OpenVINO_SOURCE_DIR}/bin/intel64/$<CONFIG>")
