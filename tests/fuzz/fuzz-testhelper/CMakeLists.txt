# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME fuzz-testhelper)

file(
    GLOB SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

add_library(
    ${TARGET_NAME} STATIC
    ${SRC_FILES})

target_include_directories(${TARGET_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

if(ENABLE_FUZZING)
    target_compile_definitions(${TARGET_NAME} PRIVATE WITH_LIBFUZZER)
endif()
