# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (TARGET_NAME "timetests_helper")

file (GLOB SRC *.cpp)
add_library(${TARGET_NAME} STATIC ${SRC})
target_include_directories(${TARGET_NAME} PUBLIC "${CMAKE_SOURCE_DIR}/include")

add_subdirectory(${OpenVINO_SOURCE_DIR}/thirdparty/gflags
                 ${CMAKE_CURRENT_BINARY_DIR}/gflags_build
                 EXCLUDE_FROM_ALL)

target_link_libraries(${TARGET_NAME} gflags tests_shared_lib)
