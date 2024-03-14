# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_thread_local_test)

file (GLOB_RECURSE SRC *.cpp)
file (GLOB_RECURSE HDR *.h)

set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/test")

# Create library file from sources.
add_executable(${TARGET_NAME} ${HDR} ${SRC})

target_link_libraries(${TARGET_NAME}
    PUBLIC
        gtest
        pugixml
        gflags
    PRIVATE
        gtest_main)
