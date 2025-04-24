# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_appverifier_tests)

file (GLOB_RECURSE SRC *.cpp)
file (GLOB_RECURSE HDR *.h)

# Create library file from sources.
add_executable(${TARGET_NAME} ${HDR} ${SRC})

target_link_libraries(${TARGET_NAME}
    PUBLIC
        gtest
        pugixml
        gflags
    PRIVATE
        gtest_main)

ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})
