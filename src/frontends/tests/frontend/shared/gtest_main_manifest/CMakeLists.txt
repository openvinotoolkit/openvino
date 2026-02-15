# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "gtest_main_manifest")

# Make this library as interface (source) to build it for each test targets which can define
# ${MANIFEST} file path define.
add_library(${TARGET_NAME} INTERFACE)
target_sources(${TARGET_NAME} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_include_directories(${TARGET_NAME} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(${TARGET_NAME} INTERFACE frontend_shared_test_classes)
