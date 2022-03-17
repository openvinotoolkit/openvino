# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "mock_abc_plugin")

file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB_RECURSE HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

# adds a shared library with plugin
ov_add_plugin(NAME ${TARGET_NAME}
              DEVICE_NAME "ABC"
              SOURCES ${SOURCES} ${HEADERS}
              SKIP_INSTALL
              ADD_CLANG_FORMAT)
