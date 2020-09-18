# Copyright (C) 2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (TARGET_NAME "time-testhelper")

find_package(gflags REQUIRED)

file (GLOB SRC *.cpp)
add_library(${TARGET_NAME} STATIC ${SRC})
target_include_directories(${TARGET_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_link_libraries(${TARGET_NAME} gflags)
