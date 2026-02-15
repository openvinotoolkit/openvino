# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (TARGET_NAME "tests_shared_lib")

file (GLOB SRC *.cpp)
add_library(${TARGET_NAME} STATIC ${SRC})

target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(${TARGET_NAME} PUBLIC ${ov_link_libraries})
