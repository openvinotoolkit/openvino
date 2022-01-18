# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (TARGET_NAME "tests_shared_lib")

file (GLOB SRC *.cpp)
add_library(${TARGET_NAME} STATIC ${SRC})

target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Search OpenVINO Runtime installed
find_package(OpenVINO REQUIRED COMPONENTS Runtime)

target_link_libraries(${TARGET_NAME} PUBLIC openvino::runtime)
