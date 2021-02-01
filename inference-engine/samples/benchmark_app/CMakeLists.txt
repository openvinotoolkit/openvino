# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file (GLOB SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file (GLOB HDR ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

ie_add_sample(NAME benchmark_app
              SOURCES ${SRC}
              HEADERS ${HDR}
              DEPENDENCIES format_reader
              OPENCV_DEPENDENCIES imgcodecs)

target_include_directories(benchmark_app SYSTEM PRIVATE
    ${IE_MAIN_SOURCE_DIR}/src/vpu/common/include
    ${IE_MAIN_SOURCE_DIR}/src/plugin_api
)
