# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file (GLOB SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file (GLOB HDR ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

ie_add_sample(NAME benchmark_app
              SOURCES ${SRC}
              HEADERS ${HDR}
              DEPENDENCIES format_reader ie_samples_utils
              OPENCV_DEPENDENCIES core)
