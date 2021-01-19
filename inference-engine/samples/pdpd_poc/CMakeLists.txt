# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file (GLOB SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file (GLOB HDR ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

ie_add_sample(NAME pdpd_poc
              SOURCES ${SRC}
              HEADERS ${HDR}
              DEPENDENCIES format_reader ngraph
              OPENCV_DEPENDENCIES imgcodecs)
