# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file (GLOB SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
file (GLOB HDR ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
               ${CMAKE_CURRENT_SOURCE_DIR}/*.h)

ie_add_sample(NAME speech_sample
              SOURCES ${SRC}
              HEADERS ${HDR}
              DEPENDENCIES cnpy ie_samples_utils)
