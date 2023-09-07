# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file (GLOB MAIN ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file (GLOB SRCCPP ${CMAKE_CURRENT_SOURCE_DIR}/llama_cpp/*.cpp)
file (GLOB SRC ${CMAKE_CURRENT_SOURCE_DIR}/llama_cpp/*.c)
file (GLOB HDRHPP ${CMAKE_CURRENT_SOURCE_DIR}/llama_cpp/*.hpp)
file (GLOB HDR ${CMAKE_CURRENT_SOURCE_DIR}/llama_cpp/*.h)
              
ie_add_sample(NAME llama_sampple
              SOURCES ${SRC} ${SRCCPP} ${MAIN}
              HEADERS ${HDR} ${HDRHPP}
              DEPENDENCIES format_reader ie_samples_utils)