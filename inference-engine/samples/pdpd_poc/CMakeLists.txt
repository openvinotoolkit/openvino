# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file (GLOB SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file (GLOB HDR ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

find_package(Protobuf REQUIRED IMPORTED)
include_directories(${Protobuf_INCLUDE_DIRS})
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS framework.proto)

ie_add_sample(NAME pdpd_poc
              SOURCES ${SRC} ${PROTO_SRCS}
              HEADERS ${HDR} ${PROTO_HDRS}
              DEPENDENCIES format_reader ngraph libprotobuf libprotoc
              OPENCV_DEPENDENCIES imgcodecs
              EXCLUDE_CPPLINT)
