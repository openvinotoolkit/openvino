# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.13)

project(libprotobuf_static)

message("Add PROTOBUF dependency - static")

set(BUILD_SHARED_LIBS OFF)
set(BUILD_STANDALONE_STATIC ON)
set(USE_STATIC_PROTOBUF ON)

add_subdirectory(../protobuf ${CMAKE_BINARY_DIR}/_deps/static-protobuf)
