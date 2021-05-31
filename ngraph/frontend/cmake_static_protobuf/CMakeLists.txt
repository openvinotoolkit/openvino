# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.13)

project(libprotobuf_static)

# DEBUG - print all defined variables
# get_cmake_property(_variableNames VARIABLES)
# set(ALL_VARS "")
# foreach (_variableName ${_variableNames})
#     set(ALL_VARS ${ALL_VARS} -D${_variableName}=${${_variableName}}\n )
# endforeach()
# message("---------------------${ALL_VARS}-------")

message("Add PROTOBUF dependency - static")

set(BUILD_SHARED_LIBS OFF)
set(BUILD_STANDALONE_STATIC ON)
set(USE_STATIC_PROTOBUF ON)

include(../../cmake/external_protobuf.cmake)
