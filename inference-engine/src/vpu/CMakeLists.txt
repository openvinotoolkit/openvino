# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#
# Build common part
#

add_subdirectory(common)
add_subdirectory(graph_transformer)

add_subdirectory(
    "${IE_MAIN_SOURCE_DIR}/thirdparty/movidius"
    "${CMAKE_CURRENT_BINARY_DIR}/thirdparty/movidius")

#
# Build plugins
#

if(ENABLE_MYRIAD)
    add_subdirectory(myriad_plugin)
endif()
