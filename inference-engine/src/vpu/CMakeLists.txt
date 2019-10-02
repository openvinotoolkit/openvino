# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#
# Build common part
#

add_subdirectory(common)

if(ENABLE_MYRIAD)
    add_subdirectory(graph_transformer)
    add_subdirectory(
            "${IE_MAIN_SOURCE_DIR}/thirdparty/movidius"
            "${CMAKE_CURRENT_BINARY_DIR}/thirdparty/movidius")

    export(TARGETS mvnc XLink NAMESPACE IE:: APPEND FILE "${CMAKE_BINARY_DIR}/targets_developer.cmake")

    #
    # Build plugins
    #
    add_subdirectory(myriad_plugin)
endif()