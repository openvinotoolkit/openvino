# Copyright (C) 2018-2020 Intel Corporation
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

    #
    # Build plugins
    #

    add_subdirectory(myriad_plugin)

    if(DEFINED VPU_CLC_MA2X8X_ROOT)
        install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/custom_kernels/
                DESTINATION ${IE_CPACK_LIBRARY_PATH}/vpu_custom_kernels
                COMPONENT myriad)
        install(DIRECTORY ${VPU_CLC_MA2X8X_ROOT}/
                DESTINATION deployment_tools/tools/cl_compiler
                COMPONENT myriad
                PATTERN ie_dependency.info EXCLUDE)
    endif()
endif()
