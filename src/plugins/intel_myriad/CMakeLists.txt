# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#
# Build common part
#

if(CMAKE_COMPILER_IS_GNUCC)
    ie_add_compiler_flags(-Wno-all)
endif()

add_subdirectory(common)

if(ENABLE_INTEL_MYRIAD)
    add_subdirectory(third_party)
    add_subdirectory(graph_transformer)
    add_subdirectory(myriad_plugin)

    # TODO: custom installation or totally remove
    if(DEFINED VPU_CLC_MA2X8X_ROOT AND NOT CPACK_GENERATOR STREQUAL "DEB")
        install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/custom_kernels/
                DESTINATION ${OV_CPACK_PLUGINSDIR}/vpu_custom_kernels
                COMPONENT myriad)
        install(DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/vpu_custom_kernels/
                DESTINATION ${OV_CPACK_PLUGINSDIR}/vpu_custom_kernels
                COMPONENT myriad)
        install(DIRECTORY ${VPU_CLC_MA2X8X_ROOT}/
                DESTINATION tools/cl_compiler
                COMPONENT myriad
                USE_SOURCE_PERMISSIONS
                PATTERN ie_dependency.info EXCLUDE)
    endif()
endif()
