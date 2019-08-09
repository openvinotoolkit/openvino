# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#
# Locate firmware files
#

if (ENABLE_MYRIAD)
    find_file(VPU_FIRMWARE_MA2450_FILE MvNCAPI-ma2450.mvcmd "${VPU_FIRMWARE_MA2450}/mvnc")
    find_file(VPU_FIRMWARE_MA2X8X_FILE MvNCAPI-ma2x8x.mvcmd "${VPU_FIRMWARE_MA2X8X}/mvnc")
    
    if(NOT VPU_FIRMWARE_MA2450_FILE OR NOT VPU_FIRMWARE_MA2X8X_FILE)
        message(FATAL_ERROR "[VPU] Missing firmware")
    endif()
endif()

#
# Build common part
#

add_subdirectory(graph_transformer)

add_subdirectory(
    "${IE_MAIN_SOURCE_DIR}/thirdparty/movidius"
    "${CMAKE_CURRENT_BINARY_DIR}/thirdparty/movidius")

#
# Build plugins
#

set(plugin_target "")

if(ENABLE_MYRIAD)
    add_subdirectory(myriad_plugin)
    set(plugin_target "myriadPlugin")
endif()


if(ENABLE_MYRIAD)
    set(firmware_out_dir "$<TARGET_FILE_DIR:${plugin_target}>")
    add_custom_target(vpu_copy_firmware ALL
        COMMAND "${CMAKE_COMMAND}" -E copy "${VPU_FIRMWARE_MA2450_FILE}" "${firmware_out_dir}/MvNCAPI-ma2450.mvcmd"
        COMMAND "${CMAKE_COMMAND}" -E copy "${VPU_FIRMWARE_MA2X8X_FILE}" "${firmware_out_dir}/MvNCAPI-ma2x8x.mvcmd"
        COMMENT "[VPU] Copy firmware")

endif()
