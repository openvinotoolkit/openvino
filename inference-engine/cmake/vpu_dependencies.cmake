# Copyright (C) 2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(VPU_SUPPORTED_SOC ma2450 ma2x8x mv0262)

#
# Default firmware packages
#

RESOLVE_DEPENDENCY(VPU_FIRMWARE_MA2450
    ARCHIVE_UNIFIED firmware_ma2450_759W.zip
    TARGET_PATH "${TEMP}/vpu/firmware/ma2450"
    ENVIRONMENT "VPU_FIRMWARE_MA2450"
    FOLDER)
debug_message(STATUS "ma2450=" ${VPU_FIRMWARE_MA2450})

RESOLVE_DEPENDENCY(VPU_FIRMWARE_MV0262
    ARCHIVE_UNIFIED firmware_mv0262_mdk_R9.8.zip
    TARGET_PATH "${TEMP}/vpu/firmware/mv0262"
    ENVIRONMENT "VPU_FIRMWARE_MV0262"
    FOLDER)
debug_message(STATUS "mv0262=" ${VPU_FIRMWARE_MV0262})

RESOLVE_DEPENDENCY(VPU_FIRMWARE_MA2X8X
    ARCHIVE_UNIFIED firmware_ma2x8x_mdk_R9.8.zip
    TARGET_PATH "${TEMP}/vpu/firmware/ma2x8x"
    ENVIRONMENT "VPU_FIRMWARE_MA2X8X"
    FOLDER)
debug_message(STATUS "ma2x8x=" ${VPU_FIRMWARE_MA2X8X})

#
# CMake variables to override default firmware files
#

foreach(soc IN LISTS VPU_SUPPORTED_SOC)
    string(TOUPPER "${soc}" soc_upper)
    set(var_name VPU_FIRMWARE_${soc_upper}_FILE)

    find_file(${var_name} MvNCAPI-${soc}.mvcmd "${VPU_FIRMWARE_${soc_upper}}/mvnc")
    if(NOT ${var_name})
        message(FATAL_ERROR "[VPU] Missing ${soc} firmware")
    endif()
endforeach()

#
# `vpu_copy_firmware` CMake target
#

foreach(soc IN LISTS VPU_SUPPORTED_SOC)
    string(TOUPPER "${soc}" soc_upper)
    set(var_name VPU_FIRMWARE_${soc_upper}_FILE)

    set(firmware_out_file "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/MvNCAPI-${soc}.mvcmd")
    list(APPEND all_firmware_files ${firmware_out_file})

    add_custom_command(
        OUTPUT ${firmware_out_file}
        COMMAND
            ${CMAKE_COMMAND} -E copy ${${var_name}} ${firmware_out_file}
        MAIN_DEPENDENCY ${${var_name}}
        COMMENT "[VPU] Copy ${${var_name}} to ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
        VERBATIM)
endforeach()

add_custom_target(vpu_copy_firmware
    DEPENDS ${all_firmware_files}
    COMMENT "[VPU] Copy firmware files")
