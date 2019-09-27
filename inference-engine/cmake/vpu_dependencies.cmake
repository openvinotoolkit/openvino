# Copyright (C) 2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(VPU_SUPPORTED_SOC ma2450 ma2x8x mv0262)

foreach(soc IN LISTS VPU_SUPPORTED_SOC)
    string(TOUPPER "${soc}" soc_upper)
    set(var_name_file VPU_FIRMWARE_${soc_upper}_FILE)
    set(var_name VPU_FIRMWARE_${soc_upper})
    set(var_name_zip firmware_${soc}_mdk_R9.8.zip)
    if(${soc} STREQUAL "ma2450")
        set(var_name_zip firmware_${soc}_759W.zip)
    endif()

    if(NOT DEFINED ${var_name_file})
        RESOLVE_DEPENDENCY(${var_name}
            ARCHIVE_UNIFIED ${var_name_zip}
            TARGET_PATH "${TEMP}/vpu/firmware/${soc}"
            ENVIRONMENT "${var_name}"
            FOLDER)
        find_file(${var_name_file} NAMES "MvNCAPI-${soc}.mvcmd" PATHS "${VPU_FIRMWARE_${soc_upper}}/mvnc" NO_CMAKE_FIND_ROOT_PATH)
    endif()

    if(NOT ${var_name_file})
        message(FATAL_ERROR "[VPU] Missing ${soc} firmware, MvNCAPI-${soc}.mvcmd not found in ${VPU_FIRMWARE_${soc_upper}}/mvnc env $ENV{${var_name}} ")
    endif()

    debug_message(STATUS "${soc}=" ${${var_name_file}})

    set(firmware_out_file "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/MvNCAPI-${soc}.mvcmd")
    list(APPEND all_firmware_files ${firmware_out_file})

    add_custom_command(
        OUTPUT ${firmware_out_file}
        COMMAND
            ${CMAKE_COMMAND} -E copy ${${var_name_file}} ${firmware_out_file}
        MAIN_DEPENDENCY ${${var_name_file}}
        COMMENT "[VPU] Copy ${${var_name_file}} to ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
        VERBATIM)
endforeach()

add_custom_target(vpu_copy_firmware
    DEPENDS ${all_firmware_files}
    COMMENT "[VPU] Copy firmware files")
