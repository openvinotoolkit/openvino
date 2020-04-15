# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(CMAKE_VERSION VERSION_GREATER 3.9.6)
    include_guard(GLOBAL)
else()
    if(__CURRENT_FILE_VAR__)
      return()
    endif()
    set(__CURRENT_FILE_VAR__ TRUE)
endif()

include(dependency_solver)

set(VPU_SUPPORTED_FIRMWARES usb-ma2450 usb-ma2x8x pcie-ma248x)

#
# Default packages
#

set(FIRMWARE_PACKAGE_VERSION 1076)

#
# CMake variables to override default firmware files
#

foreach(firmware_name IN LISTS VPU_SUPPORTED_FIRMWARES)
    string(TOUPPER "${firmware_name}" firmware_name_upper)

    set(firmware_name_full ${firmware_name}.mvcmd)
    # Handle PCIe elf firmware for Windows
    if (WIN32 AND "${firmware_name}" STREQUAL "pcie-ma248x")
        set(firmware_name_full ${firmware_name}.elf)
    endif ()

    reset_deps_cache(VPU_FIRMWARE_${firmware_name_upper}_FILE)

    RESOLVE_DEPENDENCY(VPU_FIRMWARE_${firmware_name_upper}
        ARCHIVE_UNIFIED firmware_${firmware_name}_${FIRMWARE_PACKAGE_VERSION}.zip
        TARGET_PATH "${TEMP}/vpu/firmware/${firmware_name}"
        ENVIRONMENT "VPU_FIRMWARE_${firmware_name_upper}_FILE"
        FOLDER)
    debug_message(STATUS "${firmware_name}=" ${VPU_FIRMWARE_${firmware_name_upper}})

    update_deps_cache(
        VPU_FIRMWARE_${firmware_name_upper}_FILE
        "${VPU_FIRMWARE_${firmware_name_upper}}/mvnc/${firmware_name_full}"
        "[VPU] ${firmware_name_full} firmware")

    find_file(
        VPU_FIRMWARE_${firmware_name_upper}_FILE
        NAMES ${firmware_name_full}
        NO_CMAKE_FIND_ROOT_PATH)
    if(NOT VPU_FIRMWARE_${firmware_name_upper}_FILE)
        message(FATAL_ERROR "[VPU] Missing ${firmware_name_full} firmware")
    endif()
endforeach()

#
# `vpu_copy_firmware` CMake target
#

foreach(firmware_name IN LISTS VPU_SUPPORTED_FIRMWARES)
    string(TOUPPER "${firmware_name}" firmware_name_upper)
    set(var_name VPU_FIRMWARE_${firmware_name_upper}_FILE)

    set(firmware_out_file "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${firmware_name}.mvcmd")

    # Handle PCIe elf firmware for Windows
    if (WIN32 AND "${firmware_name}" STREQUAL "pcie-ma248x")
        set(firmware_out_file "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${firmware_name}.elf")
    endif ()

    list(APPEND all_firmware_files ${firmware_out_file})
    add_custom_command(
        OUTPUT ${firmware_out_file}
        COMMAND
            ${CMAKE_COMMAND} -E copy ${${var_name}} ${firmware_out_file}
        MAIN_DEPENDENCY ${${var_name}}
        COMMENT "[VPU] Copy ${${var_name}} to ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
        VERBATIM)

    install(FILES ${${var_name}}
        DESTINATION ${IE_CPACK_LIBRARY_PATH}
        COMPONENT myriad)
endforeach()

add_custom_target(vpu_copy_firmware
    DEPENDS ${all_firmware_files}
    COMMENT "[VPU] Copy firmware files")

#
# libusb
#

if(ANDROID)
    RESOLVE_DEPENDENCY(LIBUSB
        ARCHIVE_ANDROID "libusb_39409_android.tgz"
        TARGET_PATH "${TEMP}/vpu/libusb")
    debug_message(STATUS "LIBUSB=" ${LIBUSB})

    set(LIBUSB_INCLUDE_DIR "${LIBUSB}/include")
    set(LIBUSB_LIBRARY "${LIBUSB}/libs/${ANDROID_ABI}/libusb1.0.so")

    log_rpath_from_dir(LIBUSB "${LIBUSB}/libs/${ANDROID_ABI}")
endif()
