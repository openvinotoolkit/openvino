# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include_guard(GLOBAL)

set(VPU_SUPPORTED_FIRMWARES usb-ma2x8x pcie-ma2x8x)
set(VPU_SUPPORTED_FIRMWARES_HASH
    "420b300d193f7fcfe7e3f9bbec6c247d65b784a500b5cd2effb7cb1ec6e1b209"
    "bfe3caf270b168b9de18ef88f04bde3907d7d12a679f1fa7cc580423c35db637")

#
# Default packages
#

set(FIRMWARE_PACKAGE_VERSION 1688)
set(VPU_CLC_MA2X8X_VERSION "movi-cltools-20.09.2")

#
# CMake variables to override default firmware files
#
list(LENGTH VPU_SUPPORTED_FIRMWARES num_firmwares)
math(EXPR num_firmwares "${num_firmwares} - 1")
foreach(idx RANGE 0 ${num_firmwares})
    list(GET VPU_SUPPORTED_FIRMWARES ${idx} firmware_name)
    list(GET VPU_SUPPORTED_FIRMWARES_HASH ${idx} hash)
    string(TOUPPER "${firmware_name}" firmware_name_upper)

    set(firmware_name_full ${firmware_name}.mvcmd)
    # Handle PCIe elf firmware for Windows
    if (WIN32 AND "${firmware_name}" STREQUAL "pcie-ma2x8x")
        set(firmware_name_full ${firmware_name}.elf)
    endif ()

    reset_deps_cache(VPU_FIRMWARE_${firmware_name_upper}_FILE)

    RESOLVE_DEPENDENCY(VPU_FIRMWARE_${firmware_name_upper}
        ARCHIVE_UNIFIED VPU/${firmware_name}/firmware_${firmware_name}_${FIRMWARE_PACKAGE_VERSION}.zip
        TARGET_PATH "${TEMP}/vpu/firmware/${firmware_name}"
        ENVIRONMENT "VPU_FIRMWARE_${firmware_name_upper}_FILE"
        FOLDER
        SHA256 ${hash})
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

    set(firmware_out_file "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/${firmware_name}.mvcmd")

    # Handle PCIe elf firmware for Windows
    if (WIN32 AND "${firmware_name}" STREQUAL "pcie-ma2x8x")
        set(firmware_out_file "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/${firmware_name}.elf")
    endif ()

    list(APPEND all_firmware_files ${firmware_out_file})
    add_custom_command(
        OUTPUT ${firmware_out_file}
        COMMAND
            ${CMAKE_COMMAND} -E copy ${${var_name}} ${firmware_out_file}
        MAIN_DEPENDENCY ${${var_name}}
        COMMENT "[VPU] Copy ${${var_name}} to ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}"
        VERBATIM)

    install(FILES ${${var_name}}
        DESTINATION ${IE_CPACK_RUNTIME_PATH}
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
        TARGET_PATH "${TEMP}/vpu/libusb"
        SHA256 "f9e73e95bc769abf1e9910a59b138cf387205e1b4c4e5faec236136fb1d325f7")
    debug_message(STATUS "LIBUSB=" ${LIBUSB})

    set(LIBUSB_INCLUDE_DIR "${LIBUSB}/include")
    set(LIBUSB_LIBRARY "${LIBUSB}/libs/${ANDROID_ABI}/libusb1.0.so")

    log_rpath_from_dir(LIBUSB "${LIBUSB}/libs/${ANDROID_ABI}")
endif()

#
# OpenCL compiler
#

if(LINUX AND LINUX_OS_NAME MATCHES "Ubuntu")
    if(DEFINED ENV{THIRDPARTY_SERVER_PATH})
        set(IE_PATH_TO_DEPS "$ENV{THIRDPARTY_SERVER_PATH}")
    elseif(DEFINED THIRDPARTY_SERVER_PATH)
        set(IE_PATH_TO_DEPS "${THIRDPARTY_SERVER_PATH}")
    else()
        message(WARNING "VPU_OCL_COMPILER is not found. Some tests will skipped")
    endif()

    if(DEFINED IE_PATH_TO_DEPS)
        message(STATUS "THIRDPARTY_SERVER_PATH=${IE_PATH_TO_DEPS}")

        reset_deps_cache(VPU_CLC_MA2X8X_ROOT)
        reset_deps_cache(VPU_CLC_MA2X8X_COMMAND)

        RESOLVE_DEPENDENCY(VPU_CLC_MA2X8X
            ARCHIVE_LIN "VPU_OCL_compiler/${VPU_CLC_MA2X8X_VERSION}.tar.gz"
            TARGET_PATH "${TEMP}/vpu/clc/ma2x8x/${VPU_CLC_MA2X8X_VERSION}"
            ENVIRONMENT "VPU_CLC_MA2X8X_COMMAND"
            SHA256 "0a864bd0e11cee2d85ac7e451dddae19216c8bc9bb50e1a8e0151ab97d5e3c8d")
        debug_message(STATUS "VPU_CLC_MA2X8X=" ${VPU_CLC_MA2X8X})

        update_deps_cache(
            VPU_CLC_MA2X8X_ROOT
            "${VPU_CLC_MA2X8X}"
            "[VPU] Root directory of OpenCL compiler")

        update_deps_cache(
            VPU_CLC_MA2X8X_COMMAND
            "${VPU_CLC_MA2X8X}/bin/clc"
            "[VPU] OpenCL compiler")

        find_program(VPU_CLC_MA2X8X_COMMAND clc)
        unset (IE_PATH_TO_DEPS)
    endif()
endif()

#
# `vpu_custom_kernels` CMake target
#

add_library(vpu_custom_kernels INTERFACE)

function(add_vpu_compile_custom_kernels)
    set(SRC_DIR "${IE_MAIN_SOURCE_DIR}/src/vpu/custom_kernels")
    set(DST_DIR "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/vpu_custom_kernels")

    file(MAKE_DIRECTORY "${DST_DIR}")

    file(GLOB XML_FILES "${SRC_DIR}/*.xml")
    file(GLOB CL_FILES "${SRC_DIR}/*.cl")

    foreach(xml_file IN LISTS XML_FILES)
        get_filename_component(xml_file_name ${xml_file} NAME)

        set(out_file "${DST_DIR}/${xml_file_name}")
        list(APPEND all_output_files ${out_file})

        add_custom_command(
            OUTPUT ${out_file}
            COMMAND
                ${CMAKE_COMMAND} -E copy ${xml_file} ${out_file}
            MAIN_DEPENDENCY ${xml_file}
            COMMENT "[VPU] Copy ${xml_file} to ${DST_DIR}"
            VERBATIM)
    endforeach()

    foreach(cl_file IN LISTS CL_FILES)
        get_filename_component(cl_file_name ${cl_file} NAME_WE)

        set(out_file "${DST_DIR}/${cl_file_name}.bin")
        list(APPEND all_output_files ${out_file})

        add_custom_command(
            OUTPUT ${out_file}
            COMMAND
                ${CMAKE_COMMAND} -E env
                    "SHAVE_LDSCRIPT_DIR=${VPU_CLC_MA2X8X}/ldscripts/ma2x8x"
                    "SHAVE_MA2X8XLIBS_DIR=${VPU_CLC_MA2X8X}/lib"
                    "SHAVE_MOVIASM_DIR=${VPU_CLC_MA2X8X}/bin"
                    "SHAVE_MYRIAD_LD_DIR=${VPU_CLC_MA2X8X}/bin"
                ${VPU_CLC_MA2X8X_COMMAND} --strip-binary-header -d ma2x8x ${cl_file} -o ${out_file}
            MAIN_DEPENDENCY ${cl_file}
            DEPENDS ${VPU_CLC_MA2X8X_COMMAND}
            COMMENT "[VPU] Compile ${cl_file}"
            VERBATIM)
    endforeach()

    add_custom_target(vpu_compile_custom_kernels
        DEPENDS ${all_output_files}
        COMMENT "[VPU] Compile custom kernels")

    add_dependencies(vpu_custom_kernels vpu_compile_custom_kernels)
    target_compile_definitions(vpu_custom_kernels INTERFACE "VPU_HAS_CUSTOM_KERNELS")
endfunction()

if(VPU_CLC_MA2X8X_COMMAND)
    add_vpu_compile_custom_kernels()
endif()
