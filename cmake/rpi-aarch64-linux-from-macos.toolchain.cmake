# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(OV_RPI_TOOLCHAIN_PREFIX "aarch64-linux-gnu" CACHE STRING "GNU target triplet prefix")
set(OV_RPI_SYSROOT "" CACHE PATH "Optional Raspberry Pi ARM64 sysroot")
set(OV_RPI_MISSING_TOOLCHAIN_MESSAGE "Install a Linux AArch64 GNU cross toolchain or set OV_RPI_TOOLCHAIN_PREFIX.")
set(OV_RPI_LIBRARY_DIR_SUFFIXES lib/aarch64-linux-gnu usr/lib/aarch64-linux-gnu)
set(OV_RPI_PKG_CONFIG_DIR_SUFFIXES
    usr/lib/aarch64-linux-gnu/pkgconfig
    usr/lib/pkgconfig
    usr/share/pkgconfig)

include("${CMAKE_CURRENT_LIST_DIR}/rpi_cross/rpi-aarch64-linux-common.cmake")

if(OV_RPI_SYSROOT)
    foreach(_ov_rpi_tool_name gcc g++)
        string(TOUPPER "${_ov_rpi_tool_name}" _ov_rpi_tool_upper)
        set(OV_RPI_WRAPPER_REAL_TOOL "${OV_RPI_${_ov_rpi_tool_upper}}")
        set(OV_RPI_WRAPPER_FLAGS "${_ov_rpi_sysroot_flags}")
        set(_ov_rpi_wrapper "${_ov_rpi_wrapper_dir}/${OV_RPI_TOOLCHAIN_PREFIX}-${_ov_rpi_tool_name}")
        configure_file("${_ov_rpi_template_dir}/macos-tool-wrapper.sh.in" "${_ov_rpi_wrapper}" @ONLY)
        file(CHMOD "${_ov_rpi_wrapper}"
            PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
    endforeach()
    foreach(_ov_rpi_tool_name strip ar ranlib nm objcopy objdump readelf)
        string(TOUPPER "${_ov_rpi_tool_name}" _ov_rpi_tool_upper)
        if(OV_RPI_${_ov_rpi_tool_upper})
            set(OV_RPI_WRAPPER_REAL_TOOL "${OV_RPI_${_ov_rpi_tool_upper}}")
            set(OV_RPI_WRAPPER_FLAGS "")
            set(_ov_rpi_wrapper "${_ov_rpi_wrapper_dir}/${OV_RPI_TOOLCHAIN_PREFIX}-${_ov_rpi_tool_name}")
            configure_file("${_ov_rpi_template_dir}/macos-tool-wrapper.sh.in" "${_ov_rpi_wrapper}" @ONLY)
            file(CHMOD "${_ov_rpi_wrapper}"
                PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
        endif()
    endforeach()

    set(CMAKE_C_COMPILER "${_ov_rpi_wrapper_dir}/${OV_RPI_TOOLCHAIN_PREFIX}-gcc" CACHE FILEPATH "Raspberry Pi ARM64 C compiler" FORCE)
    set(CMAKE_CXX_COMPILER "${_ov_rpi_wrapper_dir}/${OV_RPI_TOOLCHAIN_PREFIX}-g++" CACHE FILEPATH "Raspberry Pi ARM64 C++ compiler" FORCE)
endif()
