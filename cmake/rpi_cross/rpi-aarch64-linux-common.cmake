# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(_ov_rpi_template_dir "${CMAKE_CURRENT_LIST_DIR}")

set(_ov_rpi_common_tools gcc g++ strip ar ranlib nm objcopy objdump readelf)
if(OV_RPI_EXTRA_TOOLS)
    list(APPEND _ov_rpi_common_tools ${OV_RPI_EXTRA_TOOLS})
endif()

foreach(_ov_tool IN LISTS _ov_rpi_common_tools)
    string(TOUPPER "${_ov_tool}" _ov_tool_upper)
    if(OV_RPI_EXECUTABLE_SUFFIX)
        set(_ov_tool_names
            "${OV_RPI_TOOLCHAIN_PREFIX}-${_ov_tool}${OV_RPI_EXECUTABLE_SUFFIX}"
            "${OV_RPI_TOOLCHAIN_PREFIX}-${_ov_tool}")
    else()
        set(_ov_tool_names "${OV_RPI_TOOLCHAIN_PREFIX}-${_ov_tool}")
    endif()
    find_program(OV_RPI_${_ov_tool_upper}
        NAMES ${_ov_tool_names}
        DOC "Path to ${OV_RPI_TOOLCHAIN_PREFIX}-${_ov_tool}")
endforeach()

if(NOT OV_RPI_GCC OR NOT OV_RPI_G++)
    message(FATAL_ERROR
        "Cannot find ${OV_RPI_TOOLCHAIN_PREFIX}-gcc / ${OV_RPI_TOOLCHAIN_PREFIX}-g++. "
        "${OV_RPI_MISSING_TOOLCHAIN_MESSAGE}")
endif()

set(CMAKE_C_COMPILER "${OV_RPI_GCC}" CACHE FILEPATH "Raspberry Pi ARM64 C compiler")
set(CMAKE_CXX_COMPILER "${OV_RPI_G++}" CACHE FILEPATH "Raspberry Pi ARM64 C++ compiler")

foreach(_ov_tool strip ar ranlib nm objcopy objdump readelf)
    string(TOUPPER "${_ov_tool}" _ov_tool_upper)
    if(OV_RPI_${_ov_tool_upper})
        string(TOUPPER "${_ov_tool}" _ov_cmake_tool)
        set(CMAKE_${_ov_cmake_tool} "${OV_RPI_${_ov_tool_upper}}" CACHE FILEPATH "Raspberry Pi ARM64 ${_ov_tool}")
    endif()
endforeach()
if(OV_RPI_LD)
    set(CMAKE_LINKER "${OV_RPI_LD}" CACHE FILEPATH "Raspberry Pi ARM64 linker")
endif()

set(_ov_rpi_pkg_config_names "${OV_RPI_TOOLCHAIN_PREFIX}-pkg-config" pkg-config pkgconf)
if(OV_RPI_EXECUTABLE_SUFFIX)
    set(_ov_rpi_pkg_config_names
        "${OV_RPI_TOOLCHAIN_PREFIX}-pkg-config${OV_RPI_EXECUTABLE_SUFFIX}"
        ${_ov_rpi_pkg_config_names})
endif()
find_program(OV_RPI_PKG_CONFIG
    NAMES ${_ov_rpi_pkg_config_names}
    DOC "pkg-config executable for the ARM64 target")
if(OV_RPI_PKG_CONFIG)
    set(PKG_CONFIG_EXECUTABLE "${OV_RPI_PKG_CONFIG}" CACHE FILEPATH "Path to ARM64 pkg-config")
endif()

if(OV_RPI_SYSROOT)
    file(TO_CMAKE_PATH "${OV_RPI_SYSROOT}" OV_RPI_SYSROOT)
    set(CMAKE_SYSROOT "${OV_RPI_SYSROOT}" CACHE PATH "Raspberry Pi ARM64 sysroot")
    set(CMAKE_FIND_ROOT_PATH "${CMAKE_SYSROOT}" CACHE STRING "CMake target find roots")

    set(_ov_rpi_wrapper_dir "${CMAKE_BINARY_DIR}/rpi-aarch64-linux-toolchain")
    file(MAKE_DIRECTORY "${_ov_rpi_wrapper_dir}")

    set(_ov_rpi_sysroot_flags "--sysroot=\"${CMAKE_SYSROOT}\"")
    if(EXISTS "${CMAKE_SYSROOT}/usr/include/aarch64-linux-gnu")
        set(_ov_rpi_multiarch_include "${CMAKE_SYSROOT}/usr/include/aarch64-linux-gnu")
        string(APPEND _ov_rpi_sysroot_flags " -isystem \"${_ov_rpi_multiarch_include}\"")
        foreach(_ov_rpi_flags_var CMAKE_C_FLAGS_INIT CMAKE_CXX_FLAGS_INIT CMAKE_ASM_FLAGS_INIT)
            if(NOT "${${_ov_rpi_flags_var}}" MATCHES "(^| )-isystem ${_ov_rpi_multiarch_include}($| )")
                string(APPEND ${_ov_rpi_flags_var} " -isystem ${_ov_rpi_multiarch_include}")
            endif()
        endforeach()
    endif()

    set(_ov_rpi_library_dirs)
    foreach(_ov_rpi_lib_dir IN LISTS OV_RPI_LIBRARY_DIR_SUFFIXES)
        list(APPEND _ov_rpi_library_dirs "${CMAKE_SYSROOT}/${_ov_rpi_lib_dir}")
    endforeach()
    foreach(_ov_rpi_lib_dir IN LISTS _ov_rpi_library_dirs)
        if(EXISTS "${_ov_rpi_lib_dir}")
            foreach(_ov_rpi_linker_flags_var
                    CMAKE_EXE_LINKER_FLAGS_INIT
                    CMAKE_SHARED_LINKER_FLAGS_INIT
                    CMAKE_MODULE_LINKER_FLAGS_INIT)
                if(NOT "${${_ov_rpi_linker_flags_var}}" MATCHES "(^| )-L${_ov_rpi_lib_dir}($| )")
                    string(APPEND ${_ov_rpi_linker_flags_var} " -L${_ov_rpi_lib_dir}")
                endif()
                if(NOT "${${_ov_rpi_linker_flags_var}}" MATCHES "(^| )-B${_ov_rpi_lib_dir}($| )")
                    string(APPEND ${_ov_rpi_linker_flags_var} " -B${_ov_rpi_lib_dir}")
                endif()
            endforeach()
        endif()
    endforeach()

    set(_ov_rpi_pkg_config_libdir)
    foreach(_ov_rpi_pkg_config_dir IN LISTS OV_RPI_PKG_CONFIG_DIR_SUFFIXES)
        if(_ov_rpi_pkg_config_libdir)
            string(APPEND _ov_rpi_pkg_config_libdir ":")
        endif()
        string(APPEND _ov_rpi_pkg_config_libdir "${CMAKE_SYSROOT}/${_ov_rpi_pkg_config_dir}")
    endforeach()
    set(ENV{PKG_CONFIG_SYSROOT_DIR} "${CMAKE_SYSROOT}")
    set(ENV{PKG_CONFIG_LIBDIR} "${_ov_rpi_pkg_config_libdir}")
endif()

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
