# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(OV_RPI_TOOLCHAIN_PREFIX "aarch64-linux-gnu" CACHE STRING "GNU target triplet prefix")
set(OV_RPI_SYSROOT "" CACHE PATH "Optional Raspberry Pi ARM64 sysroot")
set(_ov_rpi_template_dir "${CMAKE_CURRENT_LIST_DIR}/rpi_cross")

foreach(_ov_tool gcc g++ strip ar ranlib nm objcopy objdump readelf)
    string(TOUPPER "${_ov_tool}" _ov_tool_upper)
    find_program(OV_RPI_${_ov_tool_upper}
        NAMES "${OV_RPI_TOOLCHAIN_PREFIX}-${_ov_tool}"
        DOC "Path to ${OV_RPI_TOOLCHAIN_PREFIX}-${_ov_tool}")
endforeach()

if(NOT OV_RPI_GCC OR NOT OV_RPI_G++)
    message(FATAL_ERROR
        "Cannot find ${OV_RPI_TOOLCHAIN_PREFIX}-gcc / ${OV_RPI_TOOLCHAIN_PREFIX}-g++. "
        "Install a Linux AArch64 GNU cross toolchain or set OV_RPI_TOOLCHAIN_PREFIX.")
endif()

set(CMAKE_C_COMPILER "${OV_RPI_GCC}" CACHE FILEPATH "Raspberry Pi ARM64 C compiler")
set(CMAKE_CXX_COMPILER "${OV_RPI_G++}" CACHE FILEPATH "Raspberry Pi ARM64 C++ compiler")

if(OV_RPI_STRIP)
    set(CMAKE_STRIP "${OV_RPI_STRIP}" CACHE FILEPATH "Raspberry Pi ARM64 strip")
endif()
if(OV_RPI_AR)
    set(CMAKE_AR "${OV_RPI_AR}" CACHE FILEPATH "Raspberry Pi ARM64 ar")
endif()
if(OV_RPI_RANLIB)
    set(CMAKE_RANLIB "${OV_RPI_RANLIB}" CACHE FILEPATH "Raspberry Pi ARM64 ranlib")
endif()
if(OV_RPI_NM)
    set(CMAKE_NM "${OV_RPI_NM}" CACHE FILEPATH "Raspberry Pi ARM64 nm")
endif()
if(OV_RPI_OBJCOPY)
    set(CMAKE_OBJCOPY "${OV_RPI_OBJCOPY}" CACHE FILEPATH "Raspberry Pi ARM64 objcopy")
endif()
if(OV_RPI_OBJDUMP)
    set(CMAKE_OBJDUMP "${OV_RPI_OBJDUMP}" CACHE FILEPATH "Raspberry Pi ARM64 objdump")
endif()
if(OV_RPI_READELF)
    set(CMAKE_READELF "${OV_RPI_READELF}" CACHE FILEPATH "Raspberry Pi ARM64 readelf")
endif()

find_program(OV_RPI_PKG_CONFIG
    NAMES "${OV_RPI_TOOLCHAIN_PREFIX}-pkg-config" pkg-config pkgconf
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
    foreach(_ov_rpi_lib_dir "${CMAKE_SYSROOT}/lib/aarch64-linux-gnu" "${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu")
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

    set(ENV{PKG_CONFIG_SYSROOT_DIR} "${CMAKE_SYSROOT}")
    set(ENV{PKG_CONFIG_LIBDIR}
        "${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu/pkgconfig:${CMAKE_SYSROOT}/usr/lib/pkgconfig:${CMAKE_SYSROOT}/usr/share/pkgconfig")
endif()

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

unset(_ov_rpi_lib_dir)
unset(_ov_rpi_linker_flags_var)
unset(_ov_rpi_multiarch_include)
unset(_ov_rpi_flags_var)
unset(_ov_rpi_sysroot_flags)
unset(_ov_rpi_template_dir)
unset(_ov_rpi_tool_name)
unset(_ov_rpi_wrapper)
unset(_ov_rpi_wrapper_dir)
unset(_ov_tool)
unset(_ov_tool_upper)
unset(OV_RPI_WRAPPER_FLAGS)
unset(OV_RPI_WRAPPER_REAL_TOOL)
