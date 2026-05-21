# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(NOT OV_RPI_SYSROOT)
    return()
endif()

set(_ov_rpi_grep_exe "${_ov_rpi_wrapper_dir}/grep.exe")
if(NOT EXISTS "${_ov_rpi_grep_exe}")
    set(_ov_rpi_grep_source "${_ov_rpi_template_dir}/windows-grep.cs")
    file(TO_NATIVE_PATH "${_ov_rpi_grep_source}" _ov_rpi_grep_source_native)
    file(TO_NATIVE_PATH "${_ov_rpi_grep_exe}" _ov_rpi_grep_exe_native)
    execute_process(COMMAND powershell -NoProfile -ExecutionPolicy Bypass
                            -Command "Add-Type -Path '${_ov_rpi_grep_source_native}' -OutputAssembly '${_ov_rpi_grep_exe_native}' -OutputType ConsoleApplication"
                    RESULT_VARIABLE _ov_rpi_grep_result
                    ERROR_VARIABLE _ov_rpi_grep_error)
    if(NOT _ov_rpi_grep_result EQUAL 0)
        message(FATAL_ERROR "Failed to generate grep.exe for Windows-hosted Raspberry Pi ARM64 cross build: ${_ov_rpi_grep_error}")
    endif()
endif()
set(ENV{PATH} "${_ov_rpi_wrapper_dir};$ENV{PATH}")

set(_ov_rpi_glibc_major "2")
set(_ov_rpi_glibc_minor "0")
foreach(_ov_rpi_features_header
        "${CMAKE_SYSROOT}/usr/include/features.h"
        "${CMAKE_SYSROOT}/include/features.h")
    if(EXISTS "${_ov_rpi_features_header}")
        file(STRINGS "${_ov_rpi_features_header}" _ov_rpi_features_lines
            REGEX "^#[ \t]*define[ \t]+__GLIBC(_MINOR)?__[ \t]+[0-9]+")
        foreach(_ov_rpi_features_line IN LISTS _ov_rpi_features_lines)
            if(_ov_rpi_features_line MATCHES "^#[ \t]*define[ \t]+__GLIBC__[ \t]+([0-9]+)")
                set(_ov_rpi_glibc_major "${CMAKE_MATCH_1}")
            elseif(_ov_rpi_features_line MATCHES "^#[ \t]*define[ \t]+__GLIBC_MINOR__[ \t]+([0-9]+)")
                set(_ov_rpi_glibc_minor "${CMAKE_MATCH_1}")
            endif()
        endforeach()
        break()
    endif()
endforeach()

foreach(_ov_rpi_tool_name gcc g++)
    string(TOUPPER "${_ov_rpi_tool_name}" _ov_rpi_tool_upper)
    set(OV_RPI_WRAPPER_REAL_TOOL "${OV_RPI_${_ov_rpi_tool_upper}}")
    set(OV_RPI_WRAPPER_FLAGS "${_ov_rpi_sysroot_flags}")
    set(_ov_rpi_wrapper "${_ov_rpi_wrapper_dir}/${OV_RPI_TOOLCHAIN_PREFIX}-${_ov_rpi_tool_name}.cmd")
    if(_ov_rpi_tool_name STREQUAL "g++")
        set(OV_RPI_WRAPPER_GLIBC_MAJOR "${_ov_rpi_glibc_major}")
        set(OV_RPI_WRAPPER_GLIBC_MINOR "${_ov_rpi_glibc_minor}")
        configure_file("${_ov_rpi_template_dir}/windows-cxx-wrapper.cmd.in" "${_ov_rpi_wrapper}" @ONLY)
    else()
        configure_file("${_ov_rpi_template_dir}/windows-tool-wrapper.cmd.in" "${_ov_rpi_wrapper}" @ONLY)
    endif()
endforeach()

foreach(_ov_rpi_tool_name strip ar ranlib nm objcopy objdump readelf as ld)
    string(TOUPPER "${_ov_rpi_tool_name}" _ov_rpi_tool_upper)
    if(NOT OV_RPI_${_ov_rpi_tool_upper})
        continue()
    endif()

    if(_ov_rpi_tool_name STREQUAL "ar")
        set(OV_RPI_WRAPPER_REAL_TOOL "${OV_RPI_AR}")
        set(_ov_rpi_ar_wrapper_ps1 "${_ov_rpi_wrapper_dir}/ar-rsp-normalize.ps1")
        configure_file("${_ov_rpi_template_dir}/ar-rsp-normalize.ps1.in" "${_ov_rpi_ar_wrapper_ps1}" @ONLY)
        file(TO_NATIVE_PATH "${_ov_rpi_ar_wrapper_ps1}" _ov_rpi_ar_wrapper_ps1_native)
        file(TO_NATIVE_PATH "${OV_RPI_WRAPPER_REAL_TOOL}" _ov_rpi_real_tool_native)
        set(OV_RPI_AR_WRAPPER_PS1_NATIVE "${_ov_rpi_ar_wrapper_ps1_native}")
        set(OV_RPI_WRAPPER_REAL_TOOL_NATIVE "${_ov_rpi_real_tool_native}")
        set(_ov_rpi_wrapper "${_ov_rpi_wrapper_dir}/${OV_RPI_TOOLCHAIN_PREFIX}-${_ov_rpi_tool_name}.cmd")
        configure_file("${_ov_rpi_template_dir}/windows-ar-wrapper.cmd.in" "${_ov_rpi_wrapper}" @ONLY)
    elseif(_ov_rpi_tool_name STREQUAL "as")
        set(OV_RPI_WRAPPER_REAL_TOOL "${OV_RPI_GCC}")
        set(OV_RPI_WRAPPER_FLAGS "${_ov_rpi_sysroot_flags} -x assembler-with-cpp -c")
        set(_ov_rpi_wrapper "${_ov_rpi_wrapper_dir}/${OV_RPI_TOOLCHAIN_PREFIX}-${_ov_rpi_tool_name}.cmd")
        configure_file("${_ov_rpi_template_dir}/windows-tool-wrapper.cmd.in" "${_ov_rpi_wrapper}" @ONLY)
    else()
        set(OV_RPI_WRAPPER_REAL_TOOL "${OV_RPI_${_ov_rpi_tool_upper}}")
        set(OV_RPI_WRAPPER_FLAGS "")
        set(_ov_rpi_wrapper "${_ov_rpi_wrapper_dir}/${OV_RPI_TOOLCHAIN_PREFIX}-${_ov_rpi_tool_name}.cmd")
        configure_file("${_ov_rpi_template_dir}/windows-tool-wrapper.cmd.in" "${_ov_rpi_wrapper}" @ONLY)
    endif()
endforeach()

find_program(OV_RPI_REAL_SCONS
    NAMES scons.exe scons
    DOC "SCons executable used by the Windows-hosted Raspberry Pi ARM64 cross toolchain")
if(OV_RPI_REAL_SCONS)
    set(_ov_rpi_scons_site_dir "${_ov_rpi_wrapper_dir}/scons-site")
    file(MAKE_DIRECTORY "${_ov_rpi_scons_site_dir}")
    configure_file("${_ov_rpi_template_dir}/scons-site-init.py" "${_ov_rpi_scons_site_dir}/site_init.py" COPYONLY)

    file(TO_NATIVE_PATH "${OV_RPI_REAL_SCONS}" _ov_rpi_real_scons_native)
    file(TO_NATIVE_PATH "${_ov_rpi_scons_site_dir}" _ov_rpi_scons_site_dir_native)
    set(OV_RPI_REAL_SCONS_NATIVE "${_ov_rpi_real_scons_native}")
    set(OV_RPI_SCONS_SITE_DIR_NATIVE "${_ov_rpi_scons_site_dir_native}")
    set(_ov_rpi_scons_wrapper "${_ov_rpi_wrapper_dir}/scons-rpi-aarch64-linux.cmd")
    configure_file("${_ov_rpi_template_dir}/scons-rpi-aarch64-linux.cmd.in" "${_ov_rpi_scons_wrapper}" @ONLY)
    set(SCONS "${_ov_rpi_scons_wrapper}" CACHE FILEPATH "SCons wrapper for Raspberry Pi ARM64 cross build" FORCE)
endif()

set(ARM_COMPUTE_TOOLCHAIN_PREFIX "${_ov_rpi_wrapper_dir}/${OV_RPI_TOOLCHAIN_PREFIX}-" CACHE STRING
    "Arm Compute Library toolchain prefix for Windows-hosted Raspberry Pi ARM64 cross build" FORCE)

set(CMAKE_C_COMPILER "${_ov_rpi_wrapper_dir}/${OV_RPI_TOOLCHAIN_PREFIX}-gcc.cmd" CACHE FILEPATH
    "Raspberry Pi ARM64 C compiler" FORCE)
set(CMAKE_CXX_COMPILER "${_ov_rpi_wrapper_dir}/${OV_RPI_TOOLCHAIN_PREFIX}-g++.cmd" CACHE FILEPATH
    "Raspberry Pi ARM64 C++ compiler" FORCE)
