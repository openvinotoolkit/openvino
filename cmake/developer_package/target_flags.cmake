# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Target system specific flags

if(CMAKE_CL_64)
  set(MSVC64 ON)
endif()

if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpmachine
                  OUTPUT_VARIABLE OPENVINO_GCC_TARGET_MACHINE
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(OPENVINO_GCC_TARGET_MACHINE MATCHES "amd64|x86_64|AMD64")
    set(MINGW64 ON)
  endif()
endif()

if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "amd64.*|x86_64.*|AMD64.*")
  set(OV_HOST_ARCH X86_64)
elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "i686.*|i386.*|x86.*|amd64.*|AMD64.*")
  set(OV_HOST_ARCH X86)
elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^(arm64.*|aarch64.*|AARCH64.*|ARM64.*)")
  set(OV_HOST_ARCH AARCH64)
elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^(arm.*|ARM.*)")
  set(OV_HOST_ARCH ARM)
elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^riscv64$")
  set(OV_HOST_ARCH RISCV64)
endif()

macro(_ov_detect_arch_by_processor_type)
  if(CMAKE_OSX_ARCHITECTURES AND APPLE)
    if(CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
      set(OV_ARCH AARCH64)
    elseif(CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64")
      set(OV_ARCH X86_64)
    elseif(CMAKE_OSX_ARCHITECTURES MATCHES ".*x86_64.*" AND CMAKE_OSX_ARCHITECTURES MATCHES ".*arm64.*")
      set(OV_ARCH UNIVERSAL2)
    else()
      message(FATAL_ERROR "Unsupported value: CMAKE_OSX_ARCHITECTURES = ${CMAKE_OSX_ARCHITECTURES}")
    endif()
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "amd64.*|x86_64.*|AMD64.*")
    set(OV_ARCH X86_64)
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "i686.*|i386.*|x86.*|amd64.*|AMD64.*|wasm")
    set(OV_ARCH X86)
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64.*|aarch64.*|AARCH64.*|ARM64.*|armv8)")
    set(OV_ARCH AARCH64)
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm.*|ARM.*)")
    set(OV_ARCH ARM)
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^riscv64$")
    set(OV_ARCH RISCV64)
  endif()
endmacro()

macro(_ov_process_msvc_generator_platform)
  # if cmake -A <ARM|ARM64|x64|Win32> is passed
  if(CMAKE_GENERATOR_PLATFORM STREQUAL "ARM64")
    set(OV_ARCH AARCH64)
  elseif(CMAKE_GENERATOR_PLATFORM STREQUAL "ARM")
    set(OV_ARCH ARM)
  elseif(CMAKE_GENERATOR_PLATFORM STREQUAL "x64")
    set(OV_ARCH X86_64)
  elseif(CMAKE_GENERATOR_PLATFORM STREQUAL "Win32")
    set(OV_ARCH X86)
  else()
    _ov_detect_arch_by_processor_type()
  endif()
endmacro()

if(MSVC64 OR MINGW64)
  _ov_process_msvc_generator_platform()
elseif(MINGW OR (MSVC AND NOT CMAKE_CROSSCOMPILING))
  _ov_process_msvc_generator_platform()
else()
  _ov_detect_arch_by_processor_type()
endif()

set(HOST_${OV_HOST_ARCH} ON)
set(${OV_ARCH} ON)

if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    set(EMSCRIPTEN ON)
endif()

if(UNIX AND NOT (APPLE OR ANDROID OR EMSCRIPTEN OR CYGWIN))
    set(LINUX ON)
endif()

if(CMAKE_VERSION VERSION_LESS 3.25 AND CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
    # the variable is available since 3.25
    # https://cmake.org/cmake/help/latest/variable/CMAKE_HOST_LINUX.html
    set(CMAKE_HOST_LINUX ON)
endif()

if(ENV{OECORE_NATIVE_SYSROOT} AND AARCH64)
    set(YOCTO_AARCH64 ON)
endif()

if(CMAKE_HOST_LINUX AND LINUX)
    if(EXISTS "/etc/debian_version")
        set(OV_OS_DEBIAN ON)
    elseif(EXISTS "/etc/redhat-release")
        set(OV_OS_RHEL ON)
    endif()
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "^(Apple)?Clang$")
    set(OV_COMPILER_IS_CLANG ON)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
        set(OV_COMPILER_IS_APPLECLANG ON)
    endif()
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    set(OV_COMPILER_IS_INTEL_LLVM ON)
endif()

get_property(OV_GENERATOR_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)

function(ov_detect_libc_type)
    include(CheckCXXSourceCompiles)
    check_cxx_source_compiles("
# include <string>
# ifndef _GLIBCXX_USE_CXX11_ABI
#  error \"GlibCXX ABI is not defined\"
# endif

int main() {
  return 0;
}"
    OPENVINO_STDLIB_GNU)

    if(OPENVINO_STDLIB_GNU)
        set(OPENVINO_STDLIB "GNU" PARENT_SCOPE)
    else()
        set(OPENVINO_STDLIB "CPP" PARENT_SCOPE)
    endif()

    check_cxx_source_compiles("
# ifndef _GNU_SOURCE
#   define _GNU_SOURCE
#   include <features.h>
#   ifndef __USE_GNU
#     define CMAKE_OPENVINO_MUSL_LIBC
#   endif
#   undef _GNU_SOURCE /* don't contaminate other includes unnecessarily */
# else
#   include <features.h>
#   ifndef __USE_GNU
#     define CMAKE_OPENVINO_MUSL_LIBC
#   endif
# endif
# ifndef CMAKE_OPENVINO_MUSL_LIBC
#   error \"OpenVINO GNU LIBC\"
# endif

int main() {
  return 0;
}"
    OPENVINO_GLIBC_MUSL)

    if(OPENVINO_GLIBC_MUSL)
        set(OPENVINO_MUSL_LIBC ON PARENT_SCOPE)
    else()
        set(OPENVINO_GNU_LIBC ON PARENT_SCOPE)
    endif()
endfunction()

if(LINUX)
  ov_detect_libc_type()
endif()

function(ov_get_compiler_definition definition var)
    if(NOT LINUX)
        message(FATAL_ERROR "Internal error: 'ov_get_definition' must be used only on Linux")
    endif()

    get_directory_property(_user_defines COMPILE_DEFINITIONS)
    foreach(_user_define IN LISTS _user_defines)
        # older cmake versions keep -D at the beginning, trim it
        string(REPLACE "-D" "" _user_define "${_user_define}")
        list(APPEND _ov_user_flags "-D${_user_define}")
    endforeach()
    string(REPLACE " " ";" _user_cxx_flags "${CMAKE_CXX_FLAGS}")
    foreach(_user_flag IN LISTS _user_cxx_flags)
        list(APPEND _ov_user_flags ${_user_flag})
    endforeach()

    execute_process(COMMAND echo "#include <string>"
                    COMMAND "${CMAKE_CXX_COMPILER}" ${_ov_user_flags} -x c++ - -E -dM
                    COMMAND grep -E "^#define ${definition} "
                    OUTPUT_VARIABLE output_value
                    ERROR_VARIABLE error_message
                    RESULT_VARIABLE exit_code
                    OUTPUT_STRIP_TRAILING_WHITESPACE)

    if(NOT exit_code EQUAL 0)
        message(FATAL_ERROR "Failed to detect '${definition}' definition value: ${error_message}\n${output_value}")
    endif()

    if(output_value MATCHES "^#define ${definition} ([0-9]+)")
        set("${var}" "${CMAKE_MATCH_1}" PARENT_SCOPE)
    else()
        message(FATAL_ERROR "Internal error: failed to parse ${definition} from '${output_value}'")
    endif()
endfunction()

function(ov_libc_version)
    # cmake needs to look at glibc version only when we build for Linux on Linux
    if(LINUX)
        if(OPENVINO_GNU_LIBC)
            ov_get_compiler_definition("__GLIBC__" _ov_glibc_major)
            ov_get_compiler_definition("__GLIBC_MINOR__" _ov_glibc_minor)

            set(OV_LIBC_VERSION "${_ov_glibc_major}.${_ov_glibc_minor}" PARENT_SCOPE)
        elseif(OPENVINO_MUSL_LIBC)
            # TODO: implement proper detection
            set(OV_LIBC_VERSION "1.1" PARENT_SCOPE)
        else()
            message(FATAL_ERROR "Undefined libc type")
        endif()
    else()
        set(OV_LIBC_VERSION "0.0" PARENT_SCOPE)
    endif()
endfunction()

ov_libc_version()

#
# Detects default value for _GLIBCXX_USE_CXX11_ABI for current compiler
#
macro(ov_get_glibcxx_use_cxx11_abi)
    if(LINUX AND OPENVINO_STDLIB STREQUAL "GNU")
        ov_get_compiler_definition("_GLIBCXX_USE_CXX11_ABI" OV_GLIBCXX_USE_CXX11_ABI)
    endif()
endmacro()

ov_get_glibcxx_use_cxx11_abi()
