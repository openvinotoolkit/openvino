# Copyright (C) 2018-2023 Intel Corporation
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

get_property(OV_GENERATOR_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)

function(ov_glibc_version)
    # cmake needs to look at glibc version only when we build for Linux on Linux
    if(LINUX)
        function(ov_get_definition definition var)
            execute_process(COMMAND echo "#include <errno.h>"
                            COMMAND "${CMAKE_CXX_COMPILER}" -xc - -E -dM
                            COMMAND grep -E "^#define ${definition} "
                            OUTPUT_VARIABLE glibc_version_component
                            ERROR_VARIABLE error_message
                            RESULT_VARIABLE exit_code
                            OUTPUT_STRIP_TRAILING_WHITESPACE)

            if(NOT exit_code EQUAL 0)
                message(FATAL_ERROR "Failed to detect glibc version: ${error_message}\n${glibc_version_component}")
            endif()

            if(glibc_version_component MATCHES "^#define ${definition} ([0-9]+)")
                set("${var}" "${CMAKE_MATCH_1}" PARENT_SCOPE)
            else()
                message(FATAL_ERROR "Internal error: failed to parse ${definition} from '${glibc_version_component}'")
            endif()
        endfunction()

        ov_get_definition("__GLIBC__" _ov_glibc_major)
        ov_get_definition("__GLIBC_MINOR__" _ov_glibc_minor)

        set(OV_GLIBC_VERSION "${_ov_glibc_major}.${_ov_glibc_minor}" PARENT_SCOPE)
    else()
        set(OV_GLIBC_VERSION "0.0" PARENT_SCOPE)
    endif()
endfunction()

ov_glibc_version()
