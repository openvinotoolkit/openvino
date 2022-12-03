# Copyright (C) 2018-2022 Intel Corporation
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

macro(_ie_process_msvc_generator_platform flag_name)
  # if cmake -A <ARM|ARM64> is passed
  if(CMAKE_GENERATOR_PLATFORM STREQUAL "ARM64")
    set(AARCH64 ON)
  elseif(CMAKE_GENERATOR_PLATFORM STREQUAL "ARM")
    set(ARM ON)
  elseif(CMAKE_GENERATOR_PLATFORM STREQUAL "x64")
    set(X86_64 ON)
  elseif(CMAKE_GENERATOR_PLATFORM STREQUAL "Win32")
    set(X86 ON)
  else()
    set(${flag_name} ON)
  endif()
endmacro()

if(MSVC64 OR MINGW64)
  _ie_process_msvc_generator_platform(X86_64)
elseif(MINGW OR (MSVC AND NOT CMAKE_CROSSCOMPILING))
  _ie_process_msvc_generator_platform(X86)
elseif(CMAKE_OSX_ARCHITECTURES AND APPLE)
  if(CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
    set(AARCH64 ON)
  elseif(CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64")
    set(X86_64 ON)
  elseif(CMAKE_OSX_ARCHITECTURES MATCHES ".*x86_64.*" AND CMAKE_OSX_ARCHITECTURES MATCHES ".*arm64.*")
    set(UNIVERSAL2 ON)
  else()
    message(FATAL_ERROR "Unsupported value: CMAKE_OSX_ARCHITECTURES = ${CMAKE_OSX_ARCHITECTURES}")
  endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "amd64.*|x86_64.*|AMD64.*")
  set(X86_64 ON)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "i686.*|i386.*|x86.*|amd64.*|AMD64.*")
  set(X86 ON)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64.*|aarch64.*|AARCH64.*)")
  set(AARCH64 ON)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm.*|ARM.*)")
  set(ARM ON)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^riscv64$")
  set(RISCV64 ON)
endif()

if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "amd64.*|x86_64.*|AMD64.*")
  set(HOST_X86_64 ON)
elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "i686.*|i386.*|x86.*|amd64.*|AMD64.*")
  set(HOST_X86 ON)
elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^(arm64.*|aarch64.*|AARCH64.*)")
  set(HOST_AARCH64 ON)
elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^(arm.*|ARM.*)")
  set(HOST_ARM ON)
elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^riscv64$")
  set(HOST_RISCV64 ON)
endif()

if(UNIX AND NOT (APPLE OR ANDROID))
    set(LINUX ON)
endif()

if(ENV{OECORE_NATIVE_SYSROOT} AND AARCH64)
    set(YOCTO_AARCH64 ON)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "^(Apple)?Clang$")
    set(OV_COMPILER_IS_CLANG ON)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
        set(OV_COMPILER_IS_APPLECLANG ON)
    endif()
endif()

get_property(OV_GENERATOR_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)

function(ov_glibc_version)
    if(LINUX)
        execute_process(COMMAND ${CMAKE_C_COMPILER} -print-file-name=libc.so.6
                        OUTPUT_VARIABLE glibc_library
                        ERROR_VARIABLE error_message
                        RESULT_VARIABLE exit_code
                        OUTPUT_STRIP_TRAILING_WHITESPACE)

        if(NOT exit_code EQUAL 0)
            message(FATAL_ERROR "Failed to detect glibc version: ${error_message}")
        endif()

        get_filename_component(glibc_library ${glibc_library} REALPATH)
        GET_FILENAME_COMPONENT (GLIBC_VERSION ${glibc_library} NAME)
        string(REPLACE "libc-" "" GLIBC_VERSION ${GLIBC_VERSION})
        string(REPLACE ".so" "" GLIBC_VERSION ${GLIBC_VERSION})
        if(NOT GLIBC_VERSION MATCHES "^[0-9.]+$")
            message(FATAL_ERROR "Internal error: unknown glibc version - ${GLIBC_VERSION}")
        endif()

        set(OV_GLIBC_VERSION "${GLIBC_VERSION}" PARENT_SCOPE)
    else()
        set(OV_GLIBC_VERSION "0.0" PARENT_SCOPE)
    endif()
endfunction()

ov_glibc_version()
