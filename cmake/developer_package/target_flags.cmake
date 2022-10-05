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
endif()

if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "amd64.*|x86_64.*|AMD64.*")
  set(HOST_X86_64 ON)
elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "i686.*|i386.*|x86.*|amd64.*|AMD64.*")
  set(HOST_X86 ON)
elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^(arm64.*|aarch64.*|AARCH64.*)")
  set(HOST_AARCH64 ON)
elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^(arm.*|ARM.*)")
  set(HOST_ARM ON)
endif()

if(UNIX AND NOT APPLE)
    set(LINUX ON)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "^(Apple)?Clang$")
    set(OV_COMPILER_IS_CLANG ON)
endif()

if(CMAKE_CXX_COMPILER MATCHES ".*conda.*")
    set(OV_COMPILER_IS_CONDA ON)
endif()

get_property(OV_GENERATOR_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
