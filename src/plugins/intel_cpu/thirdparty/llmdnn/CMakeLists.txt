# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.13)

project(root)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(LLMDNN_BUILD_TESTS "Build with tests" ON)

message(INFO "--------------------------------")
message(STATUS "Build with tests: ${LLMDNN_BUILD_TESTS}")
message(INFO "--------------------------------")

set(CMAKE_CXX_STANDARD 11)
if(MSVC)
  # TODO: validate
  if(MSVC_VERSION VERSION_LESS 1928)
    message(FATAL_ERROR "Insufficient msvc compiler version, current ${MSVC_VERSION}, minimum 1928.")
  endif()
  # Force to always compile with W4
  if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
    string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
  endif()
elseif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "11.3")
    message(FATAL_ERROR "Insufficient gcc compiler version, current ${CMAKE_CXX_COMPILER_VERSION}, minimum 11.3.")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=sapphirerapids -flax-vector-conversions")
elseif(OV_COMPILER_IS_CLANG)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "12")
    message(FATAL_ERROR "Insufficient clang compiler version, current ${CMAKE_CXX_COMPILER_VERSION}, minimum 12.")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=sapphirerapids -flax-vector-conversions")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "2023.0")
    message(FATAL_ERROR "Insufficient intel compiler version, current ${CMAKE_CXX_COMPILER_VERSION}, minimum 2023.0.")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=sapphirerapids")
endif()

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src/)

if (NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()
add_subdirectory(src)
if (LLMDNN_BUILD_TESTS)
  add_subdirectory(tests)
endif()
