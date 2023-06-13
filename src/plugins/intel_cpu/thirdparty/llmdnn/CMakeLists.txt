# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.13)

project(root)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(BUILD_TESTS "Build with tests" ON)
option(BUILD_PYTHON_TESTS "Build with tests need python extension" OFF)

message(INFO "--------------------------------")
message(STATUS "Build with tests: ${BUILD_TESTS}")
message(INFO "--------------------------------")

set(CMAKE_CXX_STANDARD 11)
if(MSVC)
  # TODO
  message(FATAL_ERROR "Not support yet. Use intel compiler 2023.0+.")
  # Force to always compile with W4
  if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
    string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
  endif()
elseif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
  # TODO
  message(FATAL_ERROR "Not support yet. Use intel compiler 2023.0+.")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "12.0")
    message(FATAL_ERROR "Insufficient gcc compiler version, current ${CMAKE_CXX_COMPILER_VERSION}, minimum 12.0.")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=sapphirerapids")
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
if (BUILD_TESTS)
  add_subdirectory(tests)
endif()
