# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
cmake_minimum_required(VERSION 3.13)
project(libxsmm C CXX)
set(LIBXSMM_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
add_compile_definitions(libxsmm PUBLIC LIBXSMM_DEFAULT_CONFIG)

file(GLOB _GLOB_XSMM_SRCS LIST_DIRECTORIES false CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c)
set(XSMM_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(libxsmm STATIC ${_GLOB_XSMM_SRCS})
set(LIBXSMM_LIBS libxsmm)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)


target_include_directories(libxsmm PUBLIC ${XSMM_INCLUDE_DIRS})
target_include_directories(libxsmm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_compile_definitions(libxsmm PUBLIC LIBXSMM_DEFAULT_CONFIG)
target_compile_definitions(libxsmm PRIVATE  __BLAS=0)

set_target_properties(libxsmm PROPERTIES FOLDER "libxsmm")
set_property(TARGET libxsmm PROPERTY POSITION_INDEPENDENT_CODE ON)
