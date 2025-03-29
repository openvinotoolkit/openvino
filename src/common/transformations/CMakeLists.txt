# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "openvino_transformations")

set(PUBLIC_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE PUBLIC_HEADERS ${PUBLIC_HEADERS_DIR}/*.hpp)

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${PUBLIC_HEADERS})

# Create library
if((CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG) AND CMAKE_CXX_STANDARD GREATER_EQUAL 20)
    set(CMAKE_CXX_FLAGS "-Wno-error=deprecated ${CMAKE_CXX_FLAGS}")
endif()

add_library(${TARGET_NAME}_obj OBJECT ${LIBRARY_SRC} ${PUBLIC_HEADERS})
target_compile_definitions(${TARGET_NAME}_obj PRIVATE IMPLEMENT_OPENVINO_API)

ov_build_target_faster(${TARGET_NAME}_obj
    UNITY
    PCH PRIVATE "src/precomp.hpp"
)

target_link_libraries(${TARGET_NAME}_obj PRIVATE openvino::reference openvino::itt openvino::core::dev openvino::shape_inference)

target_include_directories(${TARGET_NAME}_obj PRIVATE "${PUBLIC_HEADERS_DIR}"
                                                      "${CMAKE_CURRENT_SOURCE_DIR}/src")

ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME}_obj)

ov_mark_target_as_cc(${TARGET_NAME}_obj)

if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(${TARGET_NAME}_obj PUBLIC OPENVINO_STATIC_LIBRARY)
endif()

# INTERFACE library for BW compatibility

add_library(${TARGET_NAME} INTERFACE)

target_include_directories(${TARGET_NAME} INTERFACE
    $<BUILD_INTERFACE:${PUBLIC_HEADERS_DIR}>)

target_link_libraries(${TARGET_NAME} INTERFACE openvino::runtime)

# even the Transformations library is supposed to be Plugin API
# we still have some code compiled as transformations, but headers are
# part of openvino core API
# so, we need to mark this library as important for ABI free
ov_abi_free_target(${TARGET_NAME}_obj)

# LTO

set_target_properties(${TARGET_NAME}_obj PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()
