# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (TARGET_NAME "inference_engine_lp_transformations")

set(PUBLIC_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE PUBLIC_HEADERS ${PUBLIC_HEADERS_DIR}/low_precision/*.hpp)

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${PUBLIC_HEADERS})

# Create library

add_library(${TARGET_NAME}_obj OBJECT
            ${LIBRARY_SRC}
            ${PUBLIC_HEADERS})

target_compile_definitions(${TARGET_NAME}_obj PRIVATE IMPLEMENT_OPENVINO_API)

ie_faster_build(${TARGET_NAME}_obj UNITY)

target_link_libraries(${TARGET_NAME}_obj PRIVATE openvino::itt)

target_include_directories(${TARGET_NAME}_obj PRIVATE $<BUILD_INTERFACE:${PUBLIC_HEADERS_DIR}>
    $<BUILD_INTERFACE:$<TARGET_PROPERTY:ngraph,INTERFACE_INCLUDE_DIRECTORIES>>
    $<BUILD_INTERFACE:$<TARGET_PROPERTY:inference_engine_transformations,INTERFACE_INCLUDE_DIRECTORIES>>)

add_cpplint_target(${TARGET_NAME}_cpplint FOR_TARGETS ${TARGET_NAME}_obj)

if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(${TARGET_NAME}_obj PUBLIC OPENVINO_STATIC_LIBRARY)
endif()

# INTERFACE library for BW compatibility

add_library(${TARGET_NAME} INTERFACE)
target_link_libraries(${TARGET_NAME} INTERFACE openvino::runtime)

target_include_directories(${TARGET_NAME} INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:$<TARGET_PROPERTY:ngraph,INTERFACE_INCLUDE_DIRECTORIES>>
    $<BUILD_INTERFACE:$<TARGET_PROPERTY:inference_engine_transformations,INTERFACE_INCLUDE_DIRECTORIES>>)

# LTO

set_target_properties(${TARGET_NAME}_obj PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

# developer package

openvino_developer_export_targets(COMPONENT core_legacy TARGETS ${TARGET_NAME})
