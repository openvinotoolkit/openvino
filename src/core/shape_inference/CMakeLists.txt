# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "ov_shape_inference")

file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE PUBLIC_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)

set(SHAPE_INFER_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${PUBLIC_HEADERS})

# Create static library
add_library(${TARGET_NAME} STATIC ${LIBRARY_SRC} ${PUBLIC_HEADERS})

target_include_directories(${TARGET_NAME} PUBLIC
    $<BUILD_INTERFACE:${SHAPE_INFER_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${OV_CORE_INCLUDE_PATH}>)

add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})

if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(${TARGET_NAME} PUBLIC OPENVINO_STATIC_LIBRARY)
endif()

ov_install_static_lib(${TARGET_NAME} ${OV_CPACK_COMP_CORE})

# developer package
openvino_developer_export_targets(COMPONENT core TARGETS ${TARGET_NAME})
