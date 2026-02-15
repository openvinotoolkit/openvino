# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "openvino_onnx_common")

file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE PUBLIC_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${PUBLIC_HEADERS} ${PUBLIC_HEADERS})

# Create static library

add_library(${TARGET_NAME} STATIC ${LIBRARY_SRC} ${PUBLIC_HEADERS})

set_target_properties(${TARGET_NAME} PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

set(ONNX_COMMON_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(ONNX_COMMON_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_include_directories(${TARGET_NAME}
    PUBLIC $<BUILD_INTERFACE:${ONNX_COMMON_INCLUDE_DIR}>
           $<INSTALL_INTERFACE:${FRONTEND_INSTALL_INCLUDE}>
    PRIVATE ${ONNX_COMMON_SRC_DIR})

target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime openvino::util)

ov_link_system_libraries(${TARGET_NAME} PUBLIC onnx_proto onnx)

ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})

ov_install_static_lib(${TARGET_NAME} ${OV_CPACK_COMP_CORE})
