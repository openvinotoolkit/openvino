# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "onnx_editor")

file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE LIBRARY_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp)
file(GLOB_RECURSE LIBRARY_PUBLIC_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${LIBRARY_HEADERS})
source_group("public include" FILES ${LIBRARY_PUBLIC_HEADERS})

# Create static library

add_library(${TARGET_NAME} STATIC ${LIBRARY_SRC} ${LIBRARY_HEADERS} ${LIBRARY_PUBLIC_HEADERS})
add_library(ngraph::onnx_editor ALIAS ${TARGET_NAME})

# TODO Add handling ie_faster_build

target_link_libraries(${TARGET_NAME} PRIVATE onnx_common onnx_importer
                                     PUBLIC ngraph)

set(ONNX_EDITOR_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(ONNX_EDITOR_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(ONNX_EDITOR_INSTALL_INCLUDE "${NGRAPH_INSTALL_INCLUDE}/ngraph/frontend")

target_include_directories(${TARGET_NAME} PUBLIC $<BUILD_INTERFACE:${ONNX_EDITOR_INCLUDE_DIR}>
                                                 $<INSTALL_INTERFACE:${ONNX_EDITOR_INSTALL_INCLUDE}>)

target_include_directories(${TARGET_NAME} PRIVATE ${ONNX_EDITOR_SRC_DIR} ${NGRAPH_INCLUDE_PATH} ${Protobuf_INCLUDE_DIRS})

if(NGRAPH_USE_PROTOBUF_LITE)
    target_compile_definitions(${TARGET_NAME} PRIVATE NGRAPH_USE_PROTOBUF_LITE)
endif()

add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})
