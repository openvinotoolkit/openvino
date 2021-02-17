# Copyright (C) 2021 Intel Corporation
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

# Create shared library

add_library(${TARGET_NAME} SHARED ${LIBRARY_SRC} ${LIBRARY_HEADERS} ${LIBRARY_PUBLIC_HEADERS})

if(COMMAND ie_add_vs_version_file)
        ie_add_vs_version_file(NAME ${TARGET_NAME}
                        FILEDESCRIPTION "nGraph ONNX Importer editor")
endif()

# TODO Add handling ie_faster_build

target_link_libraries(${TARGET_NAME} PRIVATE onnx_common PUBLIC ngraph)

set(ONNX_EDITOR_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(ONNX_EDITOR_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(ONNX_EDITOR_INSTALL_INCLUDE "${NGRAPH_INSTALL_INCLUDE}/ngraph/frontend")

target_include_directories(${TARGET_NAME} SYSTEM PUBLIC $<BUILD_INTERFACE:${ONNX_EDITOR_INCLUDE_DIR}>
                                                       $<INSTALL_INTERFACE:${ONNX_EDITOR_INSTALL_INCLUDE}>)
target_include_directories(${TARGET_NAME} PRIVATE ${ONNX_EDITOR_SRC_DIR})
target_include_directories(${TARGET_NAME} SYSTEM PRIVATE $<TARGET_PROPERTY:onnx_common,INTERFACE_INCLUDE_DIRECTORIES>)
target_include_directories(${TARGET_NAME} SYSTEM PRIVATE
                $<TARGET_PROPERTY:onnx,INTERFACE_INCLUDE_DIRECTORIES>
                $<TARGET_PROPERTY:onnx_proto,INTERFACE_INCLUDE_DIRECTORIES>
                ${NGRAPH_INCLUDE_PATH})
target_compile_definitions(${TARGET_NAME}
        PRIVATE $<TARGET_PROPERTY:onnx,INTERFACE_COMPILE_DEFINITIONS>)

# onnx-related directories
target_include_directories(${TARGET_NAME} 
SYSTEM PRIVATE 
        $<TARGET_PROPERTY:onnx_common,INTERFACE_INCLUDE_DIRECTORIES>)

target_include_directories(${TARGET_NAME} SYSTEM PRIVATE ${NGRAPH_INCLUDE_PATH})

if(NGRAPH_USE_PROTOBUF_LITE)
        target_compile_definitions(${TARGET_NAME} PRIVATE NGRAPH_USE_PROTOBUF_LITE)
endif()


install(TARGETS ${TARGET_NAME} EXPORT ngraphTargets
        RUNTIME DESTINATION ${NGRAPH_INSTALL_LIB} COMPONENT ngraph
        ARCHIVE DESTINATION ${NGRAPH_INSTALL_LIB} COMPONENT ngraph
        LIBRARY DESTINATION ${NGRAPH_INSTALL_LIB} COMPONENT ngraph)

install(DIRECTORY ${ONNX_EDITOR_INCLUDE_DIR}/onnx_editor
        DESTINATION ${ONNX_EDITOR_INSTALL_INCLUDE}
        COMPONENT ngraph
        FILES_MATCHING
            PATTERN "*.hpp"
            PATTERN "*.h"
    )

if (NGRAPH_EXPORT_TARGETS_ENABLE)
    export(TARGETS ${TARGET_NAME} NAMESPACE ngraph:: APPEND FILE "${NGRAPH_TARGETS_FILE}")
endif()
