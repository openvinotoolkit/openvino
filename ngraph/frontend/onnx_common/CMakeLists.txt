# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "onnx_common")

file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE PUBLIC_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${PUBLIC_HEADERS} ${PUBLIC_HEADERS})

# Create shared library

add_library(${TARGET_NAME} SHARED ${LIBRARY_SRC})

if(COMMAND ie_add_vs_version_file)
        ie_add_vs_version_file(NAME ${TARGET_NAME}
                        FILEDESCRIPTION "nGraph ONNX Importer common part")
endif()

# TODO Add handling ie_faster_build

set(ONNX_COMMON_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(ONNX_COMMON_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(ONNX_COMMON_INSTALL_INCLUDE "${NGRAPH_INSTALL_INCLUDE}/ngraph/frontend")

target_include_directories(${TARGET_NAME} SYSTEM PUBLIC $<BUILD_INTERFACE:${ONNX_COMMON_INCLUDE_DIR}>
                                                       $<INSTALL_INTERFACE:${ONNX_COMMON_INSTALL_INCLUDE}>)
target_include_directories(${TARGET_NAME} PRIVATE ${ONNX_COMMON_SRC_DIR})

target_link_libraries(${TARGET_NAME} PRIVATE onnx onnx_proto ${Protobuf_LIBRARIES} ngraph)

target_include_directories(${TARGET_NAME}
        SYSTEM PRIVATE
                $<TARGET_PROPERTY:onnx,INTERFACE_INCLUDE_DIRECTORIES>
                $<TARGET_PROPERTY:onnx_proto,INTERFACE_INCLUDE_DIRECTORIES>
                ${Protobuf_INCLUDE_DIRS}
                ${NGRAPH_INCLUDE_PATH})

if(NGRAPH_USE_PROTOBUF_LITE)
        target_compile_definitions(${TARGET_NAME} PRIVATE NGRAPH_USE_PROTOBUF_LITE)
endif()


# install

install(TARGETS ${TARGET_NAME} EXPORT ngraphTargets
        RUNTIME DESTINATION ${NGRAPH_INSTALL_LIB} COMPONENT ngraph
        ARCHIVE DESTINATION ${NGRAPH_INSTALL_LIB} COMPONENT ngraph
        LIBRARY DESTINATION ${NGRAPH_INSTALL_LIB} COMPONENT ngraph)

if (NGRAPH_EXPORT_TARGETS_ENABLE)
        export(TARGETS ${TARGET_NAME} NAMESPACE ngraph:: APPEND FILE "${NGRAPH_TARGETS_FILE}")
endif()
