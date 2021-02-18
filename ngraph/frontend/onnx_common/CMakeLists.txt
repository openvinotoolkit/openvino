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

# onnx symbols should be available for onnx_importer and onnx_editor
# onnx_common should be the only library that links with onnx
# it guaranties that there is only one protobuf instance
if(MSVC)
        target_link_libraries(${TARGET_NAME} PRIVATE -WHOLEARCHIVE:$<TARGET_FILE:onnx>)
        target_link_libraries(${TARGET_NAME} PRIVATE -WHOLEARCHIVE:$<TARGET_FILE:onnx_proto>)
        target_link_libraries(${TARGET_NAME} PRIVATE -WHOLEARCHIVE:$<TARGET_FILE:${Protobuf_LIBRARIES}>)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "^(Apple)?Clang$")
        target_link_libraries(${TARGET_NAME} PRIVATE -Wl,-force_load,$<TARGET_FILE:onnx>)
        target_link_libraries(${TARGET_NAME} PRIVATE -Wl,-force_load,$<TARGET_FILE:onnx_proto>)
        target_link_libraries(${TARGET_NAME} PRIVATE -Wl,-force_load,$<TARGET_FILE:${Protobuf_LIBRARIES}>)
else()
        set(BEFORE_CMAKE_SHARED_LINKER_FLAGS ${CMAKE_SHARED_LINKER_FLAGS})
        string (REPLACE "-Wl,--exclude-libs,ALL" "" CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}")

        target_link_libraries(${TARGET_NAME} PRIVATE -Wl,--whole-archive
                                                onnx
                                                -Wl,--no-whole-archive)
        set(CMAKE_SHARED_LINKER_FLAGS, ${BEFORE_CMAKE_SHARED_LINKER_FLAGS})
endif()
add_dependencies(${TARGET_NAME} onnx onnx_proto ${Protobuf_LIBRARIES})

target_link_libraries(${TARGET_NAME} PRIVATE ngraph)

target_include_directories(${TARGET_NAME}
        SYSTEM PRIVATE
                $<TARGET_PROPERTY:onnx,INTERFACE_INCLUDE_DIRECTORIES>
                $<TARGET_PROPERTY:onnx_proto,INTERFACE_INCLUDE_DIRECTORIES>
                ${Protobuf_INCLUDE_DIRS}
                ${NGRAPH_INCLUDE_PATH})
target_compile_definitions(${TARGET_NAME}
        PRIVATE $<TARGET_PROPERTY:onnx,INTERFACE_COMPILE_DEFINITIONS>)

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
