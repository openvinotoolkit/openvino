# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME onnx_ngraph_frontend)

set(ONNX_OPSET_VERSION 13 CACHE INTERNAL "Supported version of ONNX operator set")
set(ONNX_FRONTEND_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE LIBRARY_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp)
file(GLOB_RECURSE LIBRARY_PUBLIC_HEADERS ${ONNX_FRONTEND_INCLUDE_DIR}/*.hpp)

# Remove disabled ops
list(REMOVE_ITEM LIBRARY_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/conv_integer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/quant_conv.cpp
    )
list(REMOVE_ITEM LIBRARY_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/conv_integer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op/quant_conv.hpp
    )

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${LIBRARY_HEADERS})
source_group("public include" FILES ${LIBRARY_PUBLIC_HEADERS})

# Create shared library
add_library(${TARGET_NAME} SHARED ${LIBRARY_SRC} ${LIBRARY_HEADERS} ${LIBRARY_PUBLIC_HEADERS})
add_library(openvino::frontend::onnx ALIAS ${TARGET_NAME})

add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})

ov_ncc_naming_style(FOR_TARGET ${TARGET_NAME}
                   INCLUDE_DIRECTORY "${ONNX_FRONTEND_INCLUDE_DIR}"
                   DEFINITIONS
                       $<TARGET_PROPERTY:onnx,INTERFACE_COMPILE_DEFINITIONS>
                   ADDITIONAL_INCLUDE_DIRECTORIES
                       $<TARGET_PROPERTY:frontend_manager::static,INTERFACE_INCLUDE_DIRECTORIES>)

if(COMMAND ie_add_vs_version_file)
    ie_add_vs_version_file(NAME ${TARGET_NAME}
                           FILEDESCRIPTION "nGraph ONNX frontend library")
endif()

target_link_libraries(${TARGET_NAME} PUBLIC ngraph PRIVATE frontend_manager ngraph::builder onnx_common inference_engine_transformations)

target_include_directories(${TARGET_NAME} PUBLIC $<BUILD_INTERFACE:${ONNX_FRONTEND_INCLUDE_DIR}>
                                                $<INSTALL_INTERFACE:${FRONTEND_INSTALL_INCLUDE}>)
target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_compile_definitions(${TARGET_NAME} PRIVATE ONNX_OPSET_VERSION=${ONNX_OPSET_VERSION})
if(NGRAPH_USE_PROTOBUF_LITE)
    target_compile_definitions(${TARGET_NAME} PRIVATE NGRAPH_USE_PROTOBUF_LITE)
endif()

set_target_properties(${TARGET_NAME} PROPERTIES EXPORT_NAME frontend::onnx)
install(TARGETS ${TARGET_NAME} EXPORT OpenVINOTargets
        RUNTIME DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT ngraph
        ARCHIVE DESTINATION ${IE_CPACK_ARCHIVE_PATH} COMPONENT ngraph
        LIBRARY DESTINATION ${IE_CPACK_LIBRARY_PATH} COMPONENT ngraph)

install(DIRECTORY ${ONNX_FRONTEND_INCLUDE_DIR}/onnx_frontend
                  ${ONNX_FRONTEND_INCLUDE_DIR}/onnx_import
        DESTINATION ${FRONTEND_INSTALL_INCLUDE}
        COMPONENT ngraph_dev
        FILES_MATCHING PATTERN "*.hpp")

export(TARGETS ${TARGET_NAME} NAMESPACE openvino::
       APPEND FILE "${CMAKE_BINARY_DIR}/OpenVINOTargets.cmake")
