# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "inference_engine_legacy")

set(PUBLIC_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
                              ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp)
file(GLOB_RECURSE PUBLIC_HEADERS ${PUBLIC_HEADERS_DIR}/*.hpp
                                 ${PUBLIC_HEADERS_DIR}/*.h)

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${PUBLIC_HEADERS})

if(ENABLE_V7_SERIALIZE)
    set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/src/cnn_network_impl.cpp"
        PROPERTIES COMPILE_DEFINITIONS ENABLE_V7_SERIALIZE)
endif()

# Create object library

file(TOUCH ${CMAKE_CURRENT_BINARY_DIR}/dummy.cpp)

add_library(${TARGET_NAME}_obj OBJECT
            ${LIBRARY_SRC}
            ${PUBLIC_HEADERS})

ie_faster_build(${TARGET_NAME}_obj
    PCH PRIVATE "src/precomp.hpp"
)

target_compile_definitions(${TARGET_NAME}_obj PRIVATE IMPLEMENT_INFERENCE_ENGINE_API)

target_include_directories(${TARGET_NAME}_obj PRIVATE
    ${PUBLIC_HEADERS_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${IE_MAIN_SOURCE_DIR}/src/inference_engine # For CNNNetworkNGraphImpl
    $<TARGET_PROPERTY:inference_engine_transformations,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:inference_engine_plugin_api,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:ngraph::ngraph,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:pugixml,INTERFACE_INCLUDE_DIRECTORIES>)

target_compile_definitions(${TARGET_NAME}_obj PRIVATE $<TARGET_PROPERTY:ngraph::ngraph,INTERFACE_COMPILE_DEFINITIONS>)

target_link_libraries(${TARGET_NAME}_obj PRIVATE openvino::itt)

add_cpplint_target(${TARGET_NAME}_obj_cpplint FOR_TARGETS ${TARGET_NAME}_obj)

# Create shared library

add_library(${TARGET_NAME} SHARED
    ${CMAKE_CURRENT_BINARY_DIR}/dummy.cpp
    $<TARGET_OBJECTS:${TARGET_NAME}_obj>)

ie_add_vs_version_file(NAME ${TARGET_NAME}
                       FILEDESCRIPTION "Inference Engine Legacy library")

target_link_libraries(${TARGET_NAME} PUBLIC inference_engine
                                     PRIVATE pugixml openvino::itt
                                             ${NGRAPH_LIBRARIES} inference_engine_transformations)

target_include_directories(${TARGET_NAME} INTERFACE ${PUBLIC_HEADERS_DIR})

ie_add_api_validator_post_build_step(TARGET ${TARGET_NAME})

# LTO

set_target_properties(${TARGET_NAME} ${TARGET_NAME}_obj
                      PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

# developer package

ie_developer_export_targets(${TARGET_NAME})

# install

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT core
        LIBRARY DESTINATION ${IE_CPACK_LIBRARY_PATH} COMPONENT core)
