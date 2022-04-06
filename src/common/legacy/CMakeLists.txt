# Copyright (C) 2018-2022 Intel Corporation
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

add_library(${TARGET_NAME}_obj OBJECT EXCLUDE_FROM_ALL
            ${LIBRARY_SRC}
            ${PUBLIC_HEADERS})

ie_faster_build(${TARGET_NAME}_obj
    PCH PRIVATE "src/precomp.hpp")

target_include_directories(${TARGET_NAME}_obj PRIVATE
    ${PUBLIC_HEADERS_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    $<TARGET_PROPERTY:inference_engine_obj,SOURCE_DIR>/src # For CNNNetworkNGraphImpl
    $<TARGET_PROPERTY:openvino::runtime::dev,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:pugixml::static,INTERFACE_INCLUDE_DIRECTORIES>)

target_compile_definitions(${TARGET_NAME}_obj PRIVATE $<TARGET_PROPERTY:ngraph,INTERFACE_COMPILE_DEFINITIONS>)

target_link_libraries(${TARGET_NAME}_obj PRIVATE openvino::itt)

add_cpplint_target(${TARGET_NAME}_obj_cpplint FOR_TARGETS ${TARGET_NAME}_obj)

# Create static library

add_library(${TARGET_NAME} STATIC EXCLUDE_FROM_ALL
    ${CMAKE_CURRENT_BINARY_DIR}/dummy.cpp
    $<TARGET_OBJECTS:${TARGET_NAME}_obj>)

target_link_libraries(${TARGET_NAME} PUBLIC openvino::runtime
                                     PRIVATE pugixml::static openvino::itt openvino::runtime::dev)

target_include_directories(${TARGET_NAME} INTERFACE
    $<BUILD_INTERFACE:${PUBLIC_HEADERS_DIR}>)

# LTO

set_target_properties(${TARGET_NAME} ${TARGET_NAME}_obj
                      PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

# developer package

openvino_developer_export_targets(COMPONENT core_legacy TARGETS ${TARGET_NAME})

# install

ov_install_static_lib(${TARGET_NAME} core OPTIONAL)
