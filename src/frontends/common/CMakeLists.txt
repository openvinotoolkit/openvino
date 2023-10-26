# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "frontend_common")

file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE LIBRARY_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp)
file(GLOB_RECURSE LIBRARY_PUBLIC_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)

set(FRONTEND_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${LIBRARY_HEADERS})
source_group("public include" FILES ${LIBRARY_PUBLIC_HEADERS})

# create library

add_library(${TARGET_NAME}_obj OBJECT ${LIBRARY_SRC} ${LIBRARY_HEADERS} ${LIBRARY_PUBLIC_HEADERS})

target_include_directories(${TARGET_NAME}_obj
    PUBLIC
        $<BUILD_INTERFACE:${FRONTEND_INCLUDE_DIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        $<TARGET_PROPERTY:ngraph,INTERFACE_INCLUDE_DIRECTORIES>
        # for ov_frontends.hpp in static build
        ${CMAKE_CURRENT_BINARY_DIR}/src)

target_compile_definitions(${TARGET_NAME}_obj PRIVATE IMPLEMENT_OPENVINO_API)

if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(${TARGET_NAME}_obj PUBLIC OPENVINO_STATIC_LIBRARY)
endif()

target_link_libraries(${TARGET_NAME}_obj PRIVATE openvino::util openvino::core::dev)

# TODO: fix lto
set_target_properties(${TARGET_NAME}_obj PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION_RELEASE OFF)

set(FRONTEND_LIB_SUFFIX "${FRONTEND_NAME_SUFFIX}${OV_BUILD_POSTFIX}")
if(APPLE)
    set(FRONTEND_LIB_SUFFIX "${FRONTEND_LIB_SUFFIX}${OpenVINO_VERSION_SUFFIX}${CMAKE_SHARED_LIBRARY_SUFFIX}")
else()
    set(FRONTEND_LIB_SUFFIX "${FRONTEND_LIB_SUFFIX}${CMAKE_SHARED_LIBRARY_SUFFIX}${OpenVINO_VERSION_SUFFIX}")
endif()

target_compile_definitions(${TARGET_NAME}_obj PRIVATE
    FRONTEND_LIB_PREFIX="${CMAKE_SHARED_LIBRARY_PREFIX}${FRONTEND_NAME_PREFIX}"
    FRONTEND_LIB_SUFFIX="${FRONTEND_LIB_SUFFIX}")

ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME}_obj)

# FEM is public API => need to mark this library as important for ABI free
ov_abi_free_target(${TARGET_NAME}_obj)

ov_ncc_naming_style(FOR_TARGET ${TARGET_NAME}_obj
                    SOURCE_DIRECTORIES "${FRONTEND_INCLUDE_DIR}"
                    ADDITIONAL_INCLUDE_DIRECTORIES
                        $<TARGET_PROPERTY:ngraph,INTERFACE_INCLUDE_DIRECTORIES>)

# INTERFACE library for BW compatibility

add_library(${TARGET_NAME} INTERFACE)
target_link_libraries(${TARGET_NAME} INTERFACE openvino::runtime)
target_include_directories(${TARGET_NAME} INTERFACE $<BUILD_INTERFACE:${FRONTEND_INCLUDE_DIR}>
    $<BUILD_INTERFACE:$<TARGET_PROPERTY:ngraph,INTERFACE_INCLUDE_DIRECTORIES>>)

add_library(ngraph::${TARGET_NAME} ALIAS ${TARGET_NAME})
add_library(openvino::frontend::common ALIAS ${TARGET_NAME})
add_library(${TARGET_NAME}::static ALIAS ${TARGET_NAME})

# Installation rules header files

install(DIRECTORY ${FRONTEND_INCLUDE_DIR}/openvino
        DESTINATION ${FRONTEND_INSTALL_INCLUDE}
        COMPONENT ${OV_CPACK_COMP_CORE_DEV})

# Shutdown protobuf library
if(Protobuf_IN_FRONTEND AND BUILD_SHARED_LIBS)
    add_subdirectory(shutdown_protobuf)
endif()
