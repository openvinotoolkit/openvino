# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if (NOT ENABLE_PROXY)
    return()
endif()

set(TARGET_NAME "openvino_proxy_plugin_obj")

set(PUBLIC_HEADERS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dev_api)
file(GLOB LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB LIBRARY_HEADERS ${PUBLIC_HEADERS_DIR}/*.hpp ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp)

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${LIBRARY_HEADERS})

add_library(${TARGET_NAME} OBJECT ${LIBRARY_SRC} ${LIBRARY_HEADERS})

if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(${TARGET_NAME} PUBLIC OPENVINO_STATIC_LIBRARY)
endif()

target_compile_definitions(${TARGET_NAME} PRIVATE
        IMPLEMENT_OPENVINO_RUNTIME_API
        CI_BUILD_NUMBER=\"${CI_BUILD_NUMBER}\")

# LTO
set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

target_include_directories(${TARGET_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${PUBLIC_HEADERS_DIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        $<BUILD_INTERFACE:$<TARGET_PROPERTY:openvino::runtime::dev,INTERFACE_INCLUDE_DIRECTORIES>>)

ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})

ov_mark_target_as_cc(${TARGET_NAME})

ov_ncc_naming_style(FOR_TARGET ${TARGET_NAME}
                    SOURCE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/dev_api"
                                       "${CMAKE_CURRENT_SOURCE_DIR}/src")

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()
