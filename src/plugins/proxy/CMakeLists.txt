# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "openvino_proxy_plugin_obj")

set(PUBLIC_HEADERS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
file(GLOB LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB LIBRARY_HEADERS ${PUBLIC_HEADERS_DIR}/*.hpp ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp)

addVersionDefines(${CMAKE_CURRENT_SOURCE_DIR}/src/plugin.cpp CI_BUILD_NUMBER)

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${LIBRARY_HEADERS})

add_library(${TARGET_NAME} OBJECT ${LIBRARY_SRC} ${LIBRARY_HEADERS})
target_compile_definitions(${TARGET_NAME} PRIVATE IMPLEMENT_OPENVINO_API)

add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})

ie_mark_target_as_cc(${TARGET_NAME})

# LTO
set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

target_include_directories(${TARGET_NAME} PUBLIC $<BUILD_INTERFACE:${PUBLIC_HEADERS_DIR}>
                                          PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
                                                  $<BUILD_INTERFACE:$<TARGET_PROPERTY:inference_engine_plugin_api,INTERFACE_INCLUDE_DIRECTORIES>>)

add_subdirectory(tests)
