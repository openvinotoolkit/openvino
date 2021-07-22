# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (TARGET_NAME "PluginHelper")

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

add_library(${TARGET_NAME} STATIC ${SOURCES})

target_include_directories(${TARGET_NAME}
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${TARGET_NAME}
        SYSTEM PUBLIC
        $<TARGET_PROPERTY:IE::inference_engine_plugin_api,INTERFACE_INCLUDE_DIRECTORIES>)

link_system_libraries(${TARGET_NAME}
        IE::inference_engine_plugin_api
        PUBLIC
        IE::inference_engine_legacy
        )
