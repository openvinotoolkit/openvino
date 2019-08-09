# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (TARGET_NAME "HeteroPlugin")

file(GLOB SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

add_library(${TARGET_NAME} SHARED ${SOURCES})

target_include_directories(${TARGET_NAME} PRIVATE
    "${IE_MAIN_SOURCE_DIR}/src/inference_engine"
)

target_compile_definitions(${TARGET_NAME} PRIVATE IMPLEMENT_INFERENCE_ENGINE_PLUGIN)

target_link_libraries(${TARGET_NAME} PRIVATE inference_engine)

set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_PDB_NAME ${TARGET_NAME})

add_cpplint_target(${TARGET_NAME}_cpplint FOR_TARGETS ${TARGET_NAME})
