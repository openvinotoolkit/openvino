# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "inference_engine_ir_v7_reader")

file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
                              ${CMAKE_CURRENT_SOURCE_DIR}/*.h
                              ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
list(APPEND LIBRARY_SRC ${IE_MAIN_SOURCE_DIR}/src/readers/ir_reader/ie_ir_reader.cpp)

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj

source_group("src" FILES ${LIBRARY_SRC})

# Create module library

add_library(${TARGET_NAME} MODULE ${LIBRARY_SRC})

ie_faster_build(${TARGET_NAME}
    UNITY
)

ie_add_vs_version_file(NAME ${TARGET_NAME}
                       FILEDESCRIPTION "Inference Engine IR v7 reader plugin")

target_compile_definitions(${TARGET_NAME} PRIVATE IMPLEMENT_INFERENCE_ENGINE_PLUGIN)

target_include_directories(${TARGET_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/"
                                                  "${IE_MAIN_SOURCE_DIR}/src/inference_engine"
                                                  "${IE_MAIN_SOURCE_DIR}/src/readers/ir_reader")

target_link_libraries(${TARGET_NAME} PRIVATE inference_engine_reader_api inference_engine_plugin_api
                                             inference_engine_legacy inference_engine pugixml openvino::itt)

ie_add_api_validator_post_build_step(TARGET ${TARGET_NAME})

ie_add_manifest(TARGET_NAME ${TARGET_NAME} DEPENDENCIES inference_engine_legacy)

set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

if(WIN32)
    set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_PDB_NAME ${TARGET_NAME})
endif()

# code style

add_cpplint_target(${TARGET_NAME}_cpplint FOR_TARGETS ${TARGET_NAME})
