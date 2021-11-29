# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "inference_engine_ir_reader")

file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
                              ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj

source_group("src" FILES ${LIBRARY_SRC})

# Create module library

add_library(${TARGET_NAME} MODULE ${LIBRARY_SRC})

ie_faster_build(${TARGET_NAME}
    UNITY
)

ie_add_vs_version_file(NAME ${TARGET_NAME}
                       FILEDESCRIPTION "Inference Engine IR reader plugin")

target_compile_definitions(${TARGET_NAME} PRIVATE IMPLEMENT_INFERENCE_ENGINE_PLUGIN
                                                  IR_READER_V10)

target_include_directories(${TARGET_NAME} PRIVATE "${IE_MAIN_SOURCE_DIR}/src/inference_engine" # for CNNNetworkNgraphImpl
                                                  "${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(${TARGET_NAME} PRIVATE ${NGRAPH_LIBRARIES}
                                             inference_engine_reader_api
                                             inference_engine_plugin_api
                                             inference_engine
                                             inference_engine_transformations
                                             pugixml
                                             openvino::itt)

ie_add_api_validator_post_build_step(TARGET ${TARGET_NAME})

set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

# code style

add_cpplint_target(${TARGET_NAME}_cpplint FOR_TARGETS ${TARGET_NAME})

# install

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT core
        LIBRARY DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT core)
