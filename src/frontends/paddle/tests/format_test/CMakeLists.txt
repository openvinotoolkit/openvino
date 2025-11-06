# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME paddle_format_test)

include(CTest)
include(GoogleTest)
find_package(yaml-cpp REQUIRED)

file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_executable(${TARGET_NAME} ${SOURCES})

# Also compile some helper sources from the paddle src directory so tests can
# use the JSON/YAML decoders directly when running unit tests.
target_sources(${TARGET_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/decoder_json.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/yaml_metadata.cpp)
target_sources(${TARGET_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/decoder.cpp)

target_link_libraries(${TARGET_NAME}
    PRIVATE
        gtest
        gtest_main
        openvino::frontend::paddle
        openvino::runtime::dev
        nlohmann_json::nlohmann_json
        yaml-cpp::yaml-cpp)

add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME}
    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

target_include_directories(${TARGET_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src)