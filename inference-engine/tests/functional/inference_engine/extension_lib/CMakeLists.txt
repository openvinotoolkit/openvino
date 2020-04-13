# Copyright (C) 2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "extension_tests")

file(GLOB_RECURSE SRC src/*.cpp)

add_definitions(-DIMPLEMENT_INFERENCE_EXTENSION_API)

add_library(${TARGET_NAME} SHARED ${SRC})

target_link_libraries(${TARGET_NAME} PRIVATE inference_engine ${NGRAPH_LIBRARIES})
target_include_directories(${TARGET_NAME} PRIVATE
    ${IE_MAIN_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_dependencies(${TARGET_NAME} inference_engine ngraph)
