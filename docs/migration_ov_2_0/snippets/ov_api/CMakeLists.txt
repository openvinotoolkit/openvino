# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(CMAKE_CXX_STANDARD 11)

set(TARGET_NAME "ov_api_2_0_snippets")

find_package(ngraph REQUIRED)
find_package(InferenceEngine REQUIRED)

file(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_library(${TARGET_NAME} STATIC ${SOURCES})

target_link_libraries(${TARGET_NAME} PRIVATE inference_engine ngraph)
