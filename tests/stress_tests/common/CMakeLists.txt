# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME StressTestsCommon)

file (GLOB_RECURSE SRC *.cpp)
file (GLOB_RECURSE HDR *.h)

add_library(${TARGET_NAME} STATIC ${SRC} ${HDR})

target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(${TARGET_NAME}
    PUBLIC
        IE::gtest
        IE::pugixml
        ${InferenceEngine_LIBRARIES}
    PRIVATE
        IE::gtest_main)
