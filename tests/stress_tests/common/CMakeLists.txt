# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME StressTestsCommon)

file (GLOB_RECURSE SRC *.cpp)
file (GLOB_RECURSE HDR *.h)

add_library(${TARGET_NAME} STATIC ${SRC} ${HDR})

target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(${OpenVINO_SOURCE_DIR}/thirdparty/gflags
                 ${CMAKE_CURRENT_BINARY_DIR}/gflags_build
                 EXCLUDE_FROM_ALL)
add_subdirectory(${OpenVINO_SOURCE_DIR}/thirdparty/gtest
                 ${CMAKE_CURRENT_BINARY_DIR}/gtest_build
                 EXCLUDE_FROM_ALL)
add_subdirectory(${OpenVINO_SOURCE_DIR}/thirdparty/pugixml
                 ${CMAKE_CURRENT_BINARY_DIR}/pugixml_build
                 EXCLUDE_FROM_ALL)

add_subdirectory("${OpenVINO_SOURCE_DIR}/tests/lib" tests_shared_lib)
target_link_libraries(${TARGET_NAME}
    PUBLIC
        gtest
        pugixml
        gflags
        tests_shared_lib
    PRIVATE
        gtest_main)
