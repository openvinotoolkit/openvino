# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_thread_local)

# Search OpenVINO Runtime installed
find_package(PkgConfig QUIET)
# TODO: fix cross-compilation later
if(PkgConfig_FOUND AND NOT CMAKE_CROSSCOMPILING AND CMAKE_BUILD_TYPE STREQUAL "Release")
    pkg_search_module(openvino REQUIRED
                      IMPORTED_TARGET
                      openvino)
    set(ov_link_libraries PkgConfig::openvino)
else()
    find_package(OpenVINO REQUIRED COMPONENTS Runtime)
    set(ov_link_libraries openvino::runtime)
endif()

file (GLOB_RECURSE SRC *.cpp)
file (GLOB_RECURSE HDR *.h)

add_library(${TARGET_NAME} SHARED ${SRC} ${HDR})

target_include_directories(${TARGET_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

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
target_link_libraries(${TARGET_NAME} PUBLIC tests_shared_lib)

ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})
