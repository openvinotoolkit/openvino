# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

project(cnpy)

set(TARGET_NAME "openvino_cnpy")

add_library(${TARGET_NAME} STATIC cnpy.cpp)
add_library(openvino::cnpy ALIAS ${TARGET_NAME})

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "^(Apple)?Clang$")
    target_compile_options(${TARGET_NAME} PUBLIC -Wno-unused-variable
                                          PRIVATE -Wno-all)
endif()

target_link_libraries(${TARGET_NAME} PUBLIC openvino::zlib)
target_include_directories(${TARGET_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

set_target_properties(${TARGET_NAME} PROPERTIES FOLDER thirdparty)
