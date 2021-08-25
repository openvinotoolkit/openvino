# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

project(cnpy)

set(TARGET_NAME "cnpy")
add_library(cnpy STATIC cnpy.cpp)

if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(${TARGET_NAME} PUBLIC -Wno-unused-variable
                                          PRIVATE -Wno-all)
endif()

target_link_libraries(${TARGET_NAME} PUBLIC zlib)
target_include_directories(${TARGET_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

set_target_properties(cnpy PROPERTIES FOLDER thirdparty)
