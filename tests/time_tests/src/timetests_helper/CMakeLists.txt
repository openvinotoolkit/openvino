# Copyright (C) 2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (TARGET_NAME "timetests_helper")

file (GLOB SRC *.cpp)
add_library(${TARGET_NAME} STATIC ${SRC})
target_include_directories(${TARGET_NAME} PUBLIC "${CMAKE_SOURCE_DIR}/include")

include(FetchContent)
FetchContent_Declare(
    gflags
    GIT_REPOSITORY "https://github.com/gflags/gflags.git"
    GIT_TAG "v2.2.2"
)
FetchContent_GetProperties(gflags)
if(NOT gflags_POPULATED)
    FetchContent_Populate(gflags)
    add_subdirectory(${gflags_SOURCE_DIR} ${gflags_BINARY_DIR})
endif()

target_link_libraries(${TARGET_NAME} gflags)
