# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (TARGET_NAME "timetests_helper")

file (GLOB SRC *.cpp)
add_library(${TARGET_NAME} STATIC ${SRC})
target_include_directories(${TARGET_NAME} PUBLIC "${CMAKE_SOURCE_DIR}/include")

if(EXISTS "${OpenVINO_MAIN_SOURCE_DIR}/thirdparty/gflags")
    function(add_gflags)
        set(GFLAGS_IS_SUBPROJECT TRUE)
        set(HAVE_SYS_STAT_H 1)
        set(HAVE_INTTYPES_H 1)
        set(INTTYPES_FORMAT C99)
        set(BUILD_TESTING OFF)
        set(BUILD_SHARED_LIBS OFF)
        add_subdirectory(${OpenVINO_MAIN_SOURCE_DIR}/thirdparty/gflags
                         ${CMAKE_CURRENT_BINARY_DIR}/gflags_build
                         EXCLUDE_FROM_ALL)
        set_target_properties(gflags_nothreads_static PROPERTIES FOLDER thirdparty)
    endfunction()
    add_gflags()
endif()

target_link_libraries(${TARGET_NAME} gflags)
