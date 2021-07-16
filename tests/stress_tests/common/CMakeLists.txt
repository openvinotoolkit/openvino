# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME StressTestsCommon)

file (GLOB_RECURSE SRC *.cpp)
file (GLOB_RECURSE HDR *.h)

add_library(${TARGET_NAME} STATIC ${SRC} ${HDR})

target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(EXISTS "${OpenVINO_SOURCE_DIR}/thirdparty/gflags")
    function(add_gflags)
        set(GFLAGS_IS_SUBPROJECT TRUE)
        set(HAVE_SYS_STAT_H 1)
        set(HAVE_INTTYPES_H 1)
        set(INTTYPES_FORMAT C99)
        set(BUILD_TESTING OFF)
        set(BUILD_SHARED_LIBS OFF)
        add_subdirectory(${OpenVINO_SOURCE_DIR}/thirdparty/gflags
                         ${CMAKE_CURRENT_BINARY_DIR}/gflags_build
                         EXCLUDE_FROM_ALL)
        set_target_properties(gflags_nothreads_static PROPERTIES FOLDER thirdparty)
    endfunction()
    add_gflags()
endif()

target_link_libraries(${TARGET_NAME}
    PUBLIC
        IE::gtest
        IE::pugixml
        ${InferenceEngine_LIBRARIES}
        gflags
    PRIVATE
        IE::gtest_main)
