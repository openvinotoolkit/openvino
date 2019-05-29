# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-warning-option -Wno-inconsistent-missing-override -Wno-pass-failed")
  set(CMAKE_CC_FLAGS "${CMAKE_CC_FLAGS} -Wno-unknown-warning-option -Wno-inconsistent-missing-override -Wno-pass-failed")
endif()

export(TARGETS pugixml NAMESPACE IE:: APPEND FILE "${CMAKE_BINARY_DIR}/targets_developer.cmake")

add_subdirectory(stb_lib)
add_subdirectory(ade)

if (ENABLE_CLDNN)
    set(CLDNN__OUTPUT_BIN_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    set(CLDNN__OUTPUT_LIB_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
    if (ENABLE_TESTS AND ENABLE_CLDNN_TESTS)
        set(CLDNN__INCLUDE_TESTS ON CACHE BOOL "" FORCE)
        set(CLDNN__INCLUDE_CORE_INTERNAL_TESTS ON CACHE BOOL "" FORCE)
    else()
        set(CLDNN__INCLUDE_TESTS OFF CACHE BOOL "" FORCE)
        set(CLDNN__INCLUDE_CORE_INTERNAL_TESTS OFF CACHE BOOL "" FORCE)
    endif()
    set(CLDNN__INCLUDE_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(CLDNN__INCLUDE_TUTORIAL OFF CACHE BOOL "" FORCE)
    if (WIN32)
        set(CLDNN__ARCHITECTURE_TARGET "Windows64" CACHE STRING "" FORCE)
    else()
        set(CLDNN__ARCHITECTURE_TARGET "Linux64" CACHE STRING "" FORCE)
    endif()

    add_subdirectory(clDNN)
endif()

if (UNIX OR APPLE AND ${CMAKE_BUILD_TYPE} STREQUAL "Release")
    remove_definitions(-fvisibility=hidden)
    add_definitions(-fvisibility=default)
endif()

include(ngraph.cmake)

if (UNIX OR APPLE AND ${CMAKE_BUILD_TYPE} STREQUAL "Release")
    remove_definitions(-fvisibility=default)
    add_definitions(-fvisibility=hidden)
endif()

if(ENABLE_MKL_DNN)
    include(mkldnn.cmake)
endif()

add_subdirectory("${IE_MAIN_SOURCE_DIR}/thirdparty/fluid/modules/gapi")
