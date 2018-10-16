# Copyright (C) 2018 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
add_subdirectory(pugixml)
add_subdirectory(stb_lib)
add_subdirectory(ade)

if (ENABLE_CLDNN AND ENABLE_CLDNN_BUILD)
    include(CheckCXXCompilerFlag)
    CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
    if(NOT COMPILER_SUPPORTS_CXX14)
        message(FATAL_ERROR "Compiler should support C++14 for clDNN building. Please update compiler.")
    endif()
    set(CLDNN__OUTPUT_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
    set(CLDNN__INCLUDE_TESTS OFF CACHE BOOL "" FORCE)
    set(CLDNN__INCLUDE_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(CLDNN__INCLUDE_TUTORIAL OFF CACHE BOOL "" FORCE)
    set(CLDNN__INCLUDE_GENERIC_SAMPLE OFF CACHE BOOL "" FORCE)
    remove_definitions(-fvisibility=default)
    add_subdirectory(clDNN)
    add_definitions(-fvisibility=default)
endif()

include(mkldnn.cmake)
