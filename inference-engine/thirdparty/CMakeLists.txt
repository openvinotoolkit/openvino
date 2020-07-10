# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=unknown-warning-option -Wno-error=inconsistent-missing-override -Wno-error=pass-failed")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-error=unknown-warning-option -Wno-error=inconsistent-missing-override -Wno-error=pass-failed")
elseif(CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 9.1)
    # On g++ 9.3.0 (Ubuntu 20.04) the ADE library raises "redundant-move" warnings
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=redundant-move")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-error=redundant-move")
endif()

function(build_with_lto)
    if(ENABLE_LTO)
        ie_enable_lto()
    endif()

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
        elseif (ANDROID)
            set(CLDNN__ARCHITECTURE_TARGET "Android64" CACHE STRING "" FORCE)
        else()
            set(CLDNN__ARCHITECTURE_TARGET "Linux64" CACHE STRING "" FORCE)
        endif()

        add_subdirectory(clDNN)
    endif()

    function(ie_build_pugixml)
        set(BUILD_TESTS_current ${BUILD_TESTS})
        set(BUILD_TESTS OFF CACHE BOOL "Build tests" FORCE)
        add_subdirectory(pugixml)
        set(BUILD_TESTS ${BUILD_TESTS_current} CACHE BOOL "Build tests" FORCE)
    endfunction()

    ie_build_pugixml()
    add_subdirectory(stb_lib)
    add_subdirectory(ade)
    add_subdirectory(fluid/modules/gapi)

    target_include_directories(pugixml INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/pugixml/src>")

    add_library(half INTERFACE)
    target_include_directories(half INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/half/include>")
    target_compile_definitions(half INTERFACE
        HALF_ERRHANDLING_FLAGS=1
        HALF_ERRHANDLING_ERRNO=0
        HALF_ERRHANDLING_FENV=0
        HALF_ERRHANDLING_THROWS=0
    )

    if (ENABLE_AVX2)
        target_compile_definitions(half INTERFACE HALF_ENABLE_F16C_INTRINSICS=1)
        target_compile_options(half INTERFACE -mf16c)
    else()
        target_compile_definitions(half INTERFACE HALF_ENABLE_F16C_INTRINSICS=0)
    endif()

    set_target_properties(pugixml ade fluid stb_image
                          PROPERTIES FOLDER thirdparty)

    # developer package

    ie_developer_export_targets(ade fluid pugixml half)
    if(TARGET pugixml_mt)
        ie_developer_export_targets(pugixml_mt)
        set_target_properties(pugixml_mt PROPERTIES FOLDER thirdparty)
    endif()

    if(ENABLE_MKL_DNN)
        set(SDL_cmake_included ON)
        include(mkldnn.cmake)
    endif()
endfunction()

build_with_lto()
 