# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=unknown-warning-option -Wno-error=inconsistent-missing-override -Wno-error=pass-failed")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-error=unknown-warning-option -Wno-error=inconsistent-missing-override -Wno-error=pass-failed")
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

    add_subdirectory(pugixml)
    add_subdirectory(stb_lib)
    add_subdirectory(ade)
    add_subdirectory(fluid/modules/gapi)

    # developer package

    target_include_directories(pugixml INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/pugixml/src>")

    ie_developer_export_targets(ade fluid pugixml)
    if(TARGET pugixml_mt)
        ie_developer_export_targets(pugixml_mt)
    endif()
endfunction()

build_with_lto()

if(ENABLE_MKL_DNN)
    set(SDL_cmake_included ON)
    include(mkldnn.cmake)
endif()
