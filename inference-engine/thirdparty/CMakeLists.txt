# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-warning-option -Wno-inconsistent-missing-override -Wno-pass-failed")
  set(CMAKE_CC_FLAGS "${CMAKE_CC_FLAGS} -Wno-unknown-warning-option -Wno-inconsistent-missing-override -Wno-pass-failed")
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
        else()
            set(CLDNN__ARCHITECTURE_TARGET "Linux64" CACHE STRING "" FORCE)
        endif()

        add_subdirectory(clDNN)
    endif()

    add_subdirectory(pugixml)
    export(TARGETS pugixml NAMESPACE IE:: APPEND FILE "${CMAKE_BINARY_DIR}/targets.cmake")
    export(TARGETS pugixml NAMESPACE IE:: APPEND FILE "${CMAKE_BINARY_DIR}/targets_developer.cmake")

    if (TARGET pugixml_mt)
        export(TARGETS pugixml_mt NAMESPACE IE:: APPEND FILE "${CMAKE_BINARY_DIR}/targets.cmake")
        export(TARGETS pugixml_mt NAMESPACE IE:: APPEND FILE "${CMAKE_BINARY_DIR}/targets_developer.cmake")
    endif()

    add_subdirectory(stb_lib)

    add_subdirectory(ade)
    export(TARGETS ade NAMESPACE IE:: APPEND FILE "${CMAKE_BINARY_DIR}/targets_developer.cmake")

    add_subdirectory(fluid/modules/gapi)
    export(TARGETS fluid NAMESPACE IE:: APPEND FILE "${CMAKE_BINARY_DIR}/targets_developer.cmake")
endfunction()

build_with_lto()

if(ENABLE_MKL_DNN)
    include(mkldnn.cmake)
endif()
