# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=unknown-warning-option -Wno-error=inconsistent-missing-override -Wno-error=pass-failed")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-error=unknown-warning-option -Wno-error=inconsistent-missing-override -Wno-error=pass-failed")
elseif(CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 9.1)
    # On g++ 9.3.0 (Ubuntu 20.04) the ADE library raises "redundant-move" warnings
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=redundant-move")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-error=redundant-move")
elseif((CMAKE_CXX_COMPILER_ID STREQUAL "MSVC") AND (MSVC_VERSION VERSION_GREATER_EQUAL "1920"))
    # 1920 version of MSVC 2019
    # This flagis needed for enabling SIMD vectorization with command '#pragma omp simd'.
    # Compilation with '/openmp:experimental' key allow us to enable vectorizatikon capability in MSVC.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp:experimental")
endif()

if(ENABLE_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
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

    # disable CLDNN docs build
    set_target_properties(docs PROPERTIES EXCLUDE_FROM_ALL ON FOLDER docs)
endif()

function(ie_build_pugixml)
    set(BUILD_TESTS_current ${BUILD_TESTS})
    set(BUILD_TESTS OFF CACHE BOOL "Build tests" FORCE)
    set(BUILD_SHARED_LIBS OFF)
    add_subdirectory(pugixml EXCLUDE_FROM_ALL)
    set(BUILD_TESTS ${BUILD_TESTS_current} CACHE BOOL "Build tests" FORCE)
endfunction()

if (USE_SYSTEM_PUGIXML)
    find_package(PugiXML REQUIRED)
    set_property(TARGET pugixml PROPERTY IMPORTED_GLOBAL TRUE)
else()
    ie_build_pugixml()
    target_include_directories(pugixml INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/pugixml/src>")
endif()

add_subdirectory(stb_lib)
add_subdirectory(ade EXCLUDE_FROM_ALL)
add_subdirectory(fluid/modules/gapi)

set_target_properties(ade fluid stb_image PROPERTIES FOLDER thirdparty)

# developer package

ie_developer_export_targets(ade fluid)

if (NOT USE_SYSTEM_PUGIXML)
    set_target_properties(pugixml PROPERTIES FOLDER thirdparty)
    ie_developer_export_targets(pugixml)
    if(TARGET pugixml_mt)
        ie_developer_export_targets(pugixml_mt)
        set_target_properties(pugixml_mt PROPERTIES FOLDER thirdparty)
    endif()
endif()

if(ENABLE_MKL_DNN)
    set(SDL_cmake_included ON)
    include(mkldnn.cmake)
endif()
