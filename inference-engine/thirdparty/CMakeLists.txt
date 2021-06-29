# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=unknown-warning-option -Wno-error=inconsistent-missing-override -Wno-error=pass-failed")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-error=unknown-warning-option -Wno-error=inconsistent-missing-override -Wno-error=pass-failed")
elseif(CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 9.1)
    # On g++ 9.3.0 (Ubuntu 20.04) the ADE library raises "redundant-move" warnings
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=redundant-move")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-error=redundant-move")
elseif((CMAKE_CXX_COMPILER_ID STREQUAL "MSVC") AND (MSVC_VERSION VERSION_GREATER_EQUAL "1910"))
    # 1910 version of Visual Studio 2017
    # This flagis needed for enabling SIMD vectorization with command '#pragma omp simd'.
    # Compilation with '/openmp:experimental' key allow us to enable vectorizatikon capability in MSVC.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp:experimental")
endif()

if(ENABLE_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()

if (ENABLE_CLDNN)
    if (ENABLE_TESTS AND ENABLE_CLDNN_TESTS)
        set(CLDNN__INCLUDE_TESTS ON CACHE BOOL "" FORCE)
    else()
        set(CLDNN__INCLUDE_TESTS OFF CACHE BOOL "" FORCE)
    endif()
    if (WIN32)
        set(CLDNN__ARCHITECTURE_TARGET "Windows64" CACHE STRING "" FORCE)
    elseif (ANDROID)
        set(CLDNN__ARCHITECTURE_TARGET "Android64" CACHE STRING "" FORCE)
    else()
        set(CLDNN__ARCHITECTURE_TARGET "Linux64" CACHE STRING "" FORCE)
    endif()
    set(CLDNN_THREADING "${THREADING}" CACHE STRING "" FORCE)
    set(GPU_DEBUG_CONFIG OFF CACHE BOOL "Enable debug config feature")
    add_subdirectory(clDNN)
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

# developer package

if (NOT USE_SYSTEM_PUGIXML)
    set_target_properties(pugixml PROPERTIES FOLDER thirdparty)
    openvino_developer_export_targets(COMPONENT openvino_common TARGETS pugixml)
    if(TARGET pugixml_mt)
        openvino_developer_export_targets(COMPONENT openvino_common TARGETS pugixml_mt)
        set_target_properties(pugixml_mt PROPERTIES FOLDER thirdparty)
    endif()
endif()

if(ENABLE_MKL_DNN)
    set(DNNL_ENABLE_CONCURRENT_EXEC ON CACHE BOOL "" FORCE)
    set(DNNL_ENABLE_PRIMITIVE_CACHE OFF CACHE BOOL "" FORCE) ## TODO: try it later
    set(DNNL_ENABLE_MAX_CPU_ISA OFF CACHE BOOL "" FORCE)     ## TODO: try it later
    set(DNNL_LIBRARY_TYPE STATIC CACHE BOOL "" FORCE)
    set(DNNL_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(DNNL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(DNNL_CPU_RUNTIME "${THREADING}" CACHE BOOL "" FORCE)
    set(DNNL_BLAS_VENDOR "NONE" CACHE BOOL "" FORCE)
    set(SDL_cmake_included ON)  ## to skip internal SDL flags. SDL flags are already set on IE level
    if (ANDROID OR ((CMAKE_CXX_COMPILER_ID STREQUAL "MSVC") AND NOT (THREADING STREQUAL "OMP")))
        set(OpenMP_cmake_included ON) ## to skip "omp simd" inside a code. Lead to some crashes inside NDK LLVM..
    endif()

    add_subdirectory(mkl-dnn EXCLUDE_FROM_ALL)
    add_library(mkldnn ALIAS dnnl)
endif()
