# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

function(get_lib_name TARGET_NAME LIBRARY_NAME)
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(BUILD_SUFFIX ${IE_DEBUG_POSTFIX})
    else()
        set(BUILD_SUFFIX ${IE_RELEASE_POSTFIX})
    endif()

    if(WIN32)
        set(LIB_SUFFIX "${BUILD_SUFFIX}${CMAKE_LINK_LIBRARY_SUFFIX}")
    else()
        set(LIB_SUFFIX "${BUILD_SUFFIX}${CMAKE_SHARED_LIBRARY_SUFFIX}")
    endif()

    set("${LIBRARY_NAME}" "${CMAKE_SHARED_MODULE_PREFIX}${TARGET_NAME}${LIB_SUFFIX}" PARENT_SCOPE)
endfunction()

function(get_lib_path OUTPUT_DIR FINAL_OUTPUT_DIR)
    if(WIN32)
        set(LIB_DIR "")
    else()
        set(LIB_DIR "lib")
    endif()

    if (NOT CMAKE_GENERATOR MATCHES "Ninja" AND NOT UNIX)
        set(OUTPUT_DIR "${OUTPUT_DIR}/${CMAKE_BUILD_TYPE}")
    endif()

    set("${FINAL_OUTPUT_DIR}" "${OUTPUT_DIR}/${LIB_DIR}" PARENT_SCOPE)
endfunction()

if(WIN32)
    set(CMAKE_DEBUG_POSTFIX "")
endif()
set(OPENCL_ICD_LOADER_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cl_headers/" CACHE PATH "Path to OCL includes" FORCE)

set(OPENCL_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cl_headers/" "${CMAKE_CURRENT_SOURCE_DIR}/clhpp_headers/include/" CACHE PATH "Path to OCL (CL and CLHPP) includes" FORCE)

# Reset compilation options for ICD loader as default IE options lead to removing a lot of symbols from the library
set(CMAKE_C_VISIBILITY_PRESET default)
set(CMAKE_SHARED_LINKER_FLAGS "")
set(CMAKE_C_FLAGS "")
set(CMAKE_CXX_FLAGS "")
set(CMAKE_C_FLAGS_RELEASE "")
set(CMAKE_CXX_FLAGS_RELEASE "")
# OpenCL has absolute paths to include directories
set(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION OFF)
if(NOT DEFINED USE_DYNAMIC_VCXX_RUNTIME)
    set(USE_DYNAMIC_VCXX_RUNTIME ON)
endif()

add_subdirectory(icd_loader EXCLUDE_FROM_ALL)

# override INTERFACE_INCLUDE_DIRECTORIES
set_property(TARGET OpenCL PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    $<BUILD_INTERFACE:${OPENCL_ICD_LOADER_HEADERS_DIR}>)
foreach(dir IN LISTS OPENCL_HEADERS_DIR)
    target_include_directories(OpenCL SYSTEM PUBLIC $<BUILD_INTERFACE:${dir}>)
endforeach()

# The following varables are needed to make find_package(OpenCL) work
set(OPENCL_VERSION_2_2 ON CACHE BOOL "" FORCE)
set(OpenCL_VERSION_STRING "2.2" CACHE STRING "" FORCE)
set(OpenCL_INCLUDE_DIR "${OPENCL_HEADERS_DIR}" CACHE PATH "" FORCE)
set(OPENCLROOT "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}" CACHE PATH "" FORCE)

get_lib_path("${OUTPUT_ROOT}/${BIN_FOLDER}" OPENCL_LIB_DIR)
get_lib_name("OpenCL" OPENCL_LIB_NAME)

set(OpenCL_LIBRARY "${OPENCL_LIB_DIR}/${OPENCL_LIB_NAME}" CACHE PATH "" FORCE)

ov_install_static_lib(OpenCL gpu)
