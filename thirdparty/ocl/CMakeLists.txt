# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

function(get_lib_name TARGET_NAME LIBRARY_NAME)
    if(WIN32)
        set(LIB_SUFFIX "${OV_BUILD_POSTFIX}${CMAKE_LINK_LIBRARY_SUFFIX}")
    else()
        set(LIB_SUFFIX "${OV_BUILD_POSTFIX}${CMAKE_SHARED_LIBRARY_SUFFIX}")
    endif()

    set("${LIBRARY_NAME}" "${CMAKE_SHARED_MODULE_PREFIX}${TARGET_NAME}${LIB_SUFFIX}" PARENT_SCOPE)
endfunction()

function(get_lib_path OUTPUT_DIR FINAL_OUTPUT_DIR)
    if(OV_GENERATOR_MULTI_CONFIG)
        set(OUTPUT_DIR "${OUTPUT_DIR}/$<CONFIG>")
    endif()

    set("${FINAL_OUTPUT_DIR}" "${OUTPUT_DIR}" PARENT_SCOPE)
endfunction()

# for icd_loader project we need to override path to OpenCL C headers
set(OPENCL_ICD_LOADER_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cl_headers/"
    CACHE PATH "Path to OCL includes" FORCE)

# Reset compilation options for ICD loader as default OpenVINO cmake options lead to removing a lot of symbols from the library
unset(CMAKE_C_VISIBILITY_PRESET)
unset(CMAKE_DEBUG_POSTFIX)

# OpenCL has absolute paths to include directories
set(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION OFF)
if(NOT DEFINED USE_DYNAMIC_VCXX_RUNTIME)
    set(USE_DYNAMIC_VCXX_RUNTIME ON)
endif()

#
# Suppress warnings
#
if(OV_COMPILER_IS_INTEL_LLVM AND WIN32)
    ov_add_compiler_flags("/Wno-macro-redefined")
    ov_add_compiler_flags("/Wno-nonportable-include-path")
    ov_add_compiler_flags("/Wno-comment")
endif()

#
# Build OpenCL ICD loader
#

add_subdirectory(icd_loader EXCLUDE_FROM_ALL)

# use CL/opencl.hpp
target_compile_definitions(OpenCL INTERFACE OV_GPU_USE_OPENCL_HPP
                                            OV_GPU_OPENCL_HPP_HAS_UUID)
target_include_directories(OpenCL INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cl_headers>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/clhpp_headers/include>)

# The following variables are needed to make find_package(OpenCL) work

get_lib_path("${CMAKE_RUNTIME_OUTPUT_DIRECTORY}" OPENCL_LIB_DIR)
get_lib_name("OpenCL" OPENCL_LIB_NAME)

set(OpenCL_INCLUDE_DIR "${OPENCL_ICD_LOADER_HEADERS_DIR}" PARENT_SCOPE)
set(OpenCL_LIBRARY "${OPENCL_LIB_DIR}/${OPENCL_LIB_NAME}" PARENT_SCOPE)
set(opencl_root_hints "${OpenCL_INCLUDE_DIR}" PARENT_SCOPE)

# installation

ov_install_static_lib(OpenCL ${OV_CPACK_COMP_CORE})
