# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(OPENCL_ICD_LOADER_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cl_headers/" CACHE PATH "Path to OCL includes" FORCE)

set(OPENCL_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cl_headers/" "${CMAKE_CURRENT_SOURCE_DIR}/clhpp_headers/include/" CACHE PATH "Path to OCL (CL and CLHPP) includes" FORCE)

# Reset compilation options for ICD loader as default IE options lead to removing a lot of symbols from the library
set(CMAKE_C_VISIBILITY_PRESET default)
set(CMAKE_SHARED_LINKER_FLAGS "")
set(CMAKE_C_FLAGS "")
set(CMAKE_CXX_FLAGS "")
set(CMAKE_C_FLAGS_RELEASE "")
set(CMAKE_CXX_FLAGS_RELEASE "")

add_subdirectory(icd_loader)

target_include_directories(OpenCL SYSTEM PUBLIC ${OPENCL_HEADERS_DIR})

# The following varables are needed to make find_package(OpenCL) work
set(OpenCL_VERSION_STRING "2.2" CACHE STRING "" FORCE)
set(OpenCL_INCLUDE_DIR "${OPENCL_ICD_LOADER_HEADERS_DIR}" CACHE PATH "" FORCE)
set(OPENCLROOT "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}" CACHE PATH "" FORCE)
