#===============================================================================
# Copyright 2019-2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

if(SYCL_cmake_included)
    return()
endif()
set(SYCL_cmake_included true)

if(NOT DNNL_WITH_SYCL)
    if (NOT DNNL_DPCPP_HOST_COMPILER STREQUAL "DEFAULT")
        message(FATAL_ERROR "DNNL_DPCPP_HOST_COMPILER is supported only for DPCPP runtime")
    endif()
    return()
endif()

include(FindPackageHandleStandardArgs)
include(CheckCXXCompilerFlag)

# Check if CXX is Intel oneAPI DPC++ Compiler
CHECK_CXX_COMPILER_FLAG(-fsycl DPCPP_SUPPORTED)
find_package(LevelZero)

if(DPCPP_SUPPORTED)
    if(LevelZero_FOUND)
        message(STATUS "DPC++ support is enabled (OpenCL and Level Zero)")
    else()
        message(STATUS "DPC++ support is enabled (OpenCL)")
    endif()

    # Explicitly link against sycl as Intel oneAPI DPC++ Compiler does not
    # always do it implicitly.
    if(WIN32)
        list(APPEND EXTRA_SHARED_LIBS
            $<$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithMDd>>:sycld>
            $<$<AND:$<NOT:$<CONFIG:Debug>>,$<NOT:$<CONFIG:RelWithMDd>>>:sycl>)
    else()
        list(APPEND EXTRA_SHARED_LIBS sycl)
    endif()

    if(NOT DNNL_DPCPP_HOST_COMPILER STREQUAL "DEFAULT" AND DNNL_SYCL_CUDA)
        message(FATAL_ERROR "DNNL_DPCPP_HOST_COMPILER options is not supported for NVIDIA.")
    endif()

    if(DNNL_SYCL_CUDA)
        # XXX: Suppress warning coming from SYCL headers:
        #   error: use of function template name with no prior declaration in
        #   function call with eplicit template arguments is a C++20 extension
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c++20-extensions")

        # Explicitly linking against OpenCL without finding the right one can
        # end up linking the tests against Nvidia OpenCL. This can be
        # problematic as Intel OpenCL CPU backend will not work. When multiple
        # OpenCL backends are available we need to make sure that we are linking
        # against the correct one.
        find_package(OpenCL REQUIRED)
        find_package(cuBLAS REQUIRED)
        find_package(cuDNN REQUIRED)
        list(APPEND EXTRA_SHARED_LIBS OpenCL::OpenCL)
    else()
        find_library(OPENCL_LIBRARY OpenCL PATHS ENV LIBRARY_PATH ENV LIB NO_DEFAULT_PATH)
        if(OPENCL_LIBRARY)
            message(STATUS "OpenCL runtime is found in the environment: ${OPENCL_LIBRARY}")
            # OpenCL runtime was found in the environment hence simply add it to
            # the EXTRA_SHARED_LIBS list
            list(APPEND EXTRA_SHARED_LIBS ${OPENCL_LIBRARY})
        else()
            message(STATUS "OpenCL runtime is not found in the environment. Try to find it using find_package(...)")
            # This is expected when using OSS compiler that doesn't distribute
            # OpenCL runtime
            find_package(OpenCL REQUIRED)
            # Unset INTERFACE_INCLUDE_DIRECTORIES property because DPCPP
            # compiler contains OpenCL headers
            set_target_properties(OpenCL::OpenCL PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "")
            list(APPEND EXTRA_SHARED_LIBS OpenCL::OpenCL)
        endif()
    endif()

    # XXX: Suppress warning coming from SYCL headers:
    #   #pragma message("The Intel extensions have been moved into cl_ext.h.
    #   Please include cl_ext.h directly.")
    if(NOT WIN32)
        if(${CMAKE_VERSION} VERSION_LESS "3.1.0")
            # Prior to CMake 3.1 the Makefile generators did not escape # correctly
            # inside make variable assignments used in generated makefiles, causing
            # them to be treated as comments. This is a workaround.
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-\\#pragma-messages")
        else()
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-#pragma-messages")
        endif()
    endif()

    add_definitions_with_host_compiler("-DCL_TARGET_OPENCL_VERSION=300")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")

    if(LevelZero_FOUND)
        set(DNNL_WITH_LEVEL_ZERO TRUE)
        include_directories_with_host_compiler(${LevelZero_INCLUDE_DIRS})
    endif()
else()
    message(FATAL_ERROR "${CMAKE_CXX_COMPILER_ID} is not Intel oneAPI DPC++ Compiler")
endif()
