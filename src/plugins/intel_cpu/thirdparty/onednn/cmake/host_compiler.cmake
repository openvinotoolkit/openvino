#===============================================================================
# Copyright 2021 Intel Corporation
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

if(host_compiler_cmake_included)
    return()
endif()
set(host_compiler_cmake_included true)

if(DNNL_DPCPP_HOST_COMPILER MATCHES "g\\+\\+")
    if(WIN32)
        message(FATAL_ERROR "${DNNL_DPCPP_HOST_COMPILER} cannot be used on Windows")
    endif()

    set(DPCPP_HOST_COMPILER_OPTS)

    if(DNNL_TARGET_ARCH STREQUAL "X64")
        if(DNNL_ARCH_OPT_FLAGS STREQUAL "HostOpts")
            platform_gnu_x64_arch_ccxx_flags(DPCPP_HOST_COMPILER_OPTS)
        else()
            # Assumption is that the passed flags are compatible with GNU compiler
            append(DPCPP_HOST_COMPILER_OPTS ${DNNL_ARCH_OPT_FLAGS})
        endif()
    else()
        message(FATAL_ERROR "The DNNL_DPCPP_HOST_COMPILER option is only supported for DNNL_TARGET_ARCH=X64")
    endif()

    platform_unix_and_mingw_common_ccxx_flags(DPCPP_HOST_COMPILER_OPTS)
    platform_unix_and_mingw_common_cxx_flags(DPCPP_HOST_COMPILER_OPTS)

    sdl_unix_common_ccxx_flags(DPCPP_HOST_COMPILER_OPTS)
    sdl_gnu_common_ccxx_flags(DPCPP_HOST_COMPILER_OPTS)
    sdl_gnu_src_ccxx_flags(DPCPP_SRC_CXX_FLAGS)
    sdl_gnu_example_ccxx_flags(DPCPP_EXAMPLE_CXX_FLAGS)

    # SYCL uses C++17 features in headers hence C++17 support should be enabled
    # for host compiler.
    # The main compiler driver doesn't automatically specify C++ standard for
    # custom host compilers.
    append(DPCPP_HOST_COMPILER_OPTS "-std=c++17")

    # Unconditionally enable OpenMP during compilation to use `#pragma omp simd`
    append(DPCPP_HOST_COMPILER_OPTS "-fopenmp")

    string(TOUPPER "${CMAKE_BUILD_TYPE}" UPPERCASE_CMAKE_BUILD_TYPE)
    if(UPPERCASE_CMAKE_BUILD_TYPE STREQUAL "RELEASE")
        append(DPCPP_HOST_COMPILER_OPTS "${CMAKE_CXX_FLAGS_RELEASE}")
    else()
        append(DPCPP_HOST_COMPILER_OPTS "${CMAKE_CXX_FLAGS_DEBUG}")
    endif()

    # SYCL headers contain some comments that trigger warning with GNU compiler
    append(DPCPP_HOST_COMPILER_OPTS "-Wno-comment")
    # SYCL deprecated some API, suppress warnings
    append(DPCPP_HOST_COMPILER_OPTS "-Wno-deprecated-declarations")

    find_program(GNU_COMPILER NAMES ${DNNL_DPCPP_HOST_COMPILER})
    if(NOT GNU_COMPILER)
        message(FATAL_ERROR "GNU host compiler not found")
    else()
        message(STATUS "GNU host compiler: ${GNU_COMPILER}")
    endif()

    execute_process(COMMAND ${GNU_COMPILER} --version OUTPUT_VARIABLE host_compiler_ver ERROR_QUIET)
    string(REGEX REPLACE ".*g\\+\\+.* ([0-9]+\\.[0-9]+)\\.[0-9]+.*" "\\1" host_compiler_ver "${host_compiler_ver}")

    string(REPLACE "." ";" host_compiler_ver_list ${host_compiler_ver})
    list(GET host_compiler_ver_list 0 host_compiler_major_ver)
    list(GET host_compiler_ver_list 1 host_compiler_minor_ver)

    if((host_compiler_major_ver LESS 7) OR (host_compiler_major_ver EQUAL 7 AND host_compiler_minor_ver LESS 4))
        message(FATAL_ERROR "The minimum GNU host compiler version is 7.4")
    else()
        message(STATUS "GNU host compiler version: ${host_compiler_major_ver}.${host_compiler_minor_ver}")
    endif()

    platform_gnu_nowarn_ccxx_flags(DPCPP_CXX_NOWARN_FLAGS ${host_compiler_major_ver}.${host_compiler_minor_ver})

    append(CMAKE_CXX_FLAGS "-fsycl-host-compiler=${GNU_COMPILER}")
    append_host_compiler_options(CMAKE_CXX_FLAGS "${DPCPP_HOST_COMPILER_OPTS}")

    # When using a non-default host compiler the main compiler doesn't
    # handle some arguments properly and issues the warning.
    # Suppress the warning until the bug is fixed.
    append(CMAKE_CXX_FLAGS "-Wno-unused-command-line-argument")
elseif(NOT DNNL_DPCPP_HOST_COMPILER STREQUAL "DEFAULT")
    message(FATAL_ERROR "The valid values for DNNL_DPCPP_HOST_COMPILER: DEFAULT and g++ or absolute path to it")
endif()
