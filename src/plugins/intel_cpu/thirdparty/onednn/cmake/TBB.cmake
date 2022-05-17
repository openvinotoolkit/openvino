#===============================================================================
# Copyright 2018-2021 Intel Corporation
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

# Manage TBB-related compiler flags
#===============================================================================

if(TBB_cmake_included)
    return()
endif()
set(TBB_cmake_included true)
include("cmake/Threading.cmake")

macro(handle_tbb_target)
    if(TBB_FOUND)
        set_property(TARGET TBB::tbb PROPERTY "MAP_IMPORTED_CONFIG_RELWITHMDD" "DEBUG")
        foreach(inc_dir ${_tbb_include_dirs})
            include_directories(BEFORE SYSTEM ${inc_dir})
            append_host_compiler_options(CMAKE_CXX_FLAGS "-I${inc_dir}")
        endforeach()
        list(APPEND EXTRA_SHARED_LIBS ${TBB_IMPORTED_TARGETS})

        # Print TBB location
        get_filename_component(_tbb_root "${_tbb_include_dirs}" PATH)
        get_filename_component(_tbb_root "${_tbb_root}" ABSOLUTE)
        message(STATUS "TBB: ${_tbb_root}")

        unset(_tbb_include_dirs)
        unset(_tbb_root)
    elseif(DNNL_CPU_RUNTIME STREQUAL "NONE")
        message(FATAL_ERROR "For GPU only SYCL configuration TBB is required for testing.")
    else()
        message(FATAL_ERROR "DNNL_CPU_THREADING_RUNTIME is ${DNNL_CPU_THREADING_RUNTIME} but TBB is not found.")
    endif()

    get_target_property(_tbb_lib_path TBB::tbb IMPORTED_LOCATION_RELEASE)
    get_filename_component(_tbb_lib_dir "${_tbb_lib_path}" PATH)

    # XXX: workaround - Intel oneAPI DPC++ Compiler "unbundles" tbb.lib
    # and loses its abosulte path
    if(DNNL_WITH_SYCL)
        link_directories(${_tbb_lib_dir})
    endif()

    # XXX: this is to make "ctest" working out-of-the-box with TBB
    string(REPLACE "/lib/" "/redist/" _tbb_redist_dir "${_tbb_lib_dir}")
    append_to_windows_path_list(CTESTCONFIG_PATH "${_tbb_redist_dir}")
endmacro()

if(NOT "${DNNL_CPU_THREADING_RUNTIME}" MATCHES "^(TBB|TBB_AUTO)$")
    return()
endif()

find_package_tbb(REQUIRED)
handle_tbb_target()

unset(_tbb_lib_path)
unset(_tbb_lib_dir)
unset(_tbb_redist_dir)
