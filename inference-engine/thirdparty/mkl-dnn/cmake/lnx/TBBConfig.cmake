# Copyright (c) 2017-2018 Intel Corporation
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
#
#
#
#

# TBB_FOUND should not be set explicitly. It is defined automatically by CMake.
# Handling of TBB_VERSION is in TBBConfigVersion.cmake.

if (NOT TBB_FIND_COMPONENTS)
    set(TBB_FIND_COMPONENTS "tbb;tbbmalloc;tbbmalloc_proxy")
    foreach (_tbb_component ${TBB_FIND_COMPONENTS})
        set(TBB_FIND_REQUIRED_${_tbb_component} 1)
    endforeach()
endif()

# Add components with internal dependencies: tbbmalloc_proxy -> tbbmalloc
list(FIND TBB_FIND_COMPONENTS tbbmalloc_proxy _tbbmalloc_proxy_ix)
if (NOT _tbbmalloc_proxy_ix EQUAL -1)
    list(FIND TBB_FIND_COMPONENTS tbbmalloc _tbbmalloc_ix)
    if (_tbbmalloc_ix EQUAL -1)
        list(APPEND TBB_FIND_COMPONENTS tbbmalloc)
        set(TBB_FIND_REQUIRED_tbbmalloc ${TBB_FIND_REQUIRED_tbbmalloc_proxy})
    endif()
endif()

# Intel MKL-DNN changes: use TBBROOT to locate Intel TBB
# get_filename_component(_tbb_root "${CMAKE_CURRENT_LIST_FILE}" PATH)
# get_filename_component(_tbb_root "${_tbb_root}" PATH)
set(_tbb_root ${TBBROOT})

set(_tbb_x32_subdir ia32)
set(_tbb_x64_subdir intel64)

if (CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(_tbb_arch_subdir ${_tbb_x64_subdir})
else()
    set(_tbb_arch_subdir ${_tbb_x32_subdir})
endif()

if (CMAKE_CXX_COMPILER_LOADED)
    set(_tbb_compiler_id ${CMAKE_CXX_COMPILER_ID})
    set(_tbb_compiler_ver ${CMAKE_CXX_COMPILER_VERSION})
elseif (CMAKE_C_COMPILER_LOADED)
    set(_tbb_compiler_id ${CMAKE_C_COMPILER_ID})
    set(_tbb_compiler_ver ${CMAKE_C_COMPILER_VERSION})
endif()

# For non-GCC compilers try to find version of system GCC to choose right compiler subdirectory.
if (NOT _tbb_compiler_id STREQUAL "GNU")
    execute_process(COMMAND gcc --version OUTPUT_VARIABLE _tbb_gcc_ver_output ERROR_QUIET)
    string(REGEX REPLACE ".*gcc.*([0-9]+\\.[0-9]+)\\.[0-9]+.*" "\\1" _tbb_compiler_ver "${_tbb_gcc_ver_output}")
    if (NOT _tbb_compiler_ver)
        message(FATAL_ERROR "This Intel TBB package is intended to be used only environment with available 'gcc'")
    endif()
    unset(_tbb_gcc_ver_output)
endif()

set(_tbb_lib ${_tbb_root}/lib/${_tbb_arch_subdir} )
file(GLOB _tbb_gcc_versions_available RELATIVE ${_tbb_lib} ${_tbb_lib}/*)
# shall we check _tbb_gcc_versions_available is not empty?
foreach (_tbb_gcc_version ${_tbb_gcc_versions_available})
    string(SUBSTRING ${_tbb_gcc_version} 3 -1 _tbb_gcc_version_number)
    if (NOT _tbb_compiler_ver VERSION_LESS _tbb_gcc_version_number)
        set(_tbb_compiler_subdir ${_tbb_gcc_version})
    endif()
endforeach()

unset(_tbb_gcc_version_number)
unset(_tbb_compiler_id)
unset(_tbb_compiler_ver)


# we need to check the version of tbb
file(READ "${_tbb_root}/include/tbb/tbb_stddef.h" _tbb_stddef)
string(REGEX REPLACE ".*#define TBB_INTERFACE_VERSION ([0-9]+).*" "\\1" TBB_INTERFACE_VERSION "${_tbb_stddef}")
if (${TBB_INTERFACE_VERSION} VERSION_LESS 9100)
    message(FATAL_ERROR "MKL-DNN requires TBB version 2017 or above")
endif()

# Now we check that all the needed component are present
get_filename_component(_tbb_lib_path "${_tbb_root}/lib/${_tbb_arch_subdir}/${_tbb_compiler_subdir}" ABSOLUTE)

foreach (_tbb_component ${TBB_FIND_COMPONENTS})
    set(_tbb_release_lib "${_tbb_lib_path}/lib${_tbb_component}.so.2")
    set(_tbb_debug_lib "${_tbb_lib_path}/lib${_tbb_component}_debug.so.2")

    if (EXISTS "${_tbb_release_lib}" AND EXISTS "${_tbb_debug_lib}")
        add_library(TBB::${_tbb_component} SHARED IMPORTED)
        set_target_properties(TBB::${_tbb_component} PROPERTIES
                              IMPORTED_CONFIGURATIONS "RELEASE;DEBUG"
                              IMPORTED_LOCATION_RELEASE     "${_tbb_release_lib}"
                              IMPORTED_LOCATION_DEBUG       "${_tbb_debug_lib}"
                              INTERFACE_INCLUDE_DIRECTORIES "${_tbb_root}/include")

        # Add internal dependencies for imported targets: TBB::tbbmalloc_proxy -> TBB::tbbmalloc
        if (_tbb_component STREQUAL tbbmalloc_proxy)
            set_target_properties(TBB::tbbmalloc_proxy PROPERTIES INTERFACE_LINK_LIBRARIES TBB::tbbmalloc)
        endif()

        list(APPEND TBB_IMPORTED_TARGETS TBB::${_tbb_component})
        set(TBB_${_tbb_component}_FOUND 1)
    elseif (TBB_FIND_REQUIRED AND TBB_FIND_REQUIRED_${_tbb_component})
        message(FATAL_ERROR "Missed required Intel TBB component: ${_tbb_component}")
    endif()
endforeach()

unset(_tbb_x32_subdir)
unset(_tbb_x64_subdir)
unset(_tbb_arch_subdir)
unset(_tbb_compiler_subdir)
unset(_tbbmalloc_proxy_ix)
unset(_tbbmalloc_ix)
unset(_tbb_lib_path)
unset(_tbb_release_lib)
unset(_tbb_debug_lib)
