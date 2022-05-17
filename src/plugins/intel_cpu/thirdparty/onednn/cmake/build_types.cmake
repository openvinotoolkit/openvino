#===============================================================================
# Copyright 2020-2021 Intel Corporation
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

# Manage custom build types
#===============================================================================

if(Build_types_cmake_included)
    return()
endif()

set(Build_types_cmake_included true)

# Use flags of the Release build type but filter out the NDEBUG flag
set(REGEX_REMOVE_DNDEBUG "[/-][D][ ]*NDEBUG")
string(REGEX REPLACE "${REGEX_REMOVE_DNDEBUG}" ""
    _CMAKE_CXX_FLAGS_RELWITHASSERT "${CMAKE_CXX_FLAGS_RELEASE}")
string(REGEX REPLACE "${REGEX_REMOVE_DNDEBUG}" ""
    _CMAKE_C_FLAGS_RELWITHASSERT "${CMAKE_C_FLAGS_RELEASE}")

set(CMAKE_CXX_FLAGS_RELWITHASSERT "${_CMAKE_CXX_FLAGS_RELWITHASSERT}" CACHE STRING
    "Flags used by the C++ compiler during RelWithAssert builds." FORCE)
set(CMAKE_C_FLAGS_RELWITHASSERT "${_CMAKE_C_FLAGS_RELWITHASSERT}" CACHE STRING
    "Flags used by the C compiler during RelWithAssert builds." FORCE)
set(CMAKE_EXE_LINKER_FLAGS_RELWITHASSERT "${CMAKE_EXE_LINKER_FLAGS_RELEASE}" CACHE STRING
    "Flags used for linking binaries during RelWithAssert builds." FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_RELWITHASSERT "${CMAKE_SHARED_LINKER_FLAGS_RELEASE}" CACHE STRING
    "Flags used by the shared libraries linker during RelWithAssert builds." FORCE)

mark_as_advanced(
    CMAKE_CXX_FLAGS_RELWITHASSERT
    CMAKE_C_FLAGS_RELWITHASSERT
    CMAKE_EXE_LINKER_FLAGS_RELWITHASSERT
    CMAKE_SHARED_LINKER_FLAGS_RELWITHASSERT)


# Release build with linking to the Debug Runtime
string(REGEX REPLACE "--dependent-lib=msvcrt" "--dependent-lib=msvcrtd"
    _CMAKE_CXX_FLAGS_RELWITHMDD "${CMAKE_CXX_FLAGS_RELEASE}")
string(REGEX REPLACE "--dependent-lib=msvcrt" "--dependent-lib=msvcrtd"
    _CMAKE_C_FLAGS_RELWITHMDD "${CMAKE_C_FLAGS_RELEASE}")

string(REGEX REPLACE "NDEBUG" "_DEBUG"
    _CMAKE_CXX_FLAGS_RELWITHMDD "${_CMAKE_CXX_FLAGS_RELWITHMDD}")
string(REGEX REPLACE "NDEBUG" "_DEBUG"
    _CMAKE_C_FLAGS_RELWITHMDD "${_CMAKE_C_FLAGS_RELWITHMDD}")

set(CMAKE_C_FLAGS_RELWITHMDD "${_CMAKE_C_FLAGS_RELWITHMDD}" CACHE STRING
     "Flags used by the C compiler during RelWithMdd build" FORCE)
set(CMAKE_CXX_FLAGS_RELWITHMDD "${_CMAKE_CXX_FLAGS_RELWITHMDD}"
     CACHE STRING "Flags used by the C++ compiler during RelWithMdd build." FORCE)

set(CMAKE_EXE_LINKER_FLAGS_RELWITHMDD "${CMAKE_EXE_LINKER_FLAGS_RELEASE}"
     CACHE STRING "Flags used for linking binaries during RelWithMdd builds." FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_RELWITHMDD "${CMAKE_SHARED_LINKER_FLAGS_RELEASE}"
     CACHE STRING "Flags used by the shared libraries linker during RelWithMdd builds." FORCE)

mark_as_advanced(
    CMAKE_CXX_FLAGS_RELWITHMDD
    CMAKE_C_FLAGS_RELWITHMDD
    CMAKE_EXE_LINKER_FLAGS_RELWITHMDD
    CMAKE_SHARED_LINKER_FLAGS_RELWITHMDD)
