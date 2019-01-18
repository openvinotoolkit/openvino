#===============================================================================
# Copyright 2018 Intel Corporation
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

if(NOT MKLDNN_THREADING STREQUAL "TBB")
    return()
endif()

if (NOT TBBROOT)
    if(DEFINED ENV{TBBROOT})
        set (TBBROOT $ENV{TBBROOT})
    else()
        message("FATAL_ERROR" "TBBROOT is unset")
    endif()
endif()

if(WIN32)
    find_package(TBB REQUIRED tbb HINTS cmake/win)
elseif(APPLE)
    find_package(TBB REQUIRED tbb HINTS cmake/mac)
elseif(UNIX)
    find_package(TBB REQUIRED tbb HINTS cmake/lnx)
endif()

set_threading("TBB")
list(APPEND mkldnn_LINKER_LIBS ${TBB_IMPORTED_TARGETS})

message(STATUS "Intel(R) TBB: ${TBBROOT}")
