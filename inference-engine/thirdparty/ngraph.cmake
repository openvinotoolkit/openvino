# ******************************************************************************
# Copyright 2017-2019 Intel Corporation
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
# ******************************************************************************
set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
if (NOT("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel"))
    set (CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
endif()
if (WIN32)
    add_definitions(-DNOMINMAX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W0 /EHsc /MP")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_CRT_SECURE_NO_WARNINGS")
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4308")
endif()

set(NGRAPH_TOOLS_ENABLE FALSE)
set(NGRAPH_STATIC_LIB_ENABLE TRUE)
set(NGRAPH_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ngraph/src/ngraph")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/ngraph/src"
    "${NGRAPH_SOURCE_DIR}")

add_definitions(-DPROJECT_ROOT_DIR="${NGRAPH_SOURCE_DIR}")

set(NGRAPH_INSTALL_LIB "${CMAKE_INSTALL_PREFIX}")

check_cxx_compiler_flag("-Wmaybe-uninitialized" HAS_MAYBE_UNINITIALIZED)
if (HAS_MAYBE_UNINITIALIZED)
        if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-uninitialized")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-uninitialized")
        else()
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-maybe-uninitialized -Wno-return-type")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-maybe-uninitialized -Wno-return-type")
        endif()
endif()
# WA for GCC 7.0
if (UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-return-type")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-return-type")
endif()
add_subdirectory(${NGRAPH_SOURCE_DIR})
