# ******************************************************************************
# Copyright 2020-2021 Arm Limited and affiliates.
# SPDX-License-Identifier: Apache-2.0
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

if(acl_cmake_included)
    return()
endif()
set(acl_cmake_included true)
include("cmake/options.cmake")

if(NOT DNNL_TARGET_ARCH STREQUAL "AARCH64")
    return()
endif()

if(NOT DNNL_AARCH64_USE_ACL)
    return()
endif()

find_package(ACL REQUIRED)

set(ACL_MINIMUM_VERSION "21.08")

if(ACL_FOUND)
    file(GLOB_RECURSE ACL_VERSION_FILE $ENV{ACL_ROOT_DIR}/*/arm_compute_version.embed)
    if ("${ACL_VERSION_FILE}" STREQUAL "")
        message(WARNING "Build may fail: Could not determine ACL version (minimum required is ${ACL_MINIMUM_VERSION})")
    else()
        file(READ ${ACL_VERSION_FILE} ACL_VERSION_STRING)
        string(REGEX MATCH "v([0-9]+\\.[0-9]+)" ACL_VERSION ${ACL_VERSION_STRING})
        set(ACL_VERSION "${CMAKE_MATCH_1}")
        if (${ACL_VERSION} VERSION_EQUAL "0.0")
            # Unreleased ACL versions come with version string "v0.0-unreleased", and may not be compatible with oneDNN.
            # It is recommended to use the latest release of ACL.
            message(WARNING "Build may fail: Using unreleased ACL version (minimum required is ${ACL_MINIMUM_VERSION})")
        elseif(${ACL_VERSION} VERSION_LESS ${ACL_MINIMUM_VERSION})
            message(FATAL_ERROR "Detected ACL version ${ACL_VERSION}, but minimum required is ${ACL_MINIMUM_VERSION}")
        endif()
    endif()

    list(APPEND EXTRA_SHARED_LIBS ${ACL_LIBRARIES})

    include_directories(${ACL_INCLUDE_DIRS})

    message(STATUS "Arm Compute Library: ${ACL_LIBRARIES}")
    message(STATUS "Arm Compute Library headers: ${ACL_INCLUDE_DIRS}")

    add_definitions(-DDNNL_AARCH64_USE_ACL)
    set(CMAKE_CXX_STANDARD 14)
    set(CMAKE_CXX_EXTENSIONS "OFF")
endif()
