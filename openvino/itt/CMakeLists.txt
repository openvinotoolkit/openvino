# ******************************************************************************
# Copyright 2017-2020 Intel Corporation
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

set(TARGET_NAME itt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

file(GLOB_RECURSE SOURCES "src/*.cpp")

if(ENABLE_PROFILING_ITT)
    if(DEFINED INTEL_VTUNE_DIR OR DEFINED ENV{INTEL_VTUNE_DIR})
        find_package(ITT)
        if(NOT ITT_FOUND)
            message(WARNING "Profiling option enabled, but no ITT library was found under INTEL_VTUNE_DIR")
        endif()
    else()
        include(ExternalProject)
        set(ITTAPI_BINARY_DIR  ${CMAKE_CURRENT_BINARY_DIR}/ittapi/build)
        set(ITTAPI_SOURCE_DIR  ${CMAKE_CURRENT_BINARY_DIR}/ittapi/source)
        set(ITTNOTIFY_LIBRARY  ${ITTAPI_BINARY_DIR}/bin/${CMAKE_STATIC_LIBRARY_PREFIX}ittnotify${CMAKE_STATIC_LIBRARY_SUFFIX})
        ExternalProject_Add(
            ext_ittapi
            PREFIX ittapi
            GIT_REPOSITORY https://github.com/intel/ittapi.git
            GIT_TAG v3.18.6
            LOG_DOWNLOAD ON
            LOG_CONFIGURE ON
            LOG_BUILD ON
            INSTALL_COMMAND ""
            UPDATE_COMMAND ""
            CMAKE_GENERATOR ${CMAKE_GENERATOR}
            CMAKE_GENERATOR_PLATFORM ${CMAKE_GENERATOR_PLATFORM}
            CMAKE_GENERATOR_TOOLSET ${CMAKE_GENERATOR_TOOLSET}
            BINARY_DIR ${ITTAPI_BINARY_DIR}
            SOURCE_DIR ${ITTAPI_SOURCE_DIR}
            EXCLUDE_FROM_ALL TRUE
            BUILD_BYPRODUCTS ${ITTNOTIFY_LIBRARY})
        add_library(ittnotify INTERFACE)
        add_dependencies(ittnotify ittapi_external)
        target_link_libraries(ittnotify INTERFACE ${ITTNOTIFY_LIBRARY})
        target_include_directories(ittnotify INTERFACE ${ITTAPI_SOURCE_DIR}/include)
        openvino_developer_export_targets(ittnotify)
    endif()
endif()

add_library(${TARGET_NAME} STATIC ${SOURCES})

add_library(openvino::itt ALIAS ${TARGET_NAME})

if(TARGET ittnotify)
    target_link_libraries(${TARGET_NAME} PUBLIC ittnotify)
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(${TARGET_NAME} PRIVATE
        "-Wall"
    )
endif()

target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
