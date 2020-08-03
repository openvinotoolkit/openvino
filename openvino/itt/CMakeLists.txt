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
    find_package(ITT REQUIRED)
endif()

add_library(${TARGET_NAME} STATIC ${SOURCES})

add_library(openvino::itt ALIAS ${TARGET_NAME})

if (ENABLE_PROFILING_ITT AND INTEL_ITT_LIBS)
    target_include_directories(${TARGET_NAME} SYSTEM PRIVATE $<TARGET_PROPERTY:ittnotify,INTERFACE_INCLUDE_DIRECTORIES>)
    target_compile_definitions(${TARGET_NAME} PRIVATE $<TARGET_PROPERTY:ittnotify,INTERFACE_COMPILE_DEFINITIONS>)
    target_link_libraries(${TARGET_NAME} PUBLIC ${INTEL_ITT_LIBS} ${CMAKE_DL_LIBS})
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(${TARGET_NAME} PRIVATE
        "-Wall"
    )
endif()

target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
