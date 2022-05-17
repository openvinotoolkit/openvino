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

file(GLOB SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    )
if(DNNL_ENABLE_JIT_PROFILING OR DNNL_ENABLE_ITT_TASKS)
    if(DNNL_TARGET_ARCH STREQUAL "AARCH64" OR DNNL_TARGET_ARCH STREQUAL "X64")
        file(GLOB ITT_SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/ittnotify/*.[ch]
            )
        list(APPEND SOURCES ${ITT_SOURCES})
    endif()
endif()

if(NOT DNNL_CPU_RUNTIME STREQUAL "THREADPOOL")
    # avoid building and linking empty objects
    list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/stream_threadpool.cpp")
endif()

set(OBJ_LIB ${DNNL_LIBRARY_NAME}_common)
add_library(${OBJ_LIB} OBJECT ${SOURCES})
set_property(GLOBAL APPEND PROPERTY DNNL_LIB_DEPS
    $<TARGET_OBJECTS:${OBJ_LIB}>)
enable_conditional_compilation4(${OBJ_LIB})
