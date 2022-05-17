#===============================================================================
# Copyright 2019-2021 Intel Corporation
# Copyright 2020 Arm Ltd. and affiliates
# Copyright 2021 FUJITSU LIMITED
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

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.[ch]pp)
file(GLOB_RECURSE SOURCES_EXTRA
    ${CMAKE_CURRENT_SOURCE_DIR}/gemm/*.[ch]pp
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul/*.[ch]pp
    ${CMAKE_CURRENT_SOURCE_DIR}/reorder/*.[ch]pp
    ${CMAKE_CURRENT_SOURCE_DIR}/rnn/*.[ch]pp
    )

foreach(SOURCE_FILE ${SOURCES_EXTRA})
    list(APPEND SOURCES "${SOURCE_FILE}")
endforeach()

if((DNNL_TARGET_ARCH STREQUAL "X64") OR (DNNL_TARGET_ARCH STREQUAL "AARCH64"))
    file(GLOB_RECURSE SOURCES_JIT_UTILS
        ${CMAKE_CURRENT_SOURCE_DIR}/jit_utils/*.[ch]
        ${CMAKE_CURRENT_SOURCE_DIR}/jit_utils/*.[ch]pp
    )
    foreach(SOURCE_FILE ${SOURCES_JIT_UTILS})
        list(APPEND SOURCES "${SOURCE_FILE}")
    endforeach()
endif()

if(NOT DNNL_ENABLE_JIT_PROFILING)
    # XXX: the profiling interface will still be built and present
    add_definitions_with_host_compiler(-DDNNL_ENABLE_JIT_PROFILING=0)
    # Don't enable support for linux_perf and VTune Amplifier
    if((DNNL_TARGET_ARCH STREQUAL "X64") OR (DNNL_TARGET_ARCH STREQUAL "AARCH64"))
        list(REMOVE_ITEM SOURCES_EXTRA
            "${CMAKE_CURRENT_SOURCE_DIR}/jit_utils/jitprofiling/jitprofiling.c"
            "${CMAKE_CURRENT_SOURCE_DIR}/jit_utils/linux_perf/linux_perf.cpp"
            )
    endif()
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    # to make computations more stable and to align the jitted code
    # with the reference one use precise division and square root
    # by default
    file(GLOB FILES_REQUIRED_PREC_SQRT
        ${CMAKE_CURRENT_SOURCE_DIR}/*normalization*.cpp
        )
    file(GLOB FILES_REQUIRED_PREC_DIV
        ${CMAKE_CURRENT_SOURCE_DIR}/*resampling*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/*normalization*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ref_eltwise.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/nhwc_pooling.cpp
        )
    if(WIN32)
        set_source_files_properties(${FILES_REQUIRED_PREC_SQRT}
            PROPERTIES COMPILE_FLAGS "/Qprec-sqrt")
        set_source_files_properties(${FILES_REQUIRED_PREC_DIV}
            PROPERTIES COMPILE_FLAGS "/Qprec-div")
    else()
        set_source_files_properties(${FILES_REQUIRED_PREC_SQRT}
            PROPERTIES COMPILE_FLAGS "-prec-sqrt")
        set_source_files_properties(${FILES_REQUIRED_PREC_DIV}
            PROPERTIES COMPILE_FLAGS "-prec-div")
    endif()
endif()

if(DNNL_WITH_SYCL)
    set(FILES_WITHNO_OPT)
    foreach(src in ${SOURCES})
        string(REGEX MATCH ".*jit.*kern.cpp" match ${src})
        if(match)
            list(APPEND FILES_WITHNO_OPT ${src})
        endif()
    endforeach()

    set_source_files_properties(${FILES_WITHNO_OPT}
        PROPERTIES COMPILE_FLAGS "-O0")
endif()

set(OBJ_LIB ${DNNL_LIBRARY_NAME}_cpu)
add_library(${OBJ_LIB} OBJECT ${SOURCES})
set_property(GLOBAL APPEND PROPERTY DNNL_LIB_DEPS
    $<TARGET_OBJECTS:${OBJ_LIB}>)
enable_conditional_compilation4(${OBJ_LIB})

if (DNNL_TARGET_ARCH STREQUAL "X64")
    add_subdirectory(x64)
endif()
if (DNNL_TARGET_ARCH STREQUAL "AARCH64")
    add_subdirectory(aarch64)
endif()
