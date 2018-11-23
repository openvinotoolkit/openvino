# Copyright (C) 2018 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "ie_cpu_extension")

if (NOT(IE_MAIN_SOURCE_DIR))
    list (APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
    include(feature_defs)

    # to use C++11 if samples are built outside of IE repo
    set (CMAKE_CXX_STANDARD 11)
    set (CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
endif()

file(GLOB_RECURSE SRC *.cpp)
file(GLOB_RECURSE HDR *.hpp)

if(WIN32)
    add_definitions(-DIMPLEMENT_INFERENCE_ENGINE_API)
endif()

if (THREADING STREQUAL "TBB")
    add_definitions(-DIE_THREAD=IE_THREAD_TBB)
elseif (THREADING STREQUAL "OMP")
    add_definitions(-DIE_THREAD=IE_THREAD_OMP)
else()
    add_definitions(-DIE_THREAD=IE_THREAD_SEQ)
endif()

include_directories (PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/common
        ${InferenceEngine_INCLUDE_DIRS}
)

add_library(${TARGET_NAME} SHARED ${SRC} ${HDR})

set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME "cpu_extension")

if (THREADING STREQUAL "TBB")
    target_include_directories(${TARGET_NAME} PUBLIC ${TBB_INCLUDE_DIRS})
    target_link_libraries(${TARGET_NAME} debug ${TBB_LIBRARIES_RELEASE} optimized ${TBB_LIBRARIES_RELEASE})
elseif (THREADING STREQUAL "OMP")
    enable_omp()
    target_link_libraries(${TARGET_NAME} ${intel_omp_lib})
endif()

target_link_libraries(${TARGET_NAME} ${InferenceEngine_LIBRARIES})
target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_PDB_NAME ${TARGET_NAME})

set_target_cpu_flags(${TARGET_NAME})

if (IE_MAIN_SOURCE_DIR)
    export(TARGETS ${TARGET_NAME} NAMESPACE IE:: APPEND FILE "${CMAKE_BINARY_DIR}/targets.cmake")
endif()
