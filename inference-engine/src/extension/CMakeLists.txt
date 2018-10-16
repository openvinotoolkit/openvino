# Copyright (C) 2018 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
set(TARGET_NAME "cpu_extension")

file(GLOB_RECURSE SRC *.cpp)
file(GLOB_RECURSE HDR *.hpp)

if(WIN32)
    add_definitions(-DIMPLEMENT_INFERENCE_ENGINE_API)
endif()

include_directories (PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/common
        ${InferenceEngine_INCLUDE_DIRS}
)

enable_omp()

add_library(${TARGET_NAME} SHARED ${SRC} ${HDR})
target_link_libraries(${TARGET_NAME} ${InferenceEngine_LIBRARIES} ${intel_omp_lib})
target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_PDB_NAME ${TARGET_NAME})

set_target_cpu_flags(${TARGET_NAME})
