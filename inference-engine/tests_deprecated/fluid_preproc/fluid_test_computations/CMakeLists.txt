# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB SRC *.cpp)
file(GLOB HDR *.hpp)

add_library(fluid_test_computations SHARED ${SRC} ${HDR})

# Workaround to avoid warnings caused with bug in the avx512intrin.h of GCC5
if((CMAKE_CXX_COMPILER_ID STREQUAL "GNU") AND
   (CMAKE_CXX_COMPILER_VERSION VERSION_LESS_EQUAL 5.5))
    set_target_properties(fluid_test_computations PROPERTIES LINK_FLAGS_RELEASE "-Wno-error=maybe-uninitialized -Wno-maybe-uninitialized")
endif()

target_include_directories(fluid_test_computations PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(fluid_test_computations PRIVATE inference_engine_preproc_s inference_engine fluid)

target_compile_definitions(fluid_test_computations PRIVATE IMPLEMENT_FLUID_COMPUTATION_API)

add_dependencies(fluid_test_computations inference_engine_preproc)
