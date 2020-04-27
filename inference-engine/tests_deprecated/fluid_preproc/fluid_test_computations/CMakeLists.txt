# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB SRC *.cpp)
file(GLOB HDR *.hpp)

add_library(fluid_test_computations SHARED ${SRC} ${HDR})

target_include_directories(fluid_test_computations PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(fluid_test_computations PRIVATE inference_engine_preproc_s inference_engine fluid)

target_compile_definitions(fluid_test_computations PRIVATE IMPLEMENT_FLUID_COMPUTATION_API)
