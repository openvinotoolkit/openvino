# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB SRC *.cpp)
file(GLOB HDR *.hpp)

set(TARGET_NAME fluid_test_computations)

add_library(${TARGET_NAME} STATIC ${SRC} ${HDR})

# Workaround to avoid warnings caused with bug in the avx512intrin.h of GCC5
if((CMAKE_CXX_COMPILER_ID STREQUAL "GNU") AND
   (CMAKE_CXX_COMPILER_VERSION VERSION_LESS_EQUAL 5.5))
    set_target_properties(${TARGET_NAME} PROPERTIES LINK_FLAGS_RELEASE "-Wno-error=maybe-uninitialized -Wno-maybe-uninitialized")
endif()

target_include_directories(${TARGET_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(${TARGET_NAME} PRIVATE openvino_gapi_preproc_s inference_engine fluid)

add_dependencies(${TARGET_NAME} openvino_gapi_preproc)
