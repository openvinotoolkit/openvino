# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(NOT OV_GPU_WITH_SYCL)
    return()
endif()

set(TARGET_NAME "openvino_intel_gpu_sycl_obj")

set(SOURCES_OCL_RT "${CMAKE_CURRENT_SOURCE_DIR}/impl_example.cpp")

if(GPU_RT_TYPE STREQUAL "SYCL")
    set(EXCLUDED_SOURCE_PATHS ${SOURCES_OCL_RT})
else()
    file(GLOB EXCLUDED_SOURCE_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
    list(REMOVE_ITEM EXCLUDED_SOURCE_PATHS ${SOURCES_OCL_RT})
endif()

ov_gpu_add_backend_target(
    NAME ${TARGET_NAME}
    BYPASS
        EXCLUDED_SOURCE_PATHS ${EXCLUDED_SOURCE_PATHS}
)

add_sycl_to_target(TARGET ${TARGET_NAME})
