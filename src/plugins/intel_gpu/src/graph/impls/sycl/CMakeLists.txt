# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(NOT OV_GPU_WITH_SYCL)
    return()
endif()

set(TARGET_NAME "openvino_intel_gpu_sycl_obj${OV_GPU_SUFFIX}")

include(${CMAKE_CURRENT_LIST_DIR}/sources.cmake)

set(SYCL_BACKEND_SOURCES ${GPU_SYCL_HEADERS})
if(OV_GPU_RT STREQUAL "SYCL")
    list(APPEND SYCL_BACKEND_SOURCES ${GPU_SYCL_ELTWISE_SOURCE})
else()
    list(APPEND SYCL_BACKEND_SOURCES ${GPU_SYCL_IMPL_EXAMPLE_SOURCE})
endif()

ov_gpu_add_backend_target(
    NAME ${TARGET_NAME}
    SOURCES ${SYCL_BACKEND_SOURCES}
)

add_sycl_to_target(TARGET ${TARGET_NAME})
