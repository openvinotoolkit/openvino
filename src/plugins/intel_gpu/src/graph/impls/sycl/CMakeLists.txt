# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(NOT OV_GPU_WITH_SYCL)
    return()
endif()

set(TARGET_NAME "openvino_intel_gpu_sycl_obj")

ov_gpu_add_backend_target(
    NAME ${TARGET_NAME}
)

add_sycl_to_target(TARGET ${TARGET_NAME})
