# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(NOT OV_GPU_RUNTIME_OCL_ENABLED AND NOT OV_GPU_RUNTIME_ZE_ENABLED AND NOT OV_GPU_RUNTIME_SYCL_ENABLED)
    return()
endif()

set(TARGET_NAME "openvino_intel_gpu_ocl_obj")

ov_gpu_add_backend_target(
    NAME ${TARGET_NAME}
)

ov_build_target_faster(${TARGET_NAME} PCH)
