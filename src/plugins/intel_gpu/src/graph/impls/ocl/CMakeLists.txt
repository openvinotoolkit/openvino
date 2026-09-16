# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "openvino_intel_gpu_ocl_obj${OV_GPU_SUFFIX}")

ov_gpu_add_backend_target(
    NAME ${TARGET_NAME}
    LINK_LIBRARIES OpenCL::OpenCL
)
ov_gpu_set_opencl_api_version_for(${TARGET_NAME})

ov_build_target_faster(${TARGET_NAME} PCH)

if(OV_GPU_RUNTIME_SYCL_ENABLED)
    # kernels_cache.cpp includes oneDNN headers that require SYCL backend support.
    # Without the SYCL compilation flag, the compiler will fail with "Unsupported compiler" error.
    add_sycl_to_target(TARGET ${TARGET_NAME} SOURCES kernels_cache.cpp)
endif()
