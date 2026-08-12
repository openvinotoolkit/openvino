# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(NOT OV_GPU_RUNTIME_VULKAN_ENABLED)
    return()
endif()

find_package(Vulkan 1.3 REQUIRED)

set(TARGET_NAME "openvino_intel_gpu_vulkan_obj")

ov_gpu_add_backend_target(
    NAME ${TARGET_NAME}
    LINK_LIBRARIES Vulkan::Vulkan
)

ov_build_target_faster(${TARGET_NAME} PCH)
