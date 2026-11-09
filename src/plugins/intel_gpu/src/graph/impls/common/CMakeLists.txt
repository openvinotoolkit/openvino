# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "openvino_intel_gpu_common_obj${OV_GPU_SUFFIX}")

include(${CMAKE_CURRENT_LIST_DIR}/sources.cmake)

ov_gpu_add_backend_target(
    NAME ${TARGET_NAME}
    SOURCES ${GPU_COMMON_IMPL_SOURCES}
)
