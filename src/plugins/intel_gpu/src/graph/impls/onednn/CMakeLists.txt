# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if (NOT ENABLE_ONEDNN_FOR_GPU)
    return()
endif()

set(TARGET_NAME "openvino_intel_gpu_onednn_obj${OV_GPU_SUFFIX}")

include(${CMAKE_CURRENT_LIST_DIR}/sources.cmake)

ov_gpu_add_backend_target(
    NAME ${TARGET_NAME}
    SOURCES ${GPU_ONEDNN_IMPL_SOURCES}
    LINK_LIBRARIES onednn_gpu_tgt${OV_GPU_SUFFIX}
)

ov_build_target_faster(${TARGET_NAME} PCH)
