# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if (NOT ENABLE_ONEDNN_FOR_GPU)
    return()
endif()

set(TARGET_NAME "openvino_intel_gpu_onednn_obj")

ov_gpu_add_backend_target(
    NAME ${TARGET_NAME}
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    LINK_LIBRARIES
        PUBLIC
            onednn_gpu_tgt
)
