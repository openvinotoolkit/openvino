# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "openvino_intel_gpu_ocl_obj")

ov_gpu_add_backend_target(
    NAME ${TARGET_NAME}
)

ov_build_target_faster(${TARGET_NAME} PCH)
