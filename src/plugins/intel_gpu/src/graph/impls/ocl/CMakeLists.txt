# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "openvino_intel_gpu_ocl_obj")

set(OCL_EXCLUDE "")
if(NOT ENABLE_EXPERIMENTAL_OPSET)
    list(APPEND OCL_EXCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/scaled_shifted_clamp_experimental.cpp)
endif()

ov_gpu_add_backend_target(
    NAME ${TARGET_NAME}
    BYPASS EXCLUDED_SOURCE_PATHS ${OCL_EXCLUDE}
)

ov_build_target_faster(${TARGET_NAME} PCH)
