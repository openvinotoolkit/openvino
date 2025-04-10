# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "openvino_intel_gpu_cpu_obj")

ov_gpu_add_backend_target(
    NAME ${TARGET_NAME}
    LINK_LIBRARIES openvino::reference
)

if(ENABLE_SSE42)
  ov_sse42_optimization_flags(sse4_2_flags)
  set_source_files_properties(detection_output.cpp PROPERTIES
    COMPILE_FLAGS "${sse4_2_flags}"
    COMPILE_DEFINITIONS "HAVE_SSE")
endif()
