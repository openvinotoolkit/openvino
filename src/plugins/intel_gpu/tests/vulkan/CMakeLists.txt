# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_executable(ov_gpu_vulkan_runtime_smoke runtime_smoke.cpp)
add_dependencies(ov_gpu_vulkan_runtime_smoke openvino_intel_gpu_plugin)
target_link_libraries(ov_gpu_vulkan_runtime_smoke PRIVATE openvino::runtime)

enable_testing()
add_test(NAME ov_gpu_vulkan_runtime_smoke COMMAND ov_gpu_vulkan_runtime_smoke)
