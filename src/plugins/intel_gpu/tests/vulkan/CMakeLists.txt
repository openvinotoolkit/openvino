# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ov_add_target(
    NAME ov_gpu_vulkan_runtime_smoke
    TYPE EXECUTABLE
    SOURCES runtime_smoke.cpp
    DEPENDENCIES openvino_intel_gpu_plugin
    LINK_LIBRARIES openvino::runtime)

ov_add_target(
    NAME ov_gpu_vulkan_memory_smoke
    TYPE EXECUTABLE
    SOURCES
        memory_smoke.cpp
        ../../src/plugin/common_utils.cpp
        ../../src/plugin/remote_context.cpp
        ../../src/plugin/remote_tensor.cpp
        ../../src/plugin/usm_host_tensor.cpp
    LINK_LIBRARIES openvino_intel_gpu_graph openvino::runtime::dev Vulkan::Vulkan)
ov_gpu_set_runtime_interface_for(ov_gpu_vulkan_memory_smoke RUNTIME VULKAN)
ov_set_threading_interface_for(ov_gpu_vulkan_memory_smoke)

enable_testing()
add_test(NAME ov_gpu_vulkan_runtime_smoke COMMAND ov_gpu_vulkan_runtime_smoke)
add_test(NAME ov_gpu_vulkan_memory_smoke COMMAND ov_gpu_vulkan_memory_smoke)
