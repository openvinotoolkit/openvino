# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ov_add_test_target(
    NAME ov_gpu_vulkan_runtime_smoke
    SOURCES runtime_smoke.cpp
    DEPENDENCIES openvino_intel_gpu_plugin
    LINK_LIBRARIES openvino::runtime gtest)

# Android's system loader does not use desktop Vulkan driver manifests.
if(NOT ANDROID)
    add_test(NAME ov_gpu_vulkan_no_driver_smoke COMMAND ov_gpu_vulkan_runtime_smoke --expect-no-driver)
    set(no_driver_manifest "${CMAKE_CURRENT_BINARY_DIR}/no-compatible-vulkan-driver.json")
    set_tests_properties(ov_gpu_vulkan_no_driver_smoke PROPERTIES
        ENVIRONMENT "VK_DRIVER_FILES=${no_driver_manifest};VK_ICD_FILENAMES=${no_driver_manifest}")
endif()

ov_add_test_target(
    NAME ov_gpu_vulkan_memory_smoke
    SOURCES memory_smoke.cpp ${GPU_PLUGIN_VULKAN_CONTEXT_SOURCES}
    LINK_LIBRARIES openvino_intel_gpu_runtime openvino::reference openvino::runtime openvino::runtime::dev Vulkan::Vulkan gtest_main)
ov_gpu_set_runtime_interface_for(ov_gpu_vulkan_memory_smoke RUNTIME VULKAN)
ov_set_threading_interface_for(ov_gpu_vulkan_memory_smoke)
