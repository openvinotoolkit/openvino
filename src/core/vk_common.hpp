// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Vulkan backend common definitions for intel_gpu.
// Cross-platform helpers shared by all Vulkan backend components.
// Works with any Vulkan implementation (GPU vendors, SwiftShader, Lavapipe, etc).

#pragma once

#include <vulkan/vulkan.h>

#include "vk_platform.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace ov::core::vulkan {
namespace cross_platform {

struct vulkan_error : std::runtime_error {
    explicit vulkan_error(const std::string& msg) : std::runtime_error(msg) {}
};

// Wraps a Vulkan call; throws vulkan_error on failure with result code.
inline void check(VkResult result, const char* what) {
    if (result != VK_SUCCESS) {
        throw vulkan_error(std::string(what) + " (VkResult=" + std::to_string(static_cast<int>(result)) + ")");
    }
}

#define VK_CALL(call, what) ::ov::core::vulkan::cross_platform::check((call), what)

// RAII deleters for Vulkan handles. Shared instances (engines, devices,
// instances) are owned via shared_ptr with these deleters.
struct instance_deleter {
    void operator()(VkInstance instance) const {
        if (instance != VK_NULL_HANDLE) {
            vkDestroyInstance(instance, nullptr);
        }
    }
};
struct device_deleter {
    void operator()(VkDevice device) const {
        if (device != VK_NULL_HANDLE) {
            vkDestroyDevice(device, nullptr);
        }
    }
};

using instance_ptr = std::shared_ptr<VkInstance_T>;
using device_ptr = std::shared_ptr<VkDevice_T>;

inline instance_ptr make_instance(VkInstance instance) {
    return std::shared_ptr<VkInstance_T>(instance, instance_deleter{});
}

inline device_ptr make_device(VkDevice device) {
    return std::shared_ptr<VkDevice_T>(device, device_deleter{});
}

// Returns the first queue family that supports compute operations.
// Compute capability is required for OpenVINO inference on any device.
inline uint32_t find_compute_queue_family(VkPhysicalDevice physical_device) {
    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, nullptr);

    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, queue_families.data());

    for (uint32_t i = 0; i < queue_family_count; ++i) {
        if (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
            return i;
    }

    throw vulkan_error("vk: no compute-capable queue family found");
}

// Returns a memory type index with the given properties, or throws.
inline uint32_t find_memory_type(VkPhysicalDevice phys_device, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties mem_properties;
    vkGetPhysicalDeviceMemoryProperties(phys_device, &mem_properties);
    for (uint32_t i = 0; i < mem_properties.memoryTypeCount; ++i) {
        if ((mem_properties.memoryTypes[i].propertyFlags & properties) == properties)
            return i;
    }
    throw vulkan_error("vk: failed to find suitable memory type");
}

// Returns max workgroup invocations count for the physical device.
inline uint32_t get_max_compute_workgroup_invocations(VkPhysicalDevice physical_device) {
    VkPhysicalDeviceProperties props = {};
    vkGetPhysicalDeviceProperties(physical_device, &props);
    return props.limits.maxComputeWorkGroupInvocations;
}

}  // namespace cross_platform
}  // namespace ov::core::vulkan