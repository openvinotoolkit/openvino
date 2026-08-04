// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Vulkan backend common definitions for intel_gpu.
// Mirrors sycl_common.hpp from the SYCL backend.

#pragma once

#include <vulkan/vulkan.h>

#include <memory>
#include <stdexcept>
#include <string>

namespace cldnn {
namespace vk {

struct vulkan_error : std::runtime_error {
    explicit vulkan_error(const std::string& msg) : std::runtime_error(msg) {}
};

// Wraps a Vulkan call; throws vulkan_error on failure with result code.
inline void check(VkResult result, const char* what) {
    if (result != VK_SUCCESS) {
        throw vulkan_error(std::string(what) + " (VkResult=" + std::to_string(static_cast<int>(result)) + ")");
    }
}

#define VK_CALL(call, what) ::cldnn::vk::check((call), what)

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

}  // namespace vk
}  // namespace cldnn
