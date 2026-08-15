// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vk_common.hpp"
#include "vk_types.hpp"

namespace ov::core::vulkan {
namespace cross_platform {

class vk_device {
public:
    vk_device(const instance_ptr& instance, VkPhysicalDevice physical_device);
    ~vk_device() = default;

    vk_device(const vk_device&) = delete;
    vk_device& operator=(const vk_device&) = delete;

    const device_info& get_info() const { return _info; }

    device_type type() const { return _info.dev_type; }
    runtime_types runtime_type() const { return runtime_types::vulkan; }

    VkInstance get_instance() const { return _instance.get(); }
    VkPhysicalDevice get_physical_device() const { return _physical_device; }
    VkDevice get_device() const { return _device.get(); }
    VkQueue get_queue() const { return _queue; }
    uint32_t get_queue_family_index() const { return _queue_family_index; }

    // Physical device class as reported by the driver (used by the
    // cross-platform routing: CPU-type devices serve the "CPU" device name).
    VkPhysicalDeviceType physical_device_type() const { return _physical_device_type; }
    bool is_cpu_device() const { return _physical_device_type == VK_PHYSICAL_DEVICE_TYPE_CPU; }

private:
    instance_ptr _instance;
    VkPhysicalDevice _physical_device;
    VkPhysicalDeviceType _physical_device_type = VK_PHYSICAL_DEVICE_TYPE_OTHER;
    device_ptr _device;
    VkQueue _queue = VK_NULL_HANDLE;
    uint32_t _queue_family_index = 0;
    device_info _info;
};

using vk_device_ptr = std::shared_ptr<vk_device>;

}  // namespace cross_platform
}  // namespace ov::core::vulkan
