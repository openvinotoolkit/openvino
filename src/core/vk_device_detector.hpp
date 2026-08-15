// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "runtime/except.hpp"
#include "vk_common.hpp"
#include "vk_device.hpp"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace ov::core::vulkan {
namespace cross_platform {

class vk_device_detector {
public:
    vk_device_detector() = default;

    // Creates one vk_device per physical device, keyed by index ("0", "1", ...).
    std::map<std::string, vk_device_ptr> get_available_devices() {
        std::map<std::string, vk_device_ptr> devices;

        VkApplicationInfo app_info = {};
        app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app_info.pApplicationName = "OpenVINO";
        app_info.applicationVersion = 1;
        app_info.apiVersion = VK_API_VERSION_1_3;

        VkInstanceCreateInfo instance_info = {};
        instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        instance_info.pApplicationInfo = &app_info;

        VkInstance raw_instance = VK_NULL_HANDLE;
        VK_CALL(vkCreateInstance(&instance_info, nullptr, &raw_instance), "vkCreateInstance");
        auto instance = make_instance(raw_instance);

        uint32_t device_count = 0;
        VK_CALL(vkEnumeratePhysicalDevices(instance.get(), &device_count, nullptr), "vkEnumeratePhysicalDevices");
        if (device_count == 0)
            return devices;

        std::vector<VkPhysicalDevice> physical_devices(device_count);
        VK_CALL(vkEnumeratePhysicalDevices(instance.get(), &device_count, physical_devices.data()), "vkEnumeratePhysicalDevices");

        for (uint32_t i = 0; i < device_count; ++i) {
            auto device = std::make_shared<vk_device>(instance, physical_devices[i]);
            devices[std::to_string(i)] = device;
        }
        return devices;
    }
};

}  // namespace cross_platform
}  // namespace ov::core::vulkan
