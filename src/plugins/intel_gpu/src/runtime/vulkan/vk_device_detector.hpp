// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/device.hpp"
#include "vk_common.hpp"
#include "vk_device.hpp"

#include <memory>
#include <string>

namespace cldnn {
namespace vk {

class vk_device_detector {
public:
    vk_device_detector() = default;

    std::map<std::string, device::ptr> get_available_devices(void* user_context,
                                                             void* user_device,
                                                             int ctx_device_id,
                                                             int target_tile_id,
                                                             bool initialize_devices) {
        std::map<std::string, device::ptr> devices;

        if (user_context != nullptr)
            OPENVINO_THROW("[GPU] Vulkan device detector does not support user context");

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
        auto instance = vk::make_instance(raw_instance);

        uint32_t device_count = 0;
        VK_CALL(vkEnumeratePhysicalDevices(instance.get(), &device_count, nullptr), "vkEnumeratePhysicalDevices");
        if (device_count == 0)
            return devices;

        std::vector<VkPhysicalDevice> physical_devices(device_count);
        VK_CALL(vkEnumeratePhysicalDevices(instance.get(), &device_count, physical_devices.data()), "vkEnumeratePhysicalDevices");

        for (uint32_t i = 0; i < device_count; ++i) {
            auto device = std::make_shared<vk_device>(instance, physical_devices[i]);
            if (initialize_devices)
                device->initialize();
            devices[device->get_info().dev_name] = device;
        }
        return devices;
    }
};

}  // namespace vk
}  // namespace cldnn
