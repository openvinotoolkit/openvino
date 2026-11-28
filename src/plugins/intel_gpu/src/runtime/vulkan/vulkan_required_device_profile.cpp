// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vulkan_required_device_profile.hpp"

namespace cldnn {
namespace vulkan {

bool vulkan_required_device_profile::is_supported() const {
    return incompatibility_reason().empty();
}

std::string vulkan_required_device_profile::incompatibility_reason() const {
    if (properties.deviceType != VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU && properties.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
        return "the physical device is not a discrete or integrated GPU";
    }
    if (properties.apiVersion < required_api_version) {
        return "Vulkan 1.3 is not supported";
    }
    if (synchronization2 != VK_TRUE) {
        return "buffer transfers require synchronization2";
    }
    return {};
}

vulkan_required_device_profile query_vulkan_required_device_profile(VkPhysicalDevice physical_device) {
    vulkan_required_device_profile profile;
    vkGetPhysicalDeviceProperties(physical_device, &profile.properties);
    if ((profile.properties.deviceType != VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU && profile.properties.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) ||
        profile.properties.apiVersion < required_api_version) {
        return profile;
    }

    VkPhysicalDeviceSynchronization2Features synchronization2{};
    synchronization2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES;

    VkPhysicalDeviceFeatures2 features{};
    features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features.pNext = &synchronization2;
    vkGetPhysicalDeviceFeatures2(physical_device, &features);

    profile.synchronization2 = synchronization2.synchronization2;
    return profile;
}

}  // namespace vulkan
}  // namespace cldnn
