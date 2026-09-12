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
    if (storage_buffer_8_bit_access != VK_TRUE) {
        return "the common Eltwise byte-address ABI requires storageBuffer8BitAccess";
    }
    if (synchronization2 != VK_TRUE) {
        return "exact buffer hazard tracking requires synchronization2";
    }
    if (timeline_semaphore != VK_TRUE) {
        return "asynchronous batch completion requires timelineSemaphore";
    }
    if (maintenance4 != VK_TRUE) {
        return "dynamic local work-group specialization requires maintenance4";
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

    VkPhysicalDevice8BitStorageFeatures storage8{};
    storage8.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES;
    VkPhysicalDeviceSynchronization2Features synchronization2{};
    synchronization2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES;
    VkPhysicalDeviceTimelineSemaphoreFeatures timeline{};
    timeline.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
    VkPhysicalDeviceMaintenance4Features maintenance4{};
    maintenance4.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES;
    storage8.pNext = &synchronization2;
    synchronization2.pNext = &timeline;
    timeline.pNext = &maintenance4;

    VkPhysicalDeviceFeatures2 features{};
    features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features.pNext = &storage8;
    vkGetPhysicalDeviceFeatures2(physical_device, &features);

    profile.storage_buffer_8_bit_access = storage8.storageBuffer8BitAccess;
    profile.synchronization2 = synchronization2.synchronization2;
    profile.timeline_semaphore = timeline.timelineSemaphore;
    profile.maintenance4 = maintenance4.maintenance4;
    return profile;
}

}  // namespace vulkan
}  // namespace cldnn
