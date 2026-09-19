// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vulkan_device_detector.hpp"

#include <algorithm>
#include <exception>
#include <optional>
#include <ostream>
#include <tuple>
#include <vector>

#include "intel_gpu/runtime/debug_configuration.hpp"
#include "openvino/core/except.hpp"
#include "vulkan_api.hpp"
#include "vulkan_device.hpp"
#include "vulkan_instance.hpp"
#include "vulkan_required_device_profile.hpp"

namespace cldnn {
namespace vulkan {
namespace {

void check_vk_result(VkResult result, const char* operation) {
    OPENVINO_ASSERT(result == VK_SUCCESS, "[GPU][Vulkan] ", operation, " failed with VkResult ", static_cast<int>(result));
}

std::optional<uint32_t> find_compute_queue_family(VkPhysicalDevice physical_device) {
    uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &family_count, nullptr);
    std::vector<VkQueueFamilyProperties> families(family_count);
    if (family_count > 0) {
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &family_count, families.data());
    }

    std::optional<uint32_t> selected;
    for (uint32_t index = 0; index < family_count; ++index) {
        const auto flags = families[index].queueFlags;
        if (families[index].queueCount == 0 || (flags & VK_QUEUE_COMPUTE_BIT) == 0) {
            continue;
        }
        if (!selected.has_value()) {
            selected = index;
            continue;
        }

        const auto& candidate = families[index];
        const auto& current = families[*selected];
        const auto candidate_granularity =
            std::tie(candidate.minImageTransferGranularity.width, candidate.minImageTransferGranularity.height, candidate.minImageTransferGranularity.depth);
        const auto current_granularity =
            std::tie(current.minImageTransferGranularity.width, current.minImageTransferGranularity.height, current.minImageTransferGranularity.depth);
        const bool better = candidate.queueCount > current.queueCount ||
                            (candidate.queueCount == current.queueCount && candidate.timestampValidBits > current.timestampValidBits) ||
                            (candidate.queueCount == current.queueCount && candidate.timestampValidBits == current.timestampValidBits &&
                             candidate_granularity < current_granularity);
        if (better) {
            selected = index;
        }
    }
    return selected;
}

}  // namespace

std::map<std::string, device::ptr> vulkan_device_detector::get_available_devices(void* user_context,
                                                                                 void* user_device,
                                                                                 int ctx_device_id,
                                                                                 int target_tile_id,
                                                                                 bool initialize_devices) const {
    OPENVINO_ASSERT(user_context == nullptr && user_device == nullptr, "[GPU][Vulkan] External Vulkan context/device import is not supported");
    OPENVINO_ASSERT(ctx_device_id == 0, "[GPU][Vulkan] A context device index is valid only with an imported context");
    OPENVINO_ASSERT(target_tile_id == -1, "[GPU][Vulkan] Sub-device/tile selection is not supported");

    const auto instance = vulkan_instance::create();
    uint32_t physical_device_count = 0;
    check_vk_result(vkEnumeratePhysicalDevices(instance->get(), &physical_device_count, nullptr), "vkEnumeratePhysicalDevices(count)");

    std::vector<VkPhysicalDevice> physical_devices(physical_device_count);
    if (physical_device_count > 0) {
        check_vk_result(vkEnumeratePhysicalDevices(instance->get(), &physical_device_count, physical_devices.data()), "vkEnumeratePhysicalDevices(list)");
    }

    std::vector<device::ptr> devices;
    for (const VkPhysicalDevice physical_device : physical_devices) {
        const auto required_profile = query_vulkan_required_device_profile(physical_device);
        const auto& properties = required_profile.properties;
        if (!required_profile.is_supported()) {
            GPU_DEBUG_LOG << "Vulkan device '" << properties.deviceName << "' is skipped because " << required_profile.incompatibility_reason() << std::endl;
            continue;
        }

        const auto compute_queue_family = find_compute_queue_family(physical_device);
        if (!compute_queue_family.has_value()) {
            GPU_DEBUG_LOG << "Vulkan device '" << properties.deviceName << "' is skipped because it has no compute-capable queue family" << std::endl;
            continue;
        }

        try {
            devices.push_back(std::make_shared<vulkan_device>(instance, physical_device, *compute_queue_family, initialize_devices));
        } catch (const std::exception& error) {
            GPU_DEBUG_LOG << "Vulkan device '" << properties.deviceName << "' initialization failed: " << error.what() << std::endl;
        }
    }

    std::sort(devices.begin(), devices.end(), [](const auto& lhs, const auto& rhs) {
        const auto& lhs_info = lhs->get_info();
        const auto& rhs_info = rhs->get_info();
        return std::tie(lhs_info.vendor_id, lhs_info.device_id, lhs_info.uuid.uuid, lhs_info.dev_name) <
               std::tie(rhs_info.vendor_id, rhs_info.device_id, rhs_info.uuid.uuid, rhs_info.dev_name);
    });

    std::map<std::string, device::ptr> result;
    for (size_t index = 0; index < devices.size(); ++index) {
        result.emplace(std::to_string(index), devices[index]);
    }
    return result;
}

}  // namespace vulkan
}  // namespace cldnn
