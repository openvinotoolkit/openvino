// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "vulkan_api.hpp"

namespace cldnn {
namespace vulkan {

struct vulkan_required_device_profile {
    VkPhysicalDeviceProperties properties{};
    VkBool32 synchronization2 = VK_FALSE;

    bool is_supported() const;
    std::string incompatibility_reason() const;
};

vulkan_required_device_profile query_vulkan_required_device_profile(VkPhysicalDevice physical_device);

}  // namespace vulkan
}  // namespace cldnn
