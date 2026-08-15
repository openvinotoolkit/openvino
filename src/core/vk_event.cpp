// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vk_event.hpp"

#include <cstdint>

namespace ov::core::vulkan {
namespace cross_platform {

vk_event_ptr vk_event::create(VkDevice device) {
    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_info.flags = 0;

    VkFence fence = VK_NULL_HANDLE;
    VK_CALL(vkCreateFence(device, &fence_info, nullptr, &fence), "vkCreateFence");

    return std::make_shared<vk_event>(device, fence, true);
}

vk_event::~vk_event() {
    if (_owns_fence && _fence != VK_NULL_HANDLE) {
        vkDestroyFence(_device, _fence, nullptr);
    }
}

void vk_event::wait() const {
    if (_fence == VK_NULL_HANDLE)
        return;

    VK_CALL(vkWaitForFences(_device, 1, &_fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
}

bool vk_event::is_set() const {
    if (_fence == VK_NULL_HANDLE)
        return true;

    return vkGetFenceStatus(_device, _fence) == VK_SUCCESS;
}

}  // namespace cross_platform
}  // namespace ov::core::vulkan
