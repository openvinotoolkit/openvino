// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vk_common.hpp"

#include <memory>

namespace ov::core::vulkan {
namespace cross_platform {

// A Vulkan fence wrapped as a runtime event. A null fence (created by
// create_user_event-style paths) is treated as always-set.
class vk_event {
public:
    vk_event(VkDevice device, VkFence fence, bool owns_fence = false)
        : _device(device)
        , _fence(fence)
        , _owns_fence(owns_fence) {}

    ~vk_event();

    vk_event(const vk_event&) = delete;
    vk_event& operator=(const vk_event&) = delete;

    static std::shared_ptr<vk_event> create(VkDevice device);

    void wait() const;
    bool is_set() const;

    VkFence get_handle() const { return _fence; }

private:
    VkDevice _device;
    VkFence _fence;
    bool _owns_fence;
};

using vk_event_ptr = std::shared_ptr<vk_event>;

}  // namespace cross_platform
}  // namespace ov::core::vulkan
