// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/event.hpp"
#include "vk_common.hpp"

#include <memory>

namespace cldnn {
namespace vk {

class vk_event : public event {
public:
    vk_event(VkDevice device, VkFence fence, bool owns_fence = false)
        : _device(device)
        , _fence(fence)
        , _owns_fence(owns_fence) {}

    ~vk_event() override;

    static event::ptr create(VkDevice device);

    VkFence get_handle() const { return _fence; }

private:
    void wait_impl() override;
    void set_impl() override;
    bool is_set_impl() override;

    VkDevice _device;
    VkFence _fence;
    bool _owns_fence;
};

}  // namespace vk
}  // namespace cldnn
