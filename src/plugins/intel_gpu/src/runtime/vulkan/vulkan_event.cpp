// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vulkan_event.hpp"

namespace cldnn {
namespace vulkan {

vulkan_event::vulkan_event(bool signaled) : _signaled(signaled) {
    _set = signaled;
}

void vulkan_event::reset() {
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _signaled = false;
    }
    event::reset();
}

void vulkan_event::wait_impl() {
    std::unique_lock<std::mutex> lock(_mutex);
    _condition.wait(lock, [this] {
        return _signaled;
    });
}

void vulkan_event::set_impl() {
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _signaled = true;
    }
    _condition.notify_all();
}

bool vulkan_event::is_set_impl() {
    std::lock_guard<std::mutex> lock(_mutex);
    return _signaled;
}

}  // namespace vulkan
}  // namespace cldnn
