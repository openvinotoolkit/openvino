// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <condition_variable>
#include <mutex>

#include "intel_gpu/runtime/event.hpp"

namespace cldnn {
namespace vulkan {

class vulkan_event final : public event {
public:
    explicit vulkan_event(bool signaled = false);

    void reset() override;

private:
    void wait_impl() override;
    void set_impl() override;
    bool is_set_impl() override;

    std::mutex _mutex;
    std::condition_variable _condition;
    bool _signaled = false;
};

}  // namespace vulkan
}  // namespace cldnn
