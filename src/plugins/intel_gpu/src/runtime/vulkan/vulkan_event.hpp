// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vulkan/vulkan.h>

#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>

#include "intel_gpu/runtime/event.hpp"

namespace cldnn {
namespace vulkan {

class vulkan_submission_state {
public:
    vulkan_submission_state(VkDevice device, VkQueue queue, VkFence fence, uint64_t stream_id);

    void wait();
    bool is_complete();
    bool belongs_to(VkDevice device, VkQueue queue) const;
    bool belongs_to_stream(VkDevice device, VkQueue queue, uint64_t stream_id) const;
    VkFence release_fence();

private:
    VkDevice _device = VK_NULL_HANDLE;
    VkQueue _queue = VK_NULL_HANDLE;
    VkFence _fence = VK_NULL_HANDLE;
    uint64_t _stream_id = 0;
    std::mutex _mutex;
    bool _completed = false;
};

class vulkan_event final : public event {
public:
    explicit vulkan_event(bool signaled = false);
    explicit vulkan_event(std::shared_ptr<vulkan_submission_state> submission);

    void reset() override;
    bool is_device_submission(VkDevice device, VkQueue queue) const;
    bool is_stream_submission(VkDevice device, VkQueue queue, uint64_t stream_id) const;

private:
    void wait_impl() override;
    void set_impl() override;
    bool is_set_impl() override;

    std::mutex _mutex;
    std::condition_variable _condition;
    bool _signaled = false;
    std::shared_ptr<vulkan_submission_state> _submission;
};

}  // namespace vulkan
}  // namespace cldnn
