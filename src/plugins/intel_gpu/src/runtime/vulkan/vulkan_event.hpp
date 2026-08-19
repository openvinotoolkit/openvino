// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vulkan/vulkan.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include "intel_gpu/runtime/event.hpp"

namespace cldnn {
namespace vulkan {

struct vulkan_profiling_query {
    VkQueryPool pool = VK_NULL_HANDLE;
    float timestamp_period_ns = 0.0f;
    uint32_t timestamp_valid_bits = 0;

    explicit operator bool() const {
        return pool != VK_NULL_HANDLE;
    }
};

struct vulkan_profiling_period {
    std::chrono::nanoseconds start;
    std::chrono::nanoseconds duration;
};

class vulkan_timeline_state {
public:
    explicit vulkan_timeline_state(VkDevice device);
    ~vulkan_timeline_state();

    vulkan_timeline_state(const vulkan_timeline_state&) = delete;
    vulkan_timeline_state& operator=(const vulkan_timeline_state&) = delete;

    void close();
    void wait(uint64_t value);
    bool is_complete(uint64_t value);

    VkDevice device() const {
        return _device;
    }

    VkSemaphore semaphore() const {
        return _semaphore;
    }

private:
    VkDevice _device = VK_NULL_HANDLE;
    VkSemaphore _semaphore = VK_NULL_HANDLE;
    std::mutex _mutex;
};

class vulkan_submission_state {
public:
    vulkan_submission_state(std::shared_ptr<vulkan_timeline_state> timeline,
                            VkQueue queue,
                            uint64_t stream_id,
                            uint64_t completion_value,
                            vulkan_profiling_query profiling = {});
    vulkan_submission_state(VkDevice device, VkQueue queue, VkFence fence, uint64_t stream_id, vulkan_profiling_query profiling = {});

    void wait();
    bool is_complete();
    void capture_profiling();
    std::optional<vulkan_profiling_period> get_profiling_period();
    bool belongs_to(VkDevice device, VkQueue queue) const;
    bool belongs_to_stream(VkDevice device, VkQueue queue, uint64_t stream_id) const;
    bool uses_fence() const {
        return _fence != VK_NULL_HANDLE;
    }
    VkFence release_fence();

private:
    void wait_locked();

    VkDevice _device = VK_NULL_HANDLE;
    std::shared_ptr<vulkan_timeline_state> _timeline;
    VkQueue _queue = VK_NULL_HANDLE;
    VkFence _fence = VK_NULL_HANDLE;
    uint64_t _stream_id = 0;
    uint64_t _completion_value = 0;
    vulkan_profiling_query _profiling;
    std::optional<vulkan_profiling_period> _profiling_period;
    bool _profiling_captured = false;
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
    bool get_profiling_info_impl(std::list<instrumentation::profiling_interval>& info) override;

    std::mutex _mutex;
    std::condition_variable _condition;
    bool _signaled = false;
    std::shared_ptr<vulkan_submission_state> _submission;
};

class vulkan_events final : public event {
public:
    explicit vulkan_events(std::vector<event::ptr> events);

private:
    void wait_impl() override;
    void set_impl() override;
    bool is_set_impl() override;
    bool get_profiling_info_impl(std::list<instrumentation::profiling_interval>& info) override;

    std::vector<event::ptr> _events;
};

}  // namespace vulkan
}  // namespace cldnn
