// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vulkan_event.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <utility>

#include "openvino/core/except.hpp"

namespace cldnn {
namespace vulkan {

vulkan_timeline_state::vulkan_timeline_state(VkDevice device) : _device(device) {
    OPENVINO_ASSERT(_device != VK_NULL_HANDLE, "[GPU][Vulkan] A completion timeline requires a valid Vulkan device");

    VkSemaphoreTypeCreateInfo type_info{};
    type_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    type_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    type_info.initialValue = 0;

    VkSemaphoreCreateInfo semaphore_info{};
    semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    semaphore_info.pNext = &type_info;
    const auto result = vkCreateSemaphore(_device, &semaphore_info, nullptr, &_semaphore);
    OPENVINO_ASSERT(result == VK_SUCCESS, "[GPU][Vulkan] vkCreateSemaphore(timeline) failed with VkResult ", static_cast<int>(result));
}

vulkan_timeline_state::~vulkan_timeline_state() {
    close();
}

void vulkan_timeline_state::close() {
    std::lock_guard<std::mutex> lock(_mutex);
    if (_semaphore != VK_NULL_HANDLE) {
        vkDestroySemaphore(_device, _semaphore, nullptr);
        _semaphore = VK_NULL_HANDLE;
    }
}

void vulkan_timeline_state::wait(uint64_t value) {
    std::lock_guard<std::mutex> lock(_mutex);
    OPENVINO_ASSERT(_semaphore != VK_NULL_HANDLE, "[GPU][Vulkan] Cannot wait on a closed completion timeline");
    VkSemaphoreWaitInfo wait_info{};
    wait_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    wait_info.semaphoreCount = 1;
    wait_info.pSemaphores = &_semaphore;
    wait_info.pValues = &value;
    const auto result = vkWaitSemaphores(_device, &wait_info, UINT64_MAX);
    OPENVINO_ASSERT(result == VK_SUCCESS, "[GPU][Vulkan] vkWaitSemaphores failed with VkResult ", static_cast<int>(result));
}

bool vulkan_timeline_state::is_complete(uint64_t value) {
    std::lock_guard<std::mutex> lock(_mutex);
    OPENVINO_ASSERT(_semaphore != VK_NULL_HANDLE, "[GPU][Vulkan] Cannot query a closed completion timeline");
    uint64_t completed_value = 0;
    const auto result = vkGetSemaphoreCounterValue(_device, _semaphore, &completed_value);
    OPENVINO_ASSERT(result == VK_SUCCESS, "[GPU][Vulkan] vkGetSemaphoreCounterValue failed with VkResult ", static_cast<int>(result));
    return completed_value >= value;
}

vulkan_submission_state::vulkan_submission_state(std::shared_ptr<vulkan_timeline_state> timeline,
                                                 VkQueue queue,
                                                 uint64_t stream_id,
                                                 uint64_t completion_value,
                                                 vulkan_profiling_query profiling)
    : _timeline(std::move(timeline)),
      _queue(queue),
      _stream_id(stream_id),
      _completion_value(completion_value),
      _profiling(profiling) {
    OPENVINO_ASSERT(_timeline != nullptr && _queue != VK_NULL_HANDLE && _stream_id != 0 && _completion_value != 0,
                    "[GPU][Vulkan] A device submission requires valid Vulkan handles");
}

vulkan_submission_state::vulkan_submission_state(VkDevice device, VkQueue queue, VkFence fence, uint64_t stream_id, vulkan_profiling_query profiling)
    : _device(device),
      _queue(queue),
      _fence(fence),
      _stream_id(stream_id),
      _profiling(profiling) {
    OPENVINO_ASSERT(_device != VK_NULL_HANDLE && _queue != VK_NULL_HANDLE && _fence != VK_NULL_HANDLE && _stream_id != 0,
                    "[GPU][Vulkan] A fence submission requires valid Vulkan handles");
}

void vulkan_submission_state::wait() {
    std::lock_guard<std::mutex> lock(_mutex);
    wait_locked();
}

void vulkan_submission_state::wait_locked() {
    if (_completed) {
        return;
    }

    if (_timeline != nullptr) {
        _timeline->wait(_completion_value);
    } else {
        const auto result = vkWaitForFences(_device, 1, &_fence, VK_TRUE, UINT64_MAX);
        OPENVINO_ASSERT(result == VK_SUCCESS, "[GPU][Vulkan] vkWaitForFences failed with VkResult ", static_cast<int>(result));
    }
    _completed = true;
}

void vulkan_submission_state::capture_profiling() {
    std::lock_guard<std::mutex> lock(_mutex);
    if (_profiling_captured || !_profiling) {
        return;
    }

    wait_locked();
    std::array<uint64_t, 2> timestamps{};
    const auto result = vkGetQueryPoolResults(_timeline != nullptr ? _timeline->device() : _device,
                                              _profiling.pool,
                                              0,
                                              static_cast<uint32_t>(timestamps.size()),
                                              sizeof(timestamps),
                                              timestamps.data(),
                                              sizeof(timestamps.front()),
                                              VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    OPENVINO_ASSERT(result == VK_SUCCESS, "[GPU][Vulkan] vkGetQueryPoolResults(timestamp) failed with VkResult ", static_cast<int>(result));

    const auto valid_mask = _profiling.timestamp_valid_bits == 64 ? std::numeric_limits<uint64_t>::max() : (uint64_t{1} << _profiling.timestamp_valid_bits) - 1;
    const auto start_ticks = timestamps[0] & valid_mask;
    const auto duration_ticks = (timestamps[1] - timestamps[0]) & valid_mask;
    const auto to_nanoseconds = [this](uint64_t ticks) {
        const auto nanoseconds = static_cast<long double>(ticks) * _profiling.timestamp_period_ns;
        OPENVINO_ASSERT(nanoseconds <= static_cast<long double>(std::numeric_limits<int64_t>::max()),
                        "[GPU][Vulkan] Profiling timestamp exceeds the OpenVINO duration range");
        return std::chrono::nanoseconds(static_cast<int64_t>(std::llround(nanoseconds)));
    };
    _profiling_period = vulkan_profiling_period{to_nanoseconds(start_ticks), to_nanoseconds(duration_ticks)};
    _profiling_captured = true;
}

std::optional<vulkan_profiling_period> vulkan_submission_state::get_profiling_period() {
    capture_profiling();
    std::lock_guard<std::mutex> lock(_mutex);
    return _profiling_period;
}

bool vulkan_submission_state::is_complete() {
    std::lock_guard<std::mutex> lock(_mutex);
    if (_completed) {
        return true;
    }

    if (_timeline != nullptr) {
        if (!_timeline->is_complete(_completion_value)) {
            return false;
        }
    } else {
        const auto result = vkGetFenceStatus(_device, _fence);
        if (result == VK_NOT_READY) {
            return false;
        }
        OPENVINO_ASSERT(result == VK_SUCCESS, "[GPU][Vulkan] vkGetFenceStatus failed with VkResult ", static_cast<int>(result));
    }
    _completed = true;
    return true;
}

bool vulkan_submission_state::belongs_to(VkDevice device, VkQueue queue) const {
    const VkDevice submission_device = _timeline != nullptr ? _timeline->device() : _device;
    return submission_device == device && _queue == queue;
}

bool vulkan_submission_state::belongs_to_stream(VkDevice device, VkQueue queue, uint64_t stream_id) const {
    return belongs_to(device, queue) && _stream_id == stream_id;
}

VkFence vulkan_submission_state::release_fence() {
    std::lock_guard<std::mutex> lock(_mutex);
    OPENVINO_ASSERT(_completed && _fence != VK_NULL_HANDLE, "[GPU][Vulkan] Cannot recycle a fence before submission completion");
    return std::exchange(_fence, VK_NULL_HANDLE);
}

vulkan_event::vulkan_event(bool signaled) : _signaled(signaled) {
    _set = signaled;
}

vulkan_event::vulkan_event(std::shared_ptr<vulkan_submission_state> submission) : _submission(std::move(submission)) {
    OPENVINO_ASSERT(_submission != nullptr, "[GPU][Vulkan] A submission event requires submission state");
}

void vulkan_event::reset() {
    OPENVINO_ASSERT(_submission == nullptr, "[GPU][Vulkan] Device submission events cannot be reset");
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _signaled = false;
    }
    event::reset();
}

bool vulkan_event::is_device_submission(VkDevice device, VkQueue queue) const {
    return _submission != nullptr && _submission->belongs_to(device, queue);
}

bool vulkan_event::is_stream_submission(VkDevice device, VkQueue queue, uint64_t stream_id) const {
    return _submission != nullptr && _submission->belongs_to_stream(device, queue, stream_id);
}

void vulkan_event::wait_impl() {
    if (_submission != nullptr) {
        _submission->wait();
        return;
    }

    std::unique_lock<std::mutex> lock(_mutex);
    _condition.wait(lock, [this] {
        return _signaled;
    });
}

void vulkan_event::set_impl() {
    if (_submission != nullptr) {
        _submission->wait();
        return;
    }

    {
        std::lock_guard<std::mutex> lock(_mutex);
        _signaled = true;
    }
    _condition.notify_all();
}

bool vulkan_event::is_set_impl() {
    if (_submission != nullptr) {
        return _submission->is_complete();
    }

    std::lock_guard<std::mutex> lock(_mutex);
    return _signaled;
}

bool vulkan_event::get_profiling_info_impl(std::list<instrumentation::profiling_interval>& info) {
    if (_submission == nullptr) {
        return true;
    }

    const auto profiling = _submission->get_profiling_period();
    if (!profiling.has_value()) {
        return true;
    }

    auto period = std::make_shared<instrumentation::profiling_period_basic>(profiling->duration);
    info.push_back({instrumentation::profiling_stage::executing, period, profiling->start, true});
    return true;
}

vulkan_events::vulkan_events(std::vector<event::ptr> events) : _events(std::move(events)) {}

void vulkan_events::wait_impl() {
    for (const auto& event : _events) {
        if (event != nullptr) {
            event->wait();
        }
    }
}

void vulkan_events::set_impl() {
    wait_impl();
}

bool vulkan_events::is_set_impl() {
    return std::all_of(_events.begin(), _events.end(), [](const auto& event) {
        return event == nullptr || event->is_set();
    });
}

bool vulkan_events::get_profiling_info_impl(std::list<instrumentation::profiling_interval>& info) {
    using interval = std::pair<std::chrono::nanoseconds, std::chrono::nanoseconds>;
    std::map<instrumentation::profiling_stage, std::vector<interval>> intervals_by_stage;

    for (const auto& event : _events) {
        if (event == nullptr) {
            continue;
        }
        for (const auto& current : event->get_profiling_info()) {
            if (!current.is_valid_start) {
                continue;
            }
            intervals_by_stage[current.stage].emplace_back(current.start, current.start + current.value->value());
        }
    }

    for (auto& [stage, intervals] : intervals_by_stage) {
        if (intervals.empty()) {
            continue;
        }
        std::sort(intervals.begin(), intervals.end());
        std::vector<interval> merged;
        merged.reserve(intervals.size());
        for (const auto& current : intervals) {
            if (merged.empty() || merged.back().second < current.first) {
                merged.push_back(current);
            } else {
                merged.back().second = std::max(merged.back().second, current.second);
            }
        }

        auto duration = std::chrono::nanoseconds::zero();
        for (const auto& current : merged) {
            duration += current.second - current.first;
        }
        auto period = std::make_shared<instrumentation::profiling_period_basic>(duration);
        info.push_back({stage, period, merged.front().first, true});
    }
    return true;
}

}  // namespace vulkan
}  // namespace cldnn
