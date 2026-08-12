// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vulkan_event.hpp"

#include <utility>

#include "openvino/core/except.hpp"

namespace cldnn {
namespace vulkan {

vulkan_submission_state::vulkan_submission_state(VkDevice device, VkQueue queue, VkFence fence) : _device(device), _queue(queue), _fence(fence) {
    OPENVINO_ASSERT(_device != VK_NULL_HANDLE && _queue != VK_NULL_HANDLE && _fence != VK_NULL_HANDLE,
                    "[GPU][Vulkan] A device submission requires valid Vulkan handles");
}

void vulkan_submission_state::wait() {
    std::lock_guard<std::mutex> lock(_mutex);
    if (_completed) {
        return;
    }

    OPENVINO_ASSERT(_fence != VK_NULL_HANDLE, "[GPU][Vulkan] A pending submission has no fence");
    const auto result = vkWaitForFences(_device, 1, &_fence, VK_TRUE, UINT64_MAX);
    OPENVINO_ASSERT(result == VK_SUCCESS, "[GPU][Vulkan] vkWaitForFences failed with VkResult ", static_cast<int>(result));
    _completed = true;
}

bool vulkan_submission_state::is_complete() {
    std::lock_guard<std::mutex> lock(_mutex);
    if (_completed) {
        return true;
    }

    OPENVINO_ASSERT(_fence != VK_NULL_HANDLE, "[GPU][Vulkan] A pending submission has no fence");
    const auto result = vkGetFenceStatus(_device, _fence);
    if (result == VK_NOT_READY) {
        return false;
    }
    OPENVINO_ASSERT(result == VK_SUCCESS, "[GPU][Vulkan] vkGetFenceStatus failed with VkResult ", static_cast<int>(result));
    _completed = true;
    return true;
}

bool vulkan_submission_state::belongs_to(VkDevice device, VkQueue queue) const {
    return _device == device && _queue == queue;
}

VkFence vulkan_submission_state::release_fence() {
    std::lock_guard<std::mutex> lock(_mutex);
    OPENVINO_ASSERT(_completed, "[GPU][Vulkan] Cannot recycle a fence before submission completion");
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

}  // namespace vulkan
}  // namespace cldnn
