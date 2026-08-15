// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vk_common.hpp"
#include "vk_event.hpp"
#include "vk_kernel.hpp"
#include "vk_types.hpp"

#include <memory>
#include <vector>

namespace ov::core::vulkan {
namespace cross_platform {

// A compute command stream: owns a command pool/buffer, the storage-buffer
// descriptor pool/set and performs kernel dispatches (blocking, in-order).
class vk_stream {
public:
    vk_stream(VkDevice device, VkQueue queue, uint32_t queue_family);
    ~vk_stream();

    vk_stream(const vk_stream&) = delete;
    vk_stream& operator=(const vk_stream&) = delete;

    VkDevice get_context() const { return _device; }
    VkQueue get_queue() const { return _queue; }

    void flush() const;
    void finish() const;
    void wait();

    // Dispatches |kernel| with the given arguments and returns an event that
    // is signaled when the submission completes (the dispatch itself is
    // blocking; the event is created for API compatibility).
    vk_event_ptr enqueue_kernel(vk_kernel& kernel, const kernel_arguments_desc& args_desc, const kernel_arguments_data& args);
    vk_event_ptr enqueue_marker();
    void enqueue_barrier();

private:
    void submit_and_wait() const;

    VkDevice _device;
    VkQueue _queue;
    uint32_t _queue_family;

    VkCommandPool _command_pool = VK_NULL_HANDLE;
    VkCommandBuffer _command_buffer = VK_NULL_HANDLE;

    VkDescriptorPool _descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSetLayout _descriptor_set_layout = VK_NULL_HANDLE;
    VkDescriptorSet _descriptor_set = VK_NULL_HANDLE;
};

using vk_stream_ptr = std::shared_ptr<vk_stream>;

}  // namespace cross_platform
}  // namespace ov::core::vulkan
