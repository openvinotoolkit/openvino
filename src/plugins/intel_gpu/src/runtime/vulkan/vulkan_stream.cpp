// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vulkan_stream.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <thread>
#include <utility>

#include "openvino/core/except.hpp"
#include "vulkan_api.hpp"
#include "vulkan_engine.hpp"
#include "vulkan_memory.hpp"

namespace cldnn::vulkan {
namespace {

bool submission_completed(VkResult result) {
    return result == VK_SUCCESS || result == VK_ERROR_DEVICE_LOST;
}

void wait_for(const std::vector<event::ptr>& dependencies) {
    for (const auto& dependency : dependencies) {
        if (dependency) {
            dependency->wait();
        }
    }
}

class vulkan_user_event final : public event {
public:
    explicit vulkan_user_event(bool signaled) : _signaled(signaled) {
        _set = signaled;
    }

private:
    void wait_impl() override {
        std::unique_lock<std::mutex> lock(_mutex);
        _changed.wait(lock, [this] {
            return _signaled;
        });
    }

    void set_impl() override {
        const std::lock_guard<std::mutex> lock(_mutex);
        _signaled = true;
        _changed.notify_all();
    }

    bool is_set_impl() override {
        const std::lock_guard<std::mutex> lock(_mutex);
        return _signaled;
    }

    std::mutex _mutex;
    std::condition_variable _changed;
    bool _signaled;
};

class vulkan_event_group final : public event {
public:
    explicit vulkan_event_group(std::vector<event::ptr> dependencies) : _dependencies(std::move(dependencies)) {}

private:
    void wait_impl() override {
        wait_for(_dependencies);
    }

    void set_impl() override {
        wait_impl();
    }

    bool is_set_impl() override {
        return std::all_of(_dependencies.begin(), _dependencies.end(), [](const event::ptr& dependency) {
            return !dependency || dependency->is_set();
        });
    }

    std::vector<event::ptr> _dependencies;
};

void memory_barrier(VkCommandBuffer commands, VkPipelineStageFlags2 destination, VkAccessFlags2 access) {
    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    barrier.dstStageMask = destination;
    barrier.dstAccessMask = access;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(commands, &dependency);
}

}  // namespace

vulkan_stream::vulkan_stream(const vulkan_engine& engine) : vulkan_stream(engine, ExecutionConfig{}) {}

vulkan_stream::vulkan_stream(const vulkan_engine& engine, const ExecutionConfig& config)
    : stream(config.get_queue_type(), SyncMethods::events),
      _device(engine.get_vulkan_device_object()) {
    const VkDevice device = _device->get_device();
    VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    pool_info.queueFamilyIndex = _device->get_compute_queue_family();
    check_vk_result(vkCreateCommandPool(device, &pool_info, nullptr, &_pool), "vkCreateCommandPool");
    try {
        VkCommandBufferAllocateInfo buffer_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        buffer_info.commandPool = _pool;
        buffer_info.commandBufferCount = 1;
        check_vk_result(vkAllocateCommandBuffers(device, &buffer_info, &_commands), "vkAllocateCommandBuffers");
        const VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        check_vk_result(vkCreateFence(device, &fence_info, nullptr, &_completion), "vkCreateFence");
    } catch (...) {
        vkDestroyCommandPool(device, _pool, nullptr);
        throw;
    }
}

vulkan_stream::~vulkan_stream() {
    const VkDevice device = _device->get_device();
    vkDestroyFence(device, _completion, nullptr);
    vkDestroyCommandPool(device, _pool, nullptr);
}

void vulkan_stream::submit(const std::function<void(VkCommandBuffer)>& record) const {
    const std::lock_guard<std::mutex> lock(_device->get_queue_mutex());
    const VkDevice device = _device->get_device();
    const VkQueue queue = _device->get_compute_queue();
    check_vk_result(vkResetCommandPool(device, _pool, 0), "vkResetCommandPool");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check_vk_result(vkBeginCommandBuffer(_commands, &begin), "vkBeginCommandBuffer");
    memory_barrier(_commands, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
    record(_commands);
    memory_barrier(_commands, VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
    check_vk_result(vkEndCommandBuffer(_commands), "vkEndCommandBuffer");

    VkSubmitInfo submit_info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &_commands;
    check_vk_result(vkResetFences(device, 1, &_completion), "vkResetFences");
    check_vk_result(vkQueueSubmit(queue, 1, &submit_info, _completion), "vkQueueSubmit");
    const auto result = vkWaitForFences(device, 1, &_completion, VK_TRUE, std::numeric_limits<uint64_t>::max());
    if (!submission_completed(result)) {
        // Keep the caller's buffers alive until the submission can be safely retired.
        while (!submission_completed(vkQueueWaitIdle(queue)) && !submission_completed(vkGetFenceStatus(device, _completion))) {
            std::this_thread::sleep_for(std::chrono::milliseconds{1});
        }
    }
    check_vk_result(result, "vkWaitForFences");
}

// Transfers wait for the fence before returning.
void vulkan_stream::flush() const {}
void vulkan_stream::finish() const {}
void vulkan_stream::wait() {}

void vulkan_stream::set_arguments(kernel& /*kernel*/, const kernel_arguments_desc& /*descriptor*/, const kernel_arguments_data& /*data*/) {
    OPENVINO_THROW("[GPU][Vulkan] Kernel dispatch is unavailable");
}

event::ptr vulkan_stream::enqueue_kernel(kernel& /*kernel*/,
                                         const kernel_arguments_desc& /*descriptor*/,
                                         const kernel_arguments_data& /*data*/,
                                         const std::vector<event::ptr>& /*dependencies*/,
                                         bool /*is_output_event*/) {
    OPENVINO_THROW("[GPU][Vulkan] Kernel dispatch is unavailable");
}

event::ptr vulkan_stream::enqueue_marker(const std::vector<event::ptr>& dependencies, bool /*is_output_event*/) {
    wait_for(dependencies);
    return create_user_event(true);
}

void vulkan_stream::enqueue_barrier(const std::vector<event::ptr>& dependencies) {
    wait_for(dependencies);
}

event::ptr vulkan_stream::group_events(const std::vector<event::ptr>& dependencies) {
    return std::make_shared<vulkan_event_group>(dependencies);
}

void vulkan_stream::wait_for_events(const std::vector<event::ptr>& dependencies) {
    wait_for(dependencies);
}

event::ptr vulkan_stream::create_user_event(bool set) {
    return std::make_shared<vulkan_user_event>(set);
}

event::ptr vulkan_stream::create_base_event() {
    return create_user_event(false);
}

std::unique_ptr<surfaces_lock> vulkan_stream::create_surfaces_lock(const std::vector<memory::ptr>& /*memory*/) const {
    return std::make_unique<surfaces_lock>();
}

event::ptr vulkan_stream::enqueue_buffer_copy(const std::shared_ptr<vulkan_buffer_allocation>& source,
                                              VkDeviceSize source_offset,
                                              const std::shared_ptr<vulkan_buffer_allocation>& destination,
                                              VkDeviceSize destination_offset,
                                              VkDeviceSize size) const {
    OPENVINO_ASSERT(source->get_device() == _device->get_device() && destination->get_device() == _device->get_device(),
                    "[GPU][Vulkan] Buffer copy requires allocations from the stream's device");
    if (size > 0) {
        const VkBufferCopy range{source_offset, destination_offset, size};
        submit([&](VkCommandBuffer commands) {
            vkCmdCopyBuffer(commands, source->get_buffer(), destination->get_buffer(), 1, &range);
        });
    }
    return std::make_shared<vulkan_user_event>(true);
}

event::ptr vulkan_stream::enqueue_buffer_fill(const std::shared_ptr<vulkan_buffer_allocation>& destination,
                                              VkDeviceSize destination_offset,
                                              VkDeviceSize size,
                                              uint32_t pattern,
                                              const std::vector<event::ptr>& dependencies) const {
    OPENVINO_ASSERT(destination->get_device() == _device->get_device(), "[GPU][Vulkan] Buffer fill requires an allocation from the stream's device");
    wait_for(dependencies);
    if (size > 0) {
        submit([&](VkCommandBuffer commands) {
            vkCmdFillBuffer(commands, destination->get_buffer(), destination_offset, size, pattern);
        });
    }
    return std::make_shared<vulkan_user_event>(true);
}

}  // namespace cldnn::vulkan
