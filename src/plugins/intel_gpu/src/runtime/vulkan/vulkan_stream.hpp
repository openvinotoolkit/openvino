// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vulkan/vulkan.h>

#include <functional>
#include <memory>
#include <vector>

#include "intel_gpu/runtime/stream.hpp"

namespace cldnn {
namespace vulkan {

class vulkan_engine;
class vulkan_device;
class vulkan_buffer_allocation;

class vulkan_stream final : public stream {
public:
    explicit vulkan_stream(const vulkan_engine& engine);
    vulkan_stream(const vulkan_engine& engine, const ExecutionConfig& config);
    vulkan_stream(const vulkan_stream&) = delete;
    vulkan_stream& operator=(const vulkan_stream&) = delete;
    ~vulkan_stream() override;

    void flush() const override;
    void finish() const override;
    void wait() override;

    void set_arguments(kernel& kernel, const kernel_arguments_desc& descriptor, const kernel_arguments_data& data) override;
    event::ptr enqueue_kernel(kernel& kernel,
                              const kernel_arguments_desc& descriptor,
                              const kernel_arguments_data& data,
                              const std::vector<event::ptr>& dependencies,
                              bool is_output_event = false) override;
    event::ptr enqueue_marker(const std::vector<event::ptr>& dependencies, bool is_output_event = false) override;
    void enqueue_barrier(const std::vector<event::ptr>& dependencies = {}) override;
    event::ptr group_events(const std::vector<event::ptr>& dependencies) override;
    void wait_for_events(const std::vector<event::ptr>& dependencies) override;
    event::ptr create_user_event(bool set) override;
    event::ptr create_base_event() override;
    std::unique_ptr<surfaces_lock> create_surfaces_lock(const std::vector<memory::ptr>& memory) const override;
    event::ptr enqueue_buffer_copy(const std::shared_ptr<vulkan_buffer_allocation>& source,
                                   VkDeviceSize source_offset,
                                   const std::shared_ptr<vulkan_buffer_allocation>& destination,
                                   VkDeviceSize destination_offset,
                                   VkDeviceSize size) const;
    event::ptr enqueue_buffer_fill(const std::shared_ptr<vulkan_buffer_allocation>& destination,
                                   VkDeviceSize destination_offset,
                                   VkDeviceSize size,
                                   uint32_t pattern,
                                   const std::vector<event::ptr>& dependencies) const;

private:
    void submit(const std::function<void(VkCommandBuffer)>& record) const;

    std::shared_ptr<vulkan_device> _device;
    VkCommandPool _pool = VK_NULL_HANDLE;
    VkCommandBuffer _commands = VK_NULL_HANDLE;
    VkFence _completion = VK_NULL_HANDLE;
};

}  // namespace vulkan
}  // namespace cldnn
