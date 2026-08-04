// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/event.hpp"
#include "intel_gpu/runtime/stream.hpp"
#include "vk_common.hpp"

#include <memory>
#include <vector>

namespace cldnn {
namespace vk {

class vk_engine;

class vk_stream : public stream {
public:
    vk_stream(VkDevice device, VkQueue queue, uint32_t queue_family, const ExecutionConfig& config);
    ~vk_stream() override;

    VkDevice get_context() const { return _device; }
    VkQueue get_queue() const { return _queue; }

    void flush() const override;
    void finish() const override;
    void wait() override;

    void set_arguments(kernel& kernel, const kernel_arguments_desc& args_desc, const kernel_arguments_data& args) override;
    event::ptr enqueue_kernel(kernel& kernel,
                              const kernel_arguments_desc& args_desc,
                              const kernel_arguments_data& args,
                              std::vector<event::ptr> const& deps,
                              bool is_output = false) override;
    event::ptr enqueue_marker(std::vector<event::ptr> const& deps, bool is_output) override;
    event::ptr group_events(std::vector<event::ptr> const& deps) override;
    void enqueue_barrier() override;
    void wait_for_events(const std::vector<event::ptr>& events) override;
    event::ptr create_user_event(bool set) override;
    event::ptr create_base_event() override;
    std::unique_ptr<surfaces_lock> create_surfaces_lock(const std::vector<memory::ptr>& mem) const override;

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

}  // namespace vk
}  // namespace cldnn
