// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vulkan/vulkan.h>

#include <memory>

#include "intel_gpu/runtime/stream.hpp"

namespace cldnn {
namespace vulkan {

class vulkan_engine;

class vulkan_stream final : public stream {
public:
    explicit vulkan_stream(const vulkan_engine& engine);
    vulkan_stream(const vulkan_engine& engine, const ExecutionConfig& config);
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
    void enqueue_barrier() override;
    event::ptr group_events(const std::vector<event::ptr>& dependencies) override;
    void wait_for_events(const std::vector<event::ptr>& events) override;
    event::ptr create_user_event(bool set) override;
    event::ptr create_base_event() override;
    std::unique_ptr<surfaces_lock> create_surfaces_lock(const std::vector<memory::ptr>& memory) const override;
    void copy_buffer(VkBuffer source, VkDeviceSize source_offset, VkBuffer destination, VkDeviceSize destination_offset, VkDeviceSize size) const;

#ifdef ENABLE_ONEDNN_FOR_GPU
    dnnl::stream& get_onednn_stream() override;
#endif

private:
    struct resource_state;

    const vulkan_engine& _engine;
    std::unique_ptr<resource_state> _resources;
};

}  // namespace vulkan
}  // namespace cldnn
