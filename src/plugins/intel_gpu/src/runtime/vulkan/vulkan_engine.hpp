// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/engine.hpp"
#include "vulkan_device.hpp"

namespace cldnn {
namespace vulkan {

class vulkan_buffer_region;
class vulkan_memory_allocator;
struct vulkan_memory_allocator_stats;

class vulkan_engine final : public engine {
public:
    vulkan_engine(const device::ptr& device, runtime_types runtime_type);

    engine_types type() const override {
        return engine_types::vulkan;
    }

    runtime_types runtime_type() const override {
        return runtime_types::vulkan;
    }

    backend_types backend_type() const override {
        return backend_types::vulkan;
    }

    memory_ptr allocate_memory(const layout& layout, allocation_type type, bool reset = true) override;
    memory_ptr reinterpret_handle(const layout& layout, shared_mem_params params) override;
    memory_ptr create_subbuffer(const memory& memory, const layout& layout, size_t byte_offset) override;
    memory_ptr create_hostbuffer(void* address, size_t size, allocation_type type, const layout layout) override;
    memory_ptr create_hostbuffer(const void* address, size_t size, allocation_type type, const layout layout, bool host_read_only = false) override;
    memory_ptr reinterpret_buffer(const memory& memory, const layout& layout) override;
    memory_ptr import_buffer(const layout& layout, ov::intel_gpu::os_handle_param external_handle) override;
    bool is_the_same_buffer(const memory& lhs, const memory& rhs) override;

    allocation_type get_default_allocation_type() const override {
        return allocation_type::device_buffer;
    }

    allocation_type detect_usm_allocation_type(const void*) const override {
        return allocation_type::unknown;
    }

    void* get_user_context(runtime_types runtime_type) const override;
    stream_ptr create_stream(const ExecutionConfig& config) const override;
    stream_ptr create_stream(const ExecutionConfig& config, void* handle) const override;
    std::shared_ptr<kernel_builder> create_kernel_builder() const override;

    VkDevice get_device_handle() const;
    VkPhysicalDevice get_physical_device() const;
    VkQueue get_compute_queue() const;
    uint32_t get_compute_queue_family() const;
    uint32_t get_max_push_constants_size() const;
    VkDeviceSize get_non_coherent_atom_size() const;
    std::mutex& get_queue_mutex() const;
    std::shared_ptr<vulkan_device> get_vulkan_device_object() const;
    std::shared_ptr<vulkan_buffer_region> allocate_buffer_region(size_t size);
    vulkan_memory_allocator_stats get_memory_allocator_stats() const;

#ifdef ENABLE_ONEDNN_FOR_GPU
    void create_onednn_engine(const ExecutionConfig&) override;
#endif

private:
    std::shared_ptr<vulkan_device> get_vulkan_device_object_impl() const;
    std::shared_ptr<vulkan_memory_allocator> _memory_allocator;
};

}  // namespace vulkan
}  // namespace cldnn
