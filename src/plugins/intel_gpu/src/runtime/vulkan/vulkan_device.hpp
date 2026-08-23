// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <mutex>

#include "intel_gpu/runtime/device.hpp"
#include "vulkan_instance.hpp"

namespace cldnn {
namespace vulkan {

class vulkan_pipeline_cache;

struct vulkan_external_memory_capabilities {
    VkExternalMemoryHandleTypeFlags importable_buffer_handle_types = 0;
    VkExternalMemoryHandleTypeFlags dedicated_buffer_handle_types = 0;
    VkDeviceSize min_imported_host_pointer_alignment = 1;

    bool supports(VkExternalMemoryHandleTypeFlagBits handle_type) const {
        return (importable_buffer_handle_types & handle_type) != 0;
    }

    bool requires_dedicated_allocation(VkExternalMemoryHandleTypeFlagBits handle_type) const {
        return (dedicated_buffer_handle_types & handle_type) != 0;
    }
};

class vulkan_device : public device {
public:
    using ptr = std::shared_ptr<vulkan_device>;

    vulkan_device(vulkan_instance::ptr instance, VkPhysicalDevice physical_device, uint32_t compute_queue_family, bool initialize_device);

    vulkan_device(const vulkan_device&) = delete;
    vulkan_device& operator=(const vulkan_device&) = delete;

    ~vulkan_device() override;

    const device_info& get_info() const override {
        return _info;
    }

    memory_capabilities get_mem_caps() const override {
        return _mem_caps;
    }

    engine_types get_engine_type() const override {
        return engine_types::vulkan;
    }

    runtime_types get_runtime_type() const override {
        return runtime_types::vulkan;
    }

    void initialize() override;
    bool is_initialized() const override;
    bool is_same(const device::ptr other) override;
    void set_mem_caps(const memory_capabilities& memory_capabilities) override;

    VkPhysicalDevice get_physical_device() const {
        return _physical_device;
    }

    VkDevice get_device() const {
        return _device;
    }

    VkQueue get_compute_queue() const {
        return _compute_queue;
    }

    uint32_t get_compute_queue_family() const {
        return _compute_queue_family;
    }

    uint32_t get_max_push_constants_size() const {
        return _max_push_constants_size;
    }

    uint32_t get_max_memory_allocation_count() const {
        return _max_memory_allocation_count;
    }

    VkDeviceSize get_non_coherent_atom_size() const {
        return _non_coherent_atom_size;
    }

    const vulkan_external_memory_capabilities& get_external_memory_capabilities() const {
        return _external_memory_capabilities;
    }

    std::mutex& get_queue_mutex() {
        return _queue_mutex;
    }

    vulkan_pipeline_cache& get_pipeline_cache() const;

private:
    void initialize_info();

    vulkan_instance::ptr _instance;
    VkPhysicalDevice _physical_device = VK_NULL_HANDLE;
    uint32_t _compute_queue_family = 0;
    VkDevice _device = VK_NULL_HANDLE;
    VkQueue _compute_queue = VK_NULL_HANDLE;
    uint32_t _max_push_constants_size = 0;
    uint32_t _max_memory_allocation_count = 1;
    VkDeviceSize _non_coherent_atom_size = 1;
    vulkan_external_memory_capabilities _external_memory_capabilities{};

    device_info _info{};
    memory_capabilities _mem_caps{{allocation_type::device_buffer}};
    mutable std::mutex _init_mutex;
    std::mutex _queue_mutex;
    std::unique_ptr<vulkan_pipeline_cache> _pipeline_cache;
};

}  // namespace vulkan
}  // namespace cldnn
