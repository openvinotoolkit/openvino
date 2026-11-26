// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <memory>
#include <mutex>

#include "intel_gpu/runtime/device.hpp"
#include "vulkan_instance.hpp"

namespace cldnn {
namespace vulkan {

class vulkan_memory_budget {
public:
    struct limits {
        VkDeviceSize total;
        VkDeviceSize heap;
        uint32_t allocations;
    };

    bool try_reserve(uint32_t heap_index, VkDeviceSize size, const limits& capacity) noexcept;
    void release(uint32_t heap_index, VkDeviceSize size) noexcept;

private:
    std::mutex _mutex;
    std::array<VkDeviceSize, VK_MAX_MEMORY_HEAPS> _heap_usage{};
    VkDeviceSize _total_usage = 0;
    uint32_t _allocation_count = 0;
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

    void initialize() override;
    bool is_initialized() const override;
    bool is_same(device::ptr other) override;
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

    uint32_t get_max_memory_allocation_count() const {
        return _max_memory_allocation_count;
    }

    VkDeviceSize get_non_coherent_atom_size() const {
        return _non_coherent_atom_size;
    }

    VkDeviceSize get_buffer_alignment() const {
        return _buffer_alignment;
    }

    VkDeviceSize get_max_buffer_size() const {
        return _max_buffer_size;
    }

    VkDeviceSize get_max_memory_allocation_size() const {
        return _max_memory_allocation_size;
    }

    const VkPhysicalDeviceMemoryProperties& get_memory_properties() const {
        return _memory_properties;
    }

    bool try_reserve_memory(uint32_t heap_index, VkDeviceSize size, VkDeviceSize limit) noexcept;
    void release_memory(uint32_t heap_index, VkDeviceSize size) noexcept;

    std::mutex& get_queue_mutex() {
        return _queue_mutex;
    }

private:
    void initialize_info();

    vulkan_instance::ptr _instance;
    VkPhysicalDevice _physical_device = VK_NULL_HANDLE;
    uint32_t _compute_queue_family = 0;
    VkDevice _device = VK_NULL_HANDLE;
    VkQueue _compute_queue = VK_NULL_HANDLE;
    uint32_t _max_memory_allocation_count = 1;
    VkDeviceSize _non_coherent_atom_size = 1;
    VkDeviceSize _buffer_alignment = 1;
    VkDeviceSize _max_buffer_size = 0;
    VkDeviceSize _max_memory_allocation_size = 0;
    VkPhysicalDeviceMemoryProperties _memory_properties{};
    vulkan_memory_budget _memory_budget;
    device_info _info{};
    memory_capabilities _mem_caps{{allocation_type::vk_buffer}, allocation_type::vk_buffer};
    mutable std::mutex _init_mutex;
    std::mutex _queue_mutex;
};

}  // namespace vulkan
}  // namespace cldnn
