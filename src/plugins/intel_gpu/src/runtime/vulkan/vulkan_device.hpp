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

    device_info _info{};
    memory_capabilities _mem_caps{{allocation_type::vulkan_buffer}};
    mutable std::mutex _init_mutex;
    std::mutex _queue_mutex;
    std::unique_ptr<vulkan_pipeline_cache> _pipeline_cache;
};

}  // namespace vulkan
}  // namespace cldnn
