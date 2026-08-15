// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "runtime/execution_config.hpp"
#include "vk_common.hpp"
#include "vk_device.hpp"
#include "vk_memory.hpp"
#include "vk_platform.hpp"
#include "vk_stream.hpp"
#include "vk_types.hpp"

#include <memory>
#include <string>
#include <vector>

namespace ov::core::vulkan {
namespace cross_platform {

using ov::intel_gpu::ExecutionConfig;

class vk_kernel_builder;

struct vk_engine_init_data {
    instance_ptr instance;
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    vk_device_ptr device;
    vk_platform_config config;
};

// Owns the Vulkan instance/device and produces streams, buffers and kernel
// builders. The self-contained counterpart of the legacy engine interface.
class vk_engine {
public:
    vk_engine();
    explicit vk_engine(const vk_platform_config& config);
    ~vk_engine();

    vk_engine(const vk_engine&) = delete;
    vk_engine& operator=(const vk_engine&) = delete;

    engine_types type() const { return engine_types::vulkan; }
    runtime_types runtime_type() const { return runtime_types::vulkan; }
    backend_types backend_type() const { return backend_types::vulkan; }

    vk_memory_ptr allocate_memory(const layout& layout, allocation_type type, bool reset = true);
    vk_memory_ptr reinterpret_buffer(const vk_memory_ptr& mem, const layout& new_layout);

    allocation_type get_default_allocation_type() const { return allocation_type::usm_device; }

    vk_stream_ptr create_stream(const ExecutionConfig& config) const;
    vk_stream& get_service_stream() const { return *_service_stream; }

    std::shared_ptr<vk_kernel_builder> create_kernel_builder() const;

    VkInstance get_instance() const { return _device_handle.instance; }
    VkDevice get_device_handle() const { return _device_handle.device; }
    VkPhysicalDevice get_physical_device() const { return _device_handle.physical; }
    VkQueue get_queue() const { return _device_handle.queue; }
    uint32_t get_queue_family_index() const { return _device_handle.queue_family; }

    static std::shared_ptr<vk_engine> create(const vk_platform_config& config = {});

private:
    vk_engine(vk_engine_init_data init);

    struct vk_device_handles {
        VkInstance instance = VK_NULL_HANDLE;
        VkPhysicalDevice physical = VK_NULL_HANDLE;
        VkDevice device = VK_NULL_HANDLE;
        uint32_t queue_family = 0;
        VkQueue queue = VK_NULL_HANDLE;
    };

    // Kept alive for the engine's lifetime: the engine stores raw handles but
    // must not outlive the vk_device that owns the VkDevice/VkInstance.
    vk_device_ptr _device;
    vk_device_handles _device_handle;
    vk_platform_config _config;
    std::unique_ptr<vk_stream> _service_stream;
};

}  // namespace cross_platform
}  // namespace ov::core::vulkan
