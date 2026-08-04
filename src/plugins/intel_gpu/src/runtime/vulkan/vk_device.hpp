// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/device.hpp"
#include "intel_gpu/runtime/engine_configuration.hpp"
#include "vk_common.hpp"

namespace cldnn {
namespace vk {

struct vk_device : public device {
public:
    vk_device(const instance_ptr& instance, VkPhysicalDevice physical_device);

    const device_info& get_info() const override { return _info; }
    memory_capabilities get_mem_caps() const override { return _mem_caps; }

    void initialize() override;
    bool is_initialized() const override { return _is_initialized; };

    bool is_same(const device::ptr other) override;
    void set_mem_caps(const memory_capabilities& memory_capabilities) override;

    device_type type() const { return _info.dev_type; }

    runtime_types runtime_type() const { return runtime_types::vulkan; }

    VkInstance get_instance() const { return _instance.get(); }
    VkPhysicalDevice get_physical_device() const { return _physical_device; }
    VkDevice get_device() const { return _device.get(); }
    VkQueue get_queue() const { return _queue; }
    uint32_t get_queue_family_index() const { return _queue_family_index; }

    ~vk_device() = default;

private:
    instance_ptr _instance;
    VkPhysicalDevice _physical_device;
    device_ptr _device;
    VkQueue _queue = VK_NULL_HANDLE;
    uint32_t _queue_family_index = 0;
    device_info _info;
    memory_capabilities _mem_caps;
    bool _is_initialized = false;
};

}  // namespace vk
}  // namespace cldnn
