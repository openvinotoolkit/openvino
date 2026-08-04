// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/stream.hpp"
#include "vk_common.hpp"
#include "vk_device.hpp"

#include <memory>
#include <string>
#include <vector>

namespace cldnn {
namespace vk {

struct vk_engine_init_data {
    vk::instance_ptr instance;
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    cldnn::device::ptr device;
};

class vk_engine : public engine {
public:
    vk_engine();
    ~vk_engine() = default;

    engine_types type() const override;
    runtime_types runtime_type() const override;
    backend_types backend_type() const override;

    memory_ptr allocate_memory(const layout& layout, allocation_type type, bool reset = true) override;
    memory_ptr reinterpret_handle(const layout& new_layout, shared_mem_params params) override;
    memory_ptr create_subbuffer(const memory& memory, const layout& new_layout, size_t byte_offset) override;
    memory_ptr create_hostbuffer(void* cpu_address, size_t data_size, allocation_type _allocation_type, const layout output_layout) override;
    memory_ptr create_hostbuffer(const void* cpu_address, size_t data_size, allocation_type _allocation_type, const layout output_layout) override;
    memory_ptr reinterpret_buffer(const memory& memory, const layout& new_layout) override;
    memory_ptr import_buffer(const layout& layout, ov::intel_gpu::os_handle_param external_handle) override;
    bool is_the_same_buffer(const memory& mem1, const memory& mem2) override;

    void* get_user_context(runtime_types rt_type) const override;

    allocation_type get_default_allocation_type() const override { return allocation_type::usm_device; }
    allocation_type detect_usm_allocation_type(const void* memory) const override;

    stream_ptr create_stream(const ExecutionConfig& config) const override;
    stream_ptr create_stream(const ExecutionConfig& config, void* handle) const override;

    std::shared_ptr<kernel_builder> create_kernel_builder() const override;

#ifdef ENABLE_ONEDNN_FOR_GPU
    void create_onednn_engine(const ExecutionConfig& config) override;
#endif

    VkInstance get_instance() const { return _device_handle.instance; }
    VkDevice get_device_handle() const { return _device_handle.device; }
    VkPhysicalDevice get_physical_device() const { return _device_handle.physical; }
    VkQueue get_queue() const { return _device_handle.queue; }
    uint32_t get_queue_family_index() const { return _device_handle.queue_family; }

    static std::shared_ptr<cldnn::engine> create();

private:
    vk_engine(vk_engine_init_data init);

    struct vk_device_handles {
        VkInstance instance = VK_NULL_HANDLE;
        VkPhysicalDevice physical = VK_NULL_HANDLE;
        VkDevice device = VK_NULL_HANDLE;
        uint32_t queue_family = 0;
        VkQueue queue = VK_NULL_HANDLE;
    };

    vk_device_handles _device_handle;
};

}  // namespace vk
}  // namespace cldnn
