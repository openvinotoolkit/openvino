// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vk_common.hpp"

#include "intel_gpu/runtime/memory.hpp"

#include <functional>
#include <mutex>
#include <vector>

namespace cldnn {
namespace vk {

// Forward declaration: vk_engine is owned by the Vulkan backend (see vk_engine.hpp).
class vk_engine;

class vk_memory : public memory {
public:
    vk_memory(engine* engine,
              const layout& layout,
              allocation_type type,
              VkDevice device,
              VkPhysicalDevice phys_device,
              VkQueue queue);
    ~vk_memory() override;

    vk_memory(const vk_memory&) = delete;
    vk_memory& operator=(const vk_memory&) = delete;

    void* lock(const stream& stream, mem_lock_type type = mem_lock_type::read_write) override;
    void unlock(const stream& stream) override;
    event::ptr fill(stream& stream, unsigned char pattern, const std::vector<event::ptr>& dep_events = {}, bool blocking = true) override;
    shared_mem_params get_internal_params(runtime_types rt_type) const override;
    void* buffer_ptr() const override { return nullptr; }

    event::ptr copy_from(stream& stream, const void* data_ptr, size_t src_offset, size_t dst_offset, size_t size, bool blocking) override;
    event::ptr copy_from(stream& stream, const memory& src_mem, size_t src_offset, size_t dst_offset, size_t size, bool blocking) override;
    event::ptr copy_to(stream& stream, void* data_ptr, size_t src_offset, size_t dst_offset, size_t size, bool blocking) const override;

    VkBuffer get_buffer() const { return _buffer; }
    VkDeviceMemory get_device_memory() const { return _memory; }
    bool is_host_visible() const { return _host_visible; }
    VkDeviceSize get_allocated_size() const { return _allocated_size; }

protected:
    void allocate();
    void deallocate();
    void ensure_command_pool() const;
    void one_shot_submit(const std::function<void(VkCommandBuffer)>& record) const;

    VkDevice _device = VK_NULL_HANDLE;
    VkPhysicalDevice _phys_device = VK_NULL_HANDLE;
    VkQueue _queue = VK_NULL_HANDLE;
    VkBuffer _buffer = VK_NULL_HANDLE;
    VkDeviceMemory _memory = VK_NULL_HANDLE;
    bool _host_visible = false;
    VkDeviceSize _allocated_size = 0;

    mutable std::mutex _mutex;
    unsigned _lock_count = 0;
    void* _mapped_ptr = nullptr;

    mutable VkCommandPool _cmd_pool = VK_NULL_HANDLE;
    mutable std::mutex _transfer_mutex;
};

}  // namespace vk
}  // namespace cldnn
