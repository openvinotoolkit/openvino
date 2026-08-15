// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vk_common.hpp"
#include "vk_types.hpp"

#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

namespace ov::core::vulkan {
namespace cross_platform {

// A Vulkan buffer (VkBuffer + VkDeviceMemory). Host-visible allocations are
// kept permanently mapped; device-local buffers use one-shot transfer submits
// with a staging buffer for host <-> device copies. A view created from
// reinterpret_buffer shares the base allocation and owns nothing.
class vk_memory {
public:
    vk_memory(VkDevice device, VkPhysicalDevice phys_device, VkQueue queue, const layout& layout, allocation_type type);
    // Creates a view sharing the base allocation (same VkBuffer/VkDeviceMemory).
    explicit vk_memory(const std::shared_ptr<vk_memory>& base, const layout& new_layout);
    ~vk_memory();

    vk_memory(const vk_memory&) = delete;
    vk_memory& operator=(const vk_memory&) = delete;

    void* lock();
    void unlock();

    // Blocking transfers (performed on the memory's own one-shot command
    // pool). The stream parameter of the legacy interface is not needed.
    void fill(unsigned char pattern);
    void copy_from(const void* data_ptr, size_t src_offset, size_t dst_offset, size_t size);
    void copy_from(const vk_memory& src_mem, size_t src_offset, size_t dst_offset, size_t size);
    void copy_to(void* data_ptr, size_t src_offset, size_t dst_offset, size_t size) const;

    VkBuffer get_buffer() const { return _buffer; }
    VkDeviceMemory get_device_memory() const { return _memory; }
    bool is_host_visible() const { return _host_visible; }
    VkDeviceSize get_allocated_size() const { return _allocated_size; }
    const layout& get_layout() const { return _layout; }
    size_t byte_size() const { return _layout.byte_size(); }
    allocation_type get_allocation_type() const { return _type; }

private:
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
    layout _layout;
    allocation_type _type = allocation_type::unknown;

    mutable std::mutex _mutex;
    unsigned _lock_count = 0;
    void* _mapped_ptr = nullptr;

    // Non-null for views created by reinterpret_buffer; owns nothing.
    std::shared_ptr<vk_memory> _base;

    mutable VkCommandPool _cmd_pool = VK_NULL_HANDLE;
    mutable std::mutex _transfer_mutex;
};

using vk_memory_ptr = std::shared_ptr<vk_memory>;

}  // namespace cross_platform
}  // namespace ov::core::vulkan
