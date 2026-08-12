// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vulkan/vulkan.h>

#include <memory>
#include <mutex>

#include "intel_gpu/runtime/memory.hpp"

namespace cldnn {
namespace vulkan {

class vulkan_engine;
class vulkan_device;

class vulkan_buffer_allocation {
public:
    using ptr = std::shared_ptr<vulkan_buffer_allocation>;

    vulkan_buffer_allocation(std::shared_ptr<vulkan_device> device_owner,
                             VkDevice device,
                             VkBuffer buffer,
                             VkDeviceMemory memory,
                             void* mapped_data,
                             VkDeviceSize size,
                             bool host_coherent);
    ~vulkan_buffer_allocation();

    vulkan_buffer_allocation(const vulkan_buffer_allocation&) = delete;
    vulkan_buffer_allocation& operator=(const vulkan_buffer_allocation&) = delete;

    void flush() const;
    void invalidate() const;

    VkDevice device = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped_data = nullptr;
    VkDeviceSize size = 0;
    bool host_coherent = false;

private:
    std::shared_ptr<vulkan_device> _device_owner;
};

class vulkan_buffer final : public memory {
public:
    using memory::fill;

    vulkan_buffer(vulkan_engine* engine, const layout& layout);
    vulkan_buffer(vulkan_engine* engine,
                  const layout& layout,
                  vulkan_buffer_allocation::ptr allocation,
                  VkDeviceSize offset,
                  std::shared_ptr<MemoryTracker> memory_tracker);

    void* lock(const stream& stream, mem_lock_type type = mem_lock_type::read_write) override;
    void unlock(const stream& stream) override;

    event::ptr fill(stream& stream, unsigned char pattern, const std::vector<event::ptr>& dep_events = {}, bool blocking = true) override;
    shared_mem_params get_internal_params(runtime_types runtime_type) const override;

    event::ptr copy_from(stream& stream, const void* source, size_t source_offset, size_t destination_offset, size_t size, bool blocking) override;
    event::ptr copy_from(stream& stream, const memory& source, size_t source_offset, size_t destination_offset, size_t size, bool blocking) override;
    event::ptr copy_to(stream& stream, void* destination, size_t source_offset, size_t destination_offset, size_t size, bool blocking) const override;

    VkBuffer get_buffer() const {
        return _allocation->buffer;
    }

    VkDeviceSize get_offset() const {
        return _offset;
    }

    const vulkan_buffer_allocation::ptr& get_allocation() const {
        return _allocation;
    }

private:
    void validate_range(size_t offset, size_t size, const char* operation) const;
    void* mapped_data() const;

    vulkan_buffer_allocation::ptr _allocation;
    VkDeviceSize _offset = 0;
    mutable std::mutex _lock_mutex;
    size_t _lock_count = 0;
    bool _write_access = false;
};

}  // namespace vulkan
}  // namespace cldnn
