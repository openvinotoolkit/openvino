// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include "intel_gpu/runtime/memory.hpp"

namespace cldnn {
namespace vulkan {

class vulkan_engine;
class vulkan_device;
class vulkan_memory_allocator;
class vulkan_stream;

namespace test {
class vulkan_buffer_allocation_access;
}

enum class vulkan_buffer_memory_usage {
    device,
    host_staging,
};

struct vulkan_memory_type_selection {
    uint32_t index = 0;
    VkMemoryPropertyFlags flags = 0;
};

vulkan_memory_type_selection select_vulkan_memory_type(const VkPhysicalDeviceMemoryProperties& properties,
                                                       uint32_t allowed_types,
                                                       vulkan_buffer_memory_usage usage,
                                                       bool prefer_unified_memory);

struct vulkan_non_coherent_range {
    VkDeviceSize offset = 0;
    VkDeviceSize size = 0;
};

vulkan_non_coherent_range align_vulkan_non_coherent_range(VkDeviceSize offset, VkDeviceSize size, VkDeviceSize allocation_size, VkDeviceSize atom_size);

class vulkan_buffer_allocation {
public:
    using ptr = std::shared_ptr<vulkan_buffer_allocation>;

    static ptr create(vulkan_engine& engine, size_t requested_size, vulkan_buffer_memory_usage usage);

    vulkan_buffer_allocation(std::shared_ptr<vulkan_device> device_owner, VkDeviceSize size);
    ~vulkan_buffer_allocation();

    vulkan_buffer_allocation(const vulkan_buffer_allocation&) = delete;
    vulkan_buffer_allocation& operator=(const vulkan_buffer_allocation&) = delete;

    void flush(VkDeviceSize offset = 0, VkDeviceSize size = VK_WHOLE_SIZE) const;
    void invalidate(VkDeviceSize offset = 0, VkDeviceSize size = VK_WHOLE_SIZE) const;

    VkDevice get_device() const {
        return _device;
    }

    VkBuffer get_buffer() const {
        return _buffer;
    }

    VkDeviceSize get_size() const {
        return _size;
    }

    void* get_mapped_data() const {
        return _mapped_data;
    }

    VkDeviceSize get_non_coherent_atom_size() const {
        return _non_coherent_atom_size;
    }

    bool is_mapped() const {
        return _mapped_data != nullptr;
    }

    bool is_host_cached() const {
        return (_memory_flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) != 0;
    }

    bool is_host_coherent() const {
        return (_memory_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
    }

private:
    friend class test::vulkan_buffer_allocation_access;

    void unmap() noexcept;

    bool is_host_visible() const {
        return (_memory_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
    }

    const VkDevice _device;
    VkBuffer _buffer = VK_NULL_HANDLE;
    VkDeviceMemory _memory = VK_NULL_HANDLE;
    void* _mapped_data = nullptr;
    const VkDeviceSize _size;
    VkDeviceSize _memory_size = 0;
    uint32_t _memory_heap = 0;
    VkMemoryPropertyFlags _memory_flags = 0;
    VkDeviceSize _non_coherent_atom_size = 1;
    std::shared_ptr<vulkan_device> _device_owner;
    mutable std::mutex _visibility_mutex;
};

class vulkan_buffer_region {
public:
    using ptr = std::shared_ptr<vulkan_buffer_region>;

    ~vulkan_buffer_region();

    const vulkan_buffer_allocation::ptr& get_allocation() const {
        return _allocation;
    }

    VkDeviceSize get_offset() const {
        return _offset;
    }

    VkDeviceSize get_size() const {
        return _size;
    }

private:
    friend class vulkan_memory_allocator;

    vulkan_buffer_region(std::weak_ptr<vulkan_memory_allocator> owner, vulkan_buffer_allocation::ptr allocation, VkDeviceSize offset, VkDeviceSize size);

    std::weak_ptr<vulkan_memory_allocator> _owner;
    vulkan_buffer_allocation::ptr _allocation;
    VkDeviceSize _offset = 0;
    VkDeviceSize _size = 0;
};

class vulkan_memory_allocator final : public std::enable_shared_from_this<vulkan_memory_allocator> {
public:
    using block_allocator = std::function<vulkan_buffer_allocation::ptr(size_t)>;

    vulkan_memory_allocator(VkDeviceSize alignment, VkDeviceSize preferred_block_size, VkDeviceSize max_block_size, block_allocator allocate_block);

    vulkan_buffer_region::ptr allocate(size_t size);
    vulkan_buffer_allocation::ptr allocate_staging(size_t size, const block_allocator& allocate_block);

private:
    friend class vulkan_buffer_region;

    struct free_range {
        VkDeviceSize offset = 0;
        VkDeviceSize size = 0;
    };

    struct block {
        vulkan_buffer_allocation::ptr allocation;
        std::vector<free_range> free_ranges;
        size_t live_regions = 0;
        bool cacheable = false;
    };

    void release(const vulkan_buffer_allocation::ptr& allocation, VkDeviceSize offset, VkDeviceSize size);
    bool reclaim_idle_blocks_locked();
    VkDeviceSize align_size(VkDeviceSize size) const;

    VkDeviceSize _alignment = 1;
    VkDeviceSize _preferred_block_size = 1;
    VkDeviceSize _max_block_size = 1;
    block_allocator _allocate_block;
    mutable std::mutex _mutex;
    std::vector<block> _blocks;
};

class vulkan_buffer final : public memory {
public:
    using memory::fill;

    vulkan_buffer(vulkan_engine* engine,
                  const layout& layout,
                  vulkan_buffer_region::ptr region,
                  VkDeviceSize view_offset,
                  std::shared_ptr<MemoryTracker> memory_tracker);

    void* lock(const stream& stream, mem_lock_type type = mem_lock_type::read_write) override;
    void unlock(const stream& stream) override;

    event::ptr fill(stream& stream, unsigned char pattern, const std::vector<event::ptr>& dep_events = {}, bool blocking = true) override;
    shared_mem_params get_internal_params(runtime_types runtime_type) const override;

    event::ptr copy_from(stream& stream, const void* source, size_t source_offset, size_t destination_offset, size_t size, bool blocking) override;
    event::ptr copy_from(stream& stream, const memory& source, size_t source_offset, size_t destination_offset, size_t size, bool blocking) override;
    event::ptr copy_to(stream& stream, void* destination, size_t source_offset, size_t destination_offset, size_t size, bool blocking) const override;

    VkBuffer get_buffer() const {
        return _region->get_allocation()->get_buffer();
    }

    VkDeviceSize get_offset() const {
        return _region->get_offset() + _view_offset;
    }

    const vulkan_buffer_allocation::ptr& get_allocation() const {
        return _region->get_allocation();
    }

    const vulkan_buffer_region::ptr& get_region() const {
        return _region;
    }

    VkDeviceSize get_view_offset() const {
        return _view_offset;
    }

private:
    void validate_range(size_t offset, size_t size, const char* operation) const;
    bool can_access_mapping(size_t offset, size_t size) const;
    void* mapped_data() const;
    vulkan_buffer_allocation::ptr allocate_staging(size_t size) const;
    event::ptr write_staged(const vulkan_stream& queue, size_t offset, size_t size, const std::function<void(unsigned char*)>& write);

    vulkan_buffer_region::ptr _region;
    VkDeviceSize _view_offset = 0;
    mutable std::mutex _lock_mutex;
    vulkan_buffer_allocation::ptr _lock_staging;
    size_t _lock_count = 0;
    bool _write_access = false;
};

}  // namespace vulkan
}  // namespace cldnn
