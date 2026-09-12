// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "intel_gpu/runtime/memory.hpp"

namespace cldnn {
namespace vulkan {

class vulkan_engine;
class vulkan_device;
class vulkan_memory_allocator;

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

    vulkan_buffer_allocation(std::shared_ptr<vulkan_device> device_owner,
                             VkDevice device,
                             VkBuffer buffer,
                             VkDeviceMemory memory,
                             void* mapped_data,
                             VkDeviceSize size,
                             VkDeviceSize memory_size,
                             VkMemoryPropertyFlags memory_flags,
                             VkDeviceSize non_coherent_atom_size,
                             bool release_to_foreign = false);
    ~vulkan_buffer_allocation();

    vulkan_buffer_allocation(const vulkan_buffer_allocation&) = delete;
    vulkan_buffer_allocation& operator=(const vulkan_buffer_allocation&) = delete;

    void flush(VkDeviceSize offset = 0, VkDeviceSize size = VK_WHOLE_SIZE) const;
    void invalidate(VkDeviceSize offset = 0, VkDeviceSize size = VK_WHOLE_SIZE) const;

    bool is_host_visible() const {
        return (memory_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
    }

    bool is_device_local() const {
        return (memory_flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
    }

    bool is_host_cached() const {
        return (memory_flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) != 0;
    }

    bool is_host_coherent() const {
        return (memory_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
    }

    VkDevice device = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped_data = nullptr;
    VkDeviceSize size = 0;
    VkDeviceSize memory_size = 0;
    VkMemoryPropertyFlags memory_flags = 0;
    VkDeviceSize non_coherent_atom_size = 1;

private:
    std::shared_ptr<vulkan_device> _device_owner;
    bool _release_to_foreign = false;
    mutable std::mutex _visibility_mutex;
};

struct vulkan_memory_allocator_stats {
    uint64_t region_allocations = 0;
    uint64_t region_reuses = 0;
    uint64_t block_allocations = 0;
    uint64_t block_releases = 0;
    uint64_t current_blocks = 0;
    uint64_t current_reserved_bytes = 0;
    uint64_t peak_reserved_bytes = 0;
    VkDeviceSize alignment = 1;
    VkDeviceSize preferred_block_size = 1;
};

class vulkan_buffer_region {
public:
    using ptr = std::shared_ptr<vulkan_buffer_region>;

    ~vulkan_buffer_region();

    static ptr create_external(vulkan_buffer_allocation::ptr allocation, VkDeviceSize size);

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

vulkan_buffer_region::ptr import_vulkan_host_buffer(vulkan_engine& engine, void* address, size_t data_size, size_t buffer_size);
vulkan_buffer_region::ptr import_vulkan_os_buffer(vulkan_engine& engine, intptr_t external_handle, size_t buffer_size);
vulkan_buffer_region::ptr import_vulkan_native_buffer(vulkan_engine& engine, void* external_handle, size_t buffer_size);

class vulkan_memory_allocator final : public std::enable_shared_from_this<vulkan_memory_allocator> {
public:
    vulkan_memory_allocator(vulkan_engine& engine, VkDeviceSize alignment, VkDeviceSize preferred_block_size);

    vulkan_buffer_region::ptr allocate(size_t size);
    vulkan_memory_allocator_stats get_stats() const;

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
    VkDeviceSize align_size(VkDeviceSize size) const;

    vulkan_engine* _engine = nullptr;
    VkDeviceSize _alignment = 1;
    VkDeviceSize _preferred_block_size = 1;
    mutable std::mutex _mutex;
    std::vector<block> _blocks;
    vulkan_memory_allocator_stats _stats;
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
        return _region->get_allocation()->buffer;
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
    void* mapped_data() const;
    vulkan_buffer_allocation::ptr allocate_staging(size_t size) const;

    vulkan_buffer_region::ptr _region;
    VkDeviceSize _view_offset = 0;
    mutable std::mutex _lock_mutex;
    vulkan_buffer_allocation::ptr _lock_staging;
    size_t _lock_count = 0;
    bool _write_access = false;
};

}  // namespace vulkan
}  // namespace cldnn
