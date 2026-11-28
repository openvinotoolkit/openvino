// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <new>
#include <optional>
#include <utility>

#include "openvino/core/except.hpp"
#include "vulkan_api.hpp"
#include "vulkan_engine.hpp"
#include "vulkan_memory.hpp"

namespace cldnn::vulkan {
namespace {

class vulkan_bad_alloc final : public std::bad_alloc {
public:
    vulkan_bad_alloc(const char* operation, VkResult result) noexcept {
        std::snprintf(_message, sizeof(_message), "[GPU][Vulkan] %s failed with %s", operation, vk_result_name(result));
    }

    const char* what() const noexcept override {
        return _message;
    }

private:
    char _message[96]{};
};

void check_vulkan_memory_result(VkResult result, const char* operation) {
    if (result == VK_ERROR_OUT_OF_HOST_MEMORY || result == VK_ERROR_OUT_OF_DEVICE_MEMORY) {
        throw vulkan_bad_alloc(operation, result);
    }
    check_vk_result(result, operation);
}

std::optional<vulkan_memory_type_selection> find_memory_type(const VkPhysicalDeviceMemoryProperties& properties,
                                                             uint32_t allowed_types,
                                                             VkMemoryPropertyFlags required,
                                                             VkMemoryPropertyFlags forbidden = 0) {
    for (uint32_t index = 0; index < properties.memoryTypeCount; ++index) {
        if ((allowed_types & (1U << index)) == 0) {
            continue;
        }

        const auto flags = properties.memoryTypes[index].propertyFlags;
        if ((flags & required) == required && (flags & forbidden) == 0) {
            return vulkan_memory_type_selection{index, flags};
        }
    }
    return std::nullopt;
}

}  // namespace

vulkan_buffer_allocation::ptr vulkan_buffer_allocation::create(vulkan_engine& engine, size_t requested_size, vulkan_buffer_memory_usage usage) {
    const VkDeviceSize buffer_size = std::max<VkDeviceSize>(requested_size, 1);
    const auto device_owner = engine.get_vulkan_device_object();
    if (buffer_size > device_owner->get_max_buffer_size()) {
        throw vulkan_bad_alloc("reserve buffer size", VK_ERROR_OUT_OF_DEVICE_MEMORY);
    }
    auto allocation = std::make_shared<vulkan_buffer_allocation>(device_owner, buffer_size);
    const VkDevice device = allocation->_device;

    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = buffer_size;
    buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (usage == vulkan_buffer_memory_usage::device) {
        buffer_info.usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    }
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer = VK_NULL_HANDLE;
    check_vulkan_memory_result(vkCreateBuffer(device, &buffer_info, nullptr, &buffer), "vkCreateBuffer");
    allocation->_buffer = buffer;

    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device, buffer, &requirements);
    const auto& memory_properties = device_owner->get_memory_properties();
    const auto& info = engine.get_device_info();
    const auto prefer_unified_memory = info.dev_type != device_type::discrete_gpu;
    const auto memory_type = select_vulkan_memory_type(memory_properties, requirements.memoryTypeBits, usage, prefer_unified_memory);
    const auto heap_index = memory_properties.memoryTypes[memory_type.index].heapIndex;

    VkMemoryAllocateInfo allocation_info{};
    allocation_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocation_info.allocationSize = requirements.size;
    allocation_info.memoryTypeIndex = memory_type.index;

    static const auto host_memory = cldnn::engine::get_host_memory_size();
    const auto device_capacity = prefer_unified_memory ? VkDeviceSize{0} : info.max_global_mem_size;
    const auto memory_limit = host_memory + std::min(device_capacity, (std::numeric_limits<VkDeviceSize>::max)() - host_memory);
    if (allocation_info.allocationSize > device_owner->get_max_memory_allocation_size() ||
        !device_owner->try_reserve_memory(heap_index, allocation_info.allocationSize, memory_limit)) {
        throw vulkan_bad_alloc("reserve memory budget", VK_ERROR_OUT_OF_DEVICE_MEMORY);
    }
    // The owner releases the reservation if any subsequent Vulkan call fails.
    allocation->_memory_size = allocation_info.allocationSize;
    allocation->_memory_heap = heap_index;

    VkDeviceMemory device_memory = VK_NULL_HANDLE;
    check_vulkan_memory_result(vkAllocateMemory(device, &allocation_info, nullptr, &device_memory), "vkAllocateMemory");
    allocation->_memory = device_memory;
    allocation->_memory_flags = memory_type.flags;

    check_vulkan_memory_result(vkBindBufferMemory(device, buffer, device_memory, 0), "vkBindBufferMemory");

    if (allocation->is_host_visible()) {
        void* mapped_data = nullptr;
        check_vulkan_memory_result(vkMapMemory(device, device_memory, 0, VK_WHOLE_SIZE, 0, &mapped_data), "vkMapMemory");
        allocation->_mapped_data = mapped_data;
    }

    return allocation;
}

vulkan_memory_type_selection select_vulkan_memory_type(const VkPhysicalDeviceMemoryProperties& properties,
                                                       uint32_t allowed_types,
                                                       vulkan_buffer_memory_usage usage,
                                                       bool prefer_unified_memory) {
    const auto device_local = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    const auto host_visible = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    const auto host_coherent = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    if (usage == vulkan_buffer_memory_usage::host_staging) {
        if (const auto selected = find_memory_type(properties, allowed_types, host_visible | host_coherent, device_local)) {
            return *selected;
        }
        if (const auto selected = find_memory_type(properties, allowed_types, host_visible, device_local)) {
            return *selected;
        }
        if (const auto selected = find_memory_type(properties, allowed_types, host_visible | host_coherent)) {
            return *selected;
        }
        if (const auto selected = find_memory_type(properties, allowed_types, host_visible)) {
            return *selected;
        }
        OPENVINO_THROW("[GPU][Vulkan] No host-visible memory type is available for a staging buffer");
    }

    if (prefer_unified_memory) {
        if (const auto selected = find_memory_type(properties, allowed_types, device_local | host_visible | host_coherent)) {
            return *selected;
        }
        if (const auto selected = find_memory_type(properties, allowed_types, device_local | host_visible)) {
            return *selected;
        }
    } else if (const auto selected = find_memory_type(properties, allowed_types, device_local, host_visible)) {
        return *selected;
    }
    if (const auto selected = find_memory_type(properties, allowed_types, device_local)) {
        return *selected;
    }
    if (const auto selected = find_memory_type(properties, allowed_types, host_visible | host_coherent)) {
        return *selected;
    }
    if (const auto selected = find_memory_type(properties, allowed_types, host_visible)) {
        return *selected;
    }
    if (const auto selected = find_memory_type(properties, allowed_types, 0)) {
        return *selected;
    }
    OPENVINO_THROW("[GPU][Vulkan] No compatible memory type is available for a storage buffer");
}

vulkan_non_coherent_range align_vulkan_non_coherent_range(VkDeviceSize offset, VkDeviceSize size, VkDeviceSize allocation_size, VkDeviceSize atom_size) {
    OPENVINO_ASSERT(allocation_size > 0, "[GPU][Vulkan] Mapped allocation size cannot be zero");
    OPENVINO_ASSERT(atom_size > 0, "[GPU][Vulkan] nonCoherentAtomSize cannot be zero");
    OPENVINO_ASSERT(offset <= allocation_size, "[GPU][Vulkan] Mapped range offset exceeds allocation size");
    const auto normalized_size = size == VK_WHOLE_SIZE ? allocation_size - offset : size;
    OPENVINO_ASSERT(normalized_size <= allocation_size - offset, "[GPU][Vulkan] Mapped range exceeds allocation size");
    if (normalized_size == 0) {
        return {offset, 0};
    }

    const auto aligned_offset = offset - offset % atom_size;
    const auto touched_end = offset + normalized_size;
    const auto end_remainder = touched_end % atom_size;
    const auto end_padding = end_remainder == 0 ? 0 : atom_size - end_remainder;
    const auto aligned_end = end_padding <= allocation_size - touched_end ? touched_end + end_padding : allocation_size;
    return {aligned_offset, aligned_end - aligned_offset};
}

vulkan_buffer_allocation::vulkan_buffer_allocation(std::shared_ptr<vulkan_device> device_owner, VkDeviceSize size)
    : _device(device_owner->get_device()),
      _size(size),
      _non_coherent_atom_size(device_owner->get_non_coherent_atom_size()),
      _device_owner(std::move(device_owner)) {}

vulkan_buffer_allocation::~vulkan_buffer_allocation() {
    unmap();
    if (_buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(_device, _buffer, nullptr);
    }
    if (_memory != VK_NULL_HANDLE) {
        vkFreeMemory(_device, _memory, nullptr);
    }
    if (_memory_size > 0) {
        _device_owner->release_memory(_memory_heap, _memory_size);
    }
}

void vulkan_buffer_allocation::unmap() noexcept {
    if (is_mapped()) {
        vkUnmapMemory(_device, _memory);
        _mapped_data = nullptr;
    }
}

void vulkan_buffer_allocation::flush(VkDeviceSize offset, VkDeviceSize range_size) const {
    OPENVINO_ASSERT(is_mapped(), "[GPU][Vulkan] Cannot flush unmapped memory");
    OPENVINO_ASSERT(offset <= _size, "[GPU][Vulkan] Flush offset exceeds buffer size");
    const auto normalized_size = range_size == VK_WHOLE_SIZE ? _size - offset : range_size;
    OPENVINO_ASSERT(normalized_size <= _size - offset, "[GPU][Vulkan] Flush range exceeds buffer size");
    if (normalized_size == 0 || is_host_coherent()) {
        return;
    }
    const auto aligned_range = align_vulkan_non_coherent_range(offset, normalized_size, _memory_size, _non_coherent_atom_size);
    std::lock_guard<std::mutex> lock(_visibility_mutex);
    VkMappedMemoryRange range{};
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = _memory;
    range.offset = aligned_range.offset;
    range.size = aligned_range.size;
    check_vulkan_memory_result(vkFlushMappedMemoryRanges(_device, 1, &range), "vkFlushMappedMemoryRanges");
}

void vulkan_buffer_allocation::invalidate(VkDeviceSize offset, VkDeviceSize range_size) const {
    OPENVINO_ASSERT(is_mapped(), "[GPU][Vulkan] Cannot invalidate unmapped memory");
    OPENVINO_ASSERT(offset <= _size, "[GPU][Vulkan] Invalidate offset exceeds buffer size");
    const auto normalized_size = range_size == VK_WHOLE_SIZE ? _size - offset : range_size;
    OPENVINO_ASSERT(normalized_size <= _size - offset, "[GPU][Vulkan] Invalidate range exceeds buffer size");
    if (normalized_size == 0 || is_host_coherent()) {
        return;
    }
    const auto aligned_range = align_vulkan_non_coherent_range(offset, normalized_size, _memory_size, _non_coherent_atom_size);
    std::lock_guard<std::mutex> lock(_visibility_mutex);
    VkMappedMemoryRange range{};
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = _memory;
    range.offset = aligned_range.offset;
    range.size = aligned_range.size;
    check_vulkan_memory_result(vkInvalidateMappedMemoryRanges(_device, 1, &range), "vkInvalidateMappedMemoryRanges");
}

vulkan_buffer_region::vulkan_buffer_region(std::weak_ptr<vulkan_memory_allocator> owner,
                                           vulkan_buffer_allocation::ptr allocation,
                                           VkDeviceSize offset,
                                           VkDeviceSize size)
    : _owner(std::move(owner)),
      _allocation(std::move(allocation)),
      _offset(offset),
      _size(size) {}

vulkan_buffer_region::~vulkan_buffer_region() {
    if (auto owner = _owner.lock()) {
        try {
            owner->release(_allocation, _offset, _size);
        } catch (...) {
            // Destructors must not throw while unwinding. The backing block remains owned by the allocator.
        }
    }
}

vulkan_memory_allocator::vulkan_memory_allocator(VkDeviceSize alignment,
                                                 VkDeviceSize preferred_block_size,
                                                 VkDeviceSize max_block_size,
                                                 block_allocator allocate_block)
    : _alignment(std::max<VkDeviceSize>(alignment, 1)),
      _preferred_block_size(std::max<VkDeviceSize>(preferred_block_size, 1)),
      _max_block_size(std::max<VkDeviceSize>(max_block_size, 1)),
      _allocate_block(std::move(allocate_block)) {}

VkDeviceSize vulkan_memory_allocator::align_size(VkDeviceSize size) const {
    const auto remainder = size % _alignment;
    if (remainder == 0) {
        return size;
    }

    const auto padding = _alignment - remainder;
    OPENVINO_ASSERT(size <= (std::numeric_limits<VkDeviceSize>::max)() - padding, "[GPU][Vulkan] Buffer suballocation size overflows VkDeviceSize");
    return size + padding;
}

vulkan_buffer_region::ptr vulkan_memory_allocator::allocate(size_t size) {
    const auto requested_size = std::max<VkDeviceSize>(static_cast<VkDeviceSize>(size), 1);
    const auto reserved_size = align_size(requested_size);
    if (reserved_size > _max_block_size) {
        throw vulkan_bad_alloc("reserve padded buffer", VK_ERROR_OUT_OF_DEVICE_MEMORY);
    }
    std::lock_guard<std::mutex> lock(_mutex);

    size_t selected_block = _blocks.size();
    size_t selected_range = 0;
    VkDeviceSize smallest_range = (std::numeric_limits<VkDeviceSize>::max)();
    for (size_t block_index = 0; block_index < _blocks.size(); ++block_index) {
        const auto& current_block = _blocks[block_index];
        for (size_t range_index = 0; range_index < current_block.free_ranges.size(); ++range_index) {
            const auto& range = current_block.free_ranges[range_index];
            if (range.size >= reserved_size && range.size < smallest_range) {
                selected_block = block_index;
                selected_range = range_index;
                smallest_range = range.size;
            }
        }
    }

    if (selected_block != _blocks.size()) {
        auto& current_block = _blocks[selected_block];
        const auto offset = current_block.free_ranges[selected_range].offset;
        auto region = vulkan_buffer_region::ptr(new vulkan_buffer_region({}, current_block.allocation, offset, reserved_size));
        // A returned region can insert one free range. Reserve space before publishing ownership.
        current_block.free_ranges.reserve(current_block.free_ranges.size() + current_block.live_regions + 1);
        auto& range = current_block.free_ranges[selected_range];
        range.offset += reserved_size;
        range.size -= reserved_size;
        if (range.size == 0) {
            current_block.free_ranges.erase(current_block.free_ranges.begin() + selected_range);
        }
        ++current_block.live_regions;
        region->_owner = weak_from_this();
        return region;
    }

    auto block_size = reserved_size;
    if (reserved_size < _preferred_block_size) {
        const auto balanced_size =
            static_cast<VkDeviceSize>(std::sqrt(static_cast<long double>(reserved_size) * static_cast<long double>(_preferred_block_size)));
        block_size = std::min(_preferred_block_size, align_size(std::max(reserved_size, balanced_size)));
    }
    if (block_size > _max_block_size) {
        block_size = reserved_size;
    }

    vulkan_buffer_allocation::ptr allocation;
    try {
        allocation = _allocate_block(static_cast<size_t>(block_size));
    } catch (const std::bad_alloc&) {
        if (!reclaim_idle_blocks_locked() && block_size == reserved_size) {
            throw;
        }
        block_size = reserved_size;
        allocation = _allocate_block(static_cast<size_t>(block_size));
    }

    auto region = vulkan_buffer_region::ptr(new vulkan_buffer_region({}, allocation, 0, reserved_size));
    block new_block;
    new_block.allocation = allocation;
    new_block.live_regions = 1;
    new_block.cacheable = block_size <= _preferred_block_size;
    new_block.free_ranges.reserve(2);
    if (block_size > reserved_size) {
        new_block.free_ranges.push_back({reserved_size, block_size - reserved_size});
    }
    _blocks.push_back(std::move(new_block));

    region->_owner = weak_from_this();
    return region;
}

vulkan_buffer_allocation::ptr vulkan_memory_allocator::allocate_staging(size_t size, const block_allocator& allocate_block) {
    try {
        return allocate_block(size);
    } catch (const std::bad_alloc&) {
        const std::lock_guard<std::mutex> lock(_mutex);
        if (!reclaim_idle_blocks_locked()) {
            throw;
        }
    }
    return allocate_block(size);
}

bool vulkan_memory_allocator::reclaim_idle_blocks_locked() {
    const auto live_blocks_end = std::remove_if(_blocks.begin(), _blocks.end(), [](const block& candidate) {
        return candidate.live_regions == 0;
    });
    if (live_blocks_end == _blocks.end()) {
        return false;
    }
    _blocks.erase(live_blocks_end, _blocks.end());
    return true;
}

void vulkan_memory_allocator::release(const vulkan_buffer_allocation::ptr& allocation, VkDeviceSize offset, VkDeviceSize size) {
    std::lock_guard<std::mutex> lock(_mutex);
    const auto block_it = std::find_if(_blocks.begin(), _blocks.end(), [&](const block& candidate) {
        return candidate.allocation == allocation;
    });
    if (block_it == _blocks.end() || block_it->live_regions == 0) {
        return;
    }

    --block_it->live_regions;
    block_it->free_ranges.push_back({offset, size});
    std::sort(block_it->free_ranges.begin(), block_it->free_ranges.end(), [](const free_range& lhs, const free_range& rhs) {
        return lhs.offset < rhs.offset;
    });

    size_t coalesced_count = 0;
    for (const auto& range : block_it->free_ranges) {
        if (coalesced_count == 0 || block_it->free_ranges[coalesced_count - 1].offset + block_it->free_ranges[coalesced_count - 1].size < range.offset) {
            block_it->free_ranges[coalesced_count++] = range;
        } else {
            auto& previous = block_it->free_ranges[coalesced_count - 1];
            const auto range_end = range.offset + range.size;
            const auto previous_end = previous.offset + previous.size;
            previous.size = std::max(previous_end, range_end) - previous.offset;
        }
    }
    block_it->free_ranges.resize(coalesced_count);

    const bool fully_empty = block_it->live_regions == 0 && block_it->free_ranges.size() == 1 && block_it->free_ranges.front().offset == 0 &&
                             block_it->free_ranges.front().size == block_it->allocation->get_size();
    if (!fully_empty) {
        return;
    }

    if (block_it->cacheable) {
        const auto retained_it = std::max_element(_blocks.begin(), _blocks.end(), [&](const block& lhs, const block& rhs) {
            const auto lhs_size = &lhs != &*block_it && lhs.cacheable && lhs.live_regions == 0 ? lhs.allocation->get_size() : 0;
            const auto rhs_size = &rhs != &*block_it && rhs.cacheable && rhs.live_regions == 0 ? rhs.allocation->get_size() : 0;
            return lhs_size < rhs_size;
        });
        if (retained_it == _blocks.end() || retained_it == block_it || !retained_it->cacheable || retained_it->live_regions != 0) {
            return;
        }
        if (retained_it->allocation->get_size() < block_it->allocation->get_size()) {
            _blocks.erase(retained_it);
            return;
        }
    } else {
        _blocks.erase(block_it);
        return;
    }

    _blocks.erase(block_it);
}

}  // namespace cldnn::vulkan
