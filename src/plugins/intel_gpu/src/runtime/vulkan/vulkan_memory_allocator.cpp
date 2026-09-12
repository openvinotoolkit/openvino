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
#include "vulkan_engine.hpp"
#include "vulkan_memory.hpp"
#include "vulkan_memory_internal.hpp"

namespace cldnn::vulkan {
namespace {

class vulkan_bad_alloc final : public std::bad_alloc {
public:
    vulkan_bad_alloc(const char* operation, VkResult result) noexcept {
        std::snprintf(_message, sizeof(_message), "[GPU][Vulkan] %s failed with VkResult %d", operation, static_cast<int>(result));
    }

    const char* what() const noexcept override {
        return _message;
    }

private:
    char _message[96]{};
};

}  // namespace

void detail::check_vulkan_memory_result(VkResult result, const char* operation) {
    if (result == VK_ERROR_OUT_OF_HOST_MEMORY || result == VK_ERROR_OUT_OF_DEVICE_MEMORY) {
        throw vulkan_bad_alloc(operation, result);
    }
    OPENVINO_ASSERT(result == VK_SUCCESS, "[GPU][Vulkan] ", operation, " failed with VkResult ", static_cast<int>(result));
}

bool detail::transition_foreign_buffer_ownership(const std::shared_ptr<vulkan_device>& device_owner, VkBuffer buffer, bool acquire) noexcept {
    if (device_owner == nullptr || buffer == VK_NULL_HANDLE) {
        return false;
    }

    const VkDevice device = device_owner->get_device();
    const auto queue_family = device_owner->get_compute_queue_family();
    std::lock_guard<std::mutex> lock(device_owner->get_queue_mutex());

    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    const auto cleanup = [&]() {
        if (fence != VK_NULL_HANDLE) {
            vkDestroyFence(device, fence, nullptr);
        }
        if (pool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device, pool, nullptr);
        }
    };

    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    pool_info.queueFamilyIndex = queue_family;
    if (vkCreateCommandPool(device, &pool_info, nullptr, &pool) != VK_SUCCESS) {
        cleanup();
        return false;
    }

    VkCommandBufferAllocateInfo command_info{};
    command_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    command_info.commandPool = pool;
    command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_info.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(device, &command_info, &command_buffer) != VK_SUCCESS) {
        cleanup();
        return false;
    }

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(command_buffer, &begin_info) != VK_SUCCESS) {
        cleanup();
        return false;
    }

    VkBufferMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    barrier.srcStageMask = acquire ? VK_PIPELINE_STAGE_2_NONE : VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    barrier.srcAccessMask = acquire ? VK_ACCESS_2_NONE : VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    barrier.dstStageMask = acquire ? VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT : VK_PIPELINE_STAGE_2_NONE;
    barrier.dstAccessMask = acquire ? VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT : VK_ACCESS_2_NONE;
    barrier.srcQueueFamilyIndex = acquire ? VK_QUEUE_FAMILY_FOREIGN_EXT : queue_family;
    barrier.dstQueueFamilyIndex = acquire ? queue_family : VK_QUEUE_FAMILY_FOREIGN_EXT;
    barrier.buffer = buffer;
    barrier.offset = 0;
    barrier.size = VK_WHOLE_SIZE;

    VkDependencyInfo dependency_info{};
    dependency_info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependency_info.bufferMemoryBarrierCount = 1;
    dependency_info.pBufferMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(command_buffer, &dependency_info);
    if (vkEndCommandBuffer(command_buffer) != VK_SUCCESS) {
        cleanup();
        return false;
    }

    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (vkCreateFence(device, &fence_info, nullptr, &fence) != VK_SUCCESS) {
        cleanup();
        return false;
    }

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;
    const auto submit_result = vkQueueSubmit(device_owner->get_compute_queue(), 1, &submit_info, fence);
    const auto wait_result = submit_result == VK_SUCCESS ? vkWaitForFences(device, 1, &fence, VK_TRUE, (std::numeric_limits<uint64_t>::max)()) : submit_result;
    cleanup();
    return submit_result == VK_SUCCESS && wait_result == VK_SUCCESS;
}

namespace {

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

vulkan_buffer_allocation::ptr detail::allocate_vulkan_buffer(vulkan_engine& engine, size_t requested_size, vulkan_buffer_memory_usage usage) {
    const VkDeviceSize buffer_size = std::max<VkDeviceSize>(requested_size, 1);

    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = buffer_size;
    buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (usage == vulkan_buffer_memory_usage::device) {
        buffer_info.usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    }
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer = VK_NULL_HANDLE;
    check_vulkan_memory_result(vkCreateBuffer(engine.get_device_handle(), &buffer_info, nullptr, &buffer), "vkCreateBuffer");

    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(engine.get_device_handle(), buffer, &requirements);
    VkPhysicalDeviceMemoryProperties memory_properties{};
    vkGetPhysicalDeviceMemoryProperties(engine.get_physical_device(), &memory_properties);
    const auto prefer_unified_memory = engine.get_device_info().dev_type != device_type::discrete_gpu;
    const auto memory_type = select_vulkan_memory_type(memory_properties, requirements.memoryTypeBits, usage, prefer_unified_memory);

    VkMemoryAllocateInfo allocation_info{};
    allocation_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocation_info.allocationSize = requirements.size;
    allocation_info.memoryTypeIndex = memory_type.index;

    VkDeviceMemory device_memory = VK_NULL_HANDLE;
    const auto allocation_result = vkAllocateMemory(engine.get_device_handle(), &allocation_info, nullptr, &device_memory);
    if (allocation_result != VK_SUCCESS) {
        vkDestroyBuffer(engine.get_device_handle(), buffer, nullptr);
        check_vulkan_memory_result(allocation_result, "vkAllocateMemory");
    }

    const auto bind_result = vkBindBufferMemory(engine.get_device_handle(), buffer, device_memory, 0);
    if (bind_result != VK_SUCCESS) {
        vkDestroyBuffer(engine.get_device_handle(), buffer, nullptr);
        vkFreeMemory(engine.get_device_handle(), device_memory, nullptr);
        check_vulkan_memory_result(bind_result, "vkBindBufferMemory");
    }

    void* mapped_data = nullptr;
    if ((memory_type.flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
        const auto map_result = vkMapMemory(engine.get_device_handle(), device_memory, 0, VK_WHOLE_SIZE, 0, &mapped_data);
        if (map_result != VK_SUCCESS) {
            vkDestroyBuffer(engine.get_device_handle(), buffer, nullptr);
            vkFreeMemory(engine.get_device_handle(), device_memory, nullptr);
            check_vulkan_memory_result(map_result, "vkMapMemory");
        }
    }

    return std::make_shared<vulkan_buffer_allocation>(engine.get_vulkan_device_object(),
                                                      engine.get_device_handle(),
                                                      buffer,
                                                      device_memory,
                                                      mapped_data,
                                                      buffer_size,
                                                      allocation_info.allocationSize,
                                                      memory_type.flags,
                                                      engine.get_non_coherent_atom_size());
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

vulkan_buffer_allocation::vulkan_buffer_allocation(std::shared_ptr<vulkan_device> device_owner,
                                                   VkDevice device,
                                                   VkBuffer buffer,
                                                   VkDeviceMemory memory,
                                                   void* mapped_data,
                                                   VkDeviceSize size,
                                                   VkDeviceSize memory_size,
                                                   VkMemoryPropertyFlags memory_flags,
                                                   VkDeviceSize non_coherent_atom_size,
                                                   bool release_to_foreign)
    : device(device),
      buffer(buffer),
      memory(memory),
      mapped_data(mapped_data),
      size(size),
      memory_size(memory_size),
      memory_flags(memory_flags),
      non_coherent_atom_size(non_coherent_atom_size),
      _device_owner(std::move(device_owner)),
      _release_to_foreign(release_to_foreign) {}

vulkan_buffer_allocation::~vulkan_buffer_allocation() {
    if (_release_to_foreign) {
        detail::transition_foreign_buffer_ownership(_device_owner, buffer, false);
    }
    if (mapped_data != nullptr) {
        vkUnmapMemory(device, memory);
    }
    if (buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, buffer, nullptr);
    }
    if (memory != VK_NULL_HANDLE) {
        vkFreeMemory(device, memory, nullptr);
    }
}

void vulkan_buffer_allocation::flush(VkDeviceSize offset, VkDeviceSize range_size) const {
    OPENVINO_ASSERT(is_host_visible(), "[GPU][Vulkan] Cannot flush non-host-visible memory");
    OPENVINO_ASSERT(offset <= size, "[GPU][Vulkan] Flush offset exceeds buffer size");
    const auto normalized_size = range_size == VK_WHOLE_SIZE ? size - offset : range_size;
    OPENVINO_ASSERT(normalized_size <= size - offset, "[GPU][Vulkan] Flush range exceeds buffer size");
    if (normalized_size == 0 || (memory_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0) {
        return;
    }
    const auto aligned_range = align_vulkan_non_coherent_range(offset, normalized_size, memory_size, non_coherent_atom_size);
    std::lock_guard<std::mutex> lock(_visibility_mutex);
    VkMappedMemoryRange range{};
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = memory;
    range.offset = aligned_range.offset;
    range.size = aligned_range.size;
    detail::check_vulkan_memory_result(vkFlushMappedMemoryRanges(device, 1, &range), "vkFlushMappedMemoryRanges");
}

void vulkan_buffer_allocation::invalidate(VkDeviceSize offset, VkDeviceSize range_size) const {
    OPENVINO_ASSERT(is_host_visible(), "[GPU][Vulkan] Cannot invalidate non-host-visible memory");
    OPENVINO_ASSERT(offset <= size, "[GPU][Vulkan] Invalidate offset exceeds buffer size");
    const auto normalized_size = range_size == VK_WHOLE_SIZE ? size - offset : range_size;
    OPENVINO_ASSERT(normalized_size <= size - offset, "[GPU][Vulkan] Invalidate range exceeds buffer size");
    if (normalized_size == 0 || (memory_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0) {
        return;
    }
    const auto aligned_range = align_vulkan_non_coherent_range(offset, normalized_size, memory_size, non_coherent_atom_size);
    std::lock_guard<std::mutex> lock(_visibility_mutex);
    VkMappedMemoryRange range{};
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = memory;
    range.offset = aligned_range.offset;
    range.size = aligned_range.size;
    detail::check_vulkan_memory_result(vkInvalidateMappedMemoryRanges(device, 1, &range), "vkInvalidateMappedMemoryRanges");
}

vulkan_buffer_region::vulkan_buffer_region(std::weak_ptr<vulkan_memory_allocator> owner,
                                           vulkan_buffer_allocation::ptr allocation,
                                           VkDeviceSize offset,
                                           VkDeviceSize size)
    : _owner(std::move(owner)),
      _allocation(std::move(allocation)),
      _offset(offset),
      _size(size) {}

vulkan_buffer_region::ptr vulkan_buffer_region::create_external(vulkan_buffer_allocation::ptr allocation, VkDeviceSize size) {
    OPENVINO_ASSERT(allocation != nullptr && size <= allocation->size, "[GPU][Vulkan] External buffer region exceeds the imported allocation");
    return vulkan_buffer_region::ptr(new vulkan_buffer_region({}, std::move(allocation), 0, size));
}

vulkan_buffer_region::~vulkan_buffer_region() {
    if (auto owner = _owner.lock()) {
        try {
            owner->release(_allocation, _offset, _size);
        } catch (...) {
            // Destructors must not throw while unwinding. The backing block remains owned by the allocator.
        }
    }
}

vulkan_memory_allocator::vulkan_memory_allocator(vulkan_engine& engine, VkDeviceSize alignment, VkDeviceSize preferred_block_size)
    : _engine(&engine),
      _alignment(std::max<VkDeviceSize>(alignment, 1)),
      _preferred_block_size(std::max<VkDeviceSize>(preferred_block_size, 1)) {
    _stats.alignment = _alignment;
    _stats.preferred_block_size = _preferred_block_size;
}

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
    auto reserved_size = align_size(requested_size);
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
        auto& range = current_block.free_ranges[selected_range];
        const auto offset = range.offset;
        range.offset += reserved_size;
        range.size -= reserved_size;
        if (range.size == 0) {
            current_block.free_ranges.erase(current_block.free_ranges.begin() + selected_range);
        }
        ++current_block.live_regions;
        ++_stats.region_allocations;
        ++_stats.region_reuses;
        return vulkan_buffer_region::ptr(new vulkan_buffer_region(shared_from_this(), current_block.allocation, offset, reserved_size));
    }

    const auto max_block_size = std::max<VkDeviceSize>(_engine->get_device_info().max_alloc_mem_size, 1);
    auto block_size = reserved_size;
    if (reserved_size < _preferred_block_size) {
        const auto balanced_size =
            static_cast<VkDeviceSize>(std::sqrt(static_cast<long double>(reserved_size) * static_cast<long double>(_preferred_block_size)));
        block_size = std::min(_preferred_block_size, align_size(std::max(reserved_size, balanced_size)));
    }
    if (block_size > max_block_size && requested_size <= max_block_size) {
        block_size = requested_size;
        reserved_size = requested_size;
    }

    vulkan_buffer_allocation::ptr allocation;
    try {
        allocation = detail::allocate_vulkan_buffer(*_engine, static_cast<size_t>(block_size), vulkan_buffer_memory_usage::device);
    } catch (const std::bad_alloc&) {
        if (block_size == requested_size) {
            throw;
        }
        block_size = requested_size;
        reserved_size = requested_size;
        allocation = detail::allocate_vulkan_buffer(*_engine, size, vulkan_buffer_memory_usage::device);
    }

    block new_block;
    new_block.allocation = allocation;
    new_block.live_regions = 1;
    new_block.cacheable = block_size <= _preferred_block_size;
    if (block_size > reserved_size) {
        new_block.free_ranges.push_back({reserved_size, block_size - reserved_size});
    }
    _blocks.push_back(std::move(new_block));

    ++_stats.region_allocations;
    ++_stats.block_allocations;
    _stats.current_blocks = _blocks.size();
    _stats.current_reserved_bytes += block_size;
    _stats.peak_reserved_bytes = std::max(_stats.peak_reserved_bytes, _stats.current_reserved_bytes);
    return vulkan_buffer_region::ptr(new vulkan_buffer_region(shared_from_this(), std::move(allocation), 0, reserved_size));
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

    std::vector<free_range> coalesced;
    coalesced.reserve(block_it->free_ranges.size());
    for (const auto& range : block_it->free_ranges) {
        if (coalesced.empty() || coalesced.back().offset + coalesced.back().size < range.offset) {
            coalesced.push_back(range);
        } else {
            const auto range_end = range.offset + range.size;
            const auto previous_end = coalesced.back().offset + coalesced.back().size;
            coalesced.back().size = std::max(previous_end, range_end) - coalesced.back().offset;
        }
    }
    block_it->free_ranges = std::move(coalesced);

    const bool fully_empty = block_it->live_regions == 0 && block_it->free_ranges.size() == 1 && block_it->free_ranges.front().offset == 0 &&
                             block_it->free_ranges.front().size == block_it->allocation->size;
    if (!fully_empty) {
        return;
    }

    if (block_it->cacheable) {
        const auto retained_it = std::max_element(_blocks.begin(), _blocks.end(), [&](const block& lhs, const block& rhs) {
            const auto lhs_size = &lhs != &*block_it && lhs.cacheable && lhs.live_regions == 0 ? lhs.allocation->size : 0;
            const auto rhs_size = &rhs != &*block_it && rhs.cacheable && rhs.live_regions == 0 ? rhs.allocation->size : 0;
            return lhs_size < rhs_size;
        });
        if (retained_it == _blocks.end() || retained_it == block_it || !retained_it->cacheable || retained_it->live_regions != 0) {
            return;
        }
        if (retained_it->allocation->size < block_it->allocation->size) {
            _stats.current_reserved_bytes -= retained_it->allocation->size;
            ++_stats.block_releases;
            _blocks.erase(retained_it);
            _stats.current_blocks = _blocks.size();
            return;
        }
    } else {
        _stats.current_reserved_bytes -= block_it->allocation->size;
        ++_stats.block_releases;
        _blocks.erase(block_it);
        _stats.current_blocks = _blocks.size();
        return;
    }

    _stats.current_reserved_bytes -= block_it->allocation->size;
    ++_stats.block_releases;
    _blocks.erase(block_it);
    _stats.current_blocks = _blocks.size();
}

vulkan_memory_allocator_stats vulkan_memory_allocator::get_stats() const {
    std::lock_guard<std::mutex> lock(_mutex);
    return _stats;
}

}  // namespace cldnn::vulkan
