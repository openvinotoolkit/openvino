// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vulkan_memory.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <new>
#include <optional>
#include <utility>

#include "intel_gpu/runtime/stream.hpp"
#include "openvino/core/except.hpp"
#include "vulkan_engine.hpp"
#include "vulkan_stream.hpp"

namespace cldnn {
namespace vulkan {
namespace {

class vulkan_bad_alloc final : public std::bad_alloc {
public:
    vulkan_bad_alloc(const char* operation, VkResult result) noexcept {
        std::snprintf(_message,
                      sizeof(_message),
                      "[GPU][Vulkan] %s failed with VkResult %d",
                      operation,
                      static_cast<int>(result));
    }

    const char* what() const noexcept override {
        return _message;
    }

private:
    char _message[96]{};
};

void check_vk_result(VkResult result, const char* operation) {
    if (result == VK_ERROR_OUT_OF_HOST_MEMORY || result == VK_ERROR_OUT_OF_DEVICE_MEMORY) {
        throw vulkan_bad_alloc(operation, result);
    }
    OPENVINO_ASSERT(result == VK_SUCCESS, "[GPU][Vulkan] ", operation, " failed with VkResult ", static_cast<int>(result));
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

vulkan_buffer_allocation::ptr allocate_buffer(vulkan_engine& engine, size_t requested_size, vulkan_buffer_memory_usage usage) {
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
    check_vk_result(vkCreateBuffer(engine.get_device_handle(), &buffer_info, nullptr, &buffer), "vkCreateBuffer");

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
        check_vk_result(allocation_result, "vkAllocateMemory");
    }

    const auto bind_result = vkBindBufferMemory(engine.get_device_handle(), buffer, device_memory, 0);
    if (bind_result != VK_SUCCESS) {
        vkDestroyBuffer(engine.get_device_handle(), buffer, nullptr);
        vkFreeMemory(engine.get_device_handle(), device_memory, nullptr);
        check_vk_result(bind_result, "vkBindBufferMemory");
    }

    void* mapped_data = nullptr;
    if ((memory_type.flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
        const auto map_result = vkMapMemory(engine.get_device_handle(), device_memory, 0, VK_WHOLE_SIZE, 0, &mapped_data);
        if (map_result != VK_SUCCESS) {
            vkDestroyBuffer(engine.get_device_handle(), buffer, nullptr);
            vkFreeMemory(engine.get_device_handle(), device_memory, nullptr);
            check_vk_result(map_result, "vkMapMemory");
        }
    }

    return std::make_shared<vulkan_buffer_allocation>(engine.get_vulkan_device_object(),
                                                      engine.get_device_handle(),
                                                      buffer,
                                                      device_memory,
                                                      mapped_data,
                                                      buffer_size,
                                                      memory_type.flags);
}

const vulkan_stream& validate_stream(const stream& stream) {
    const auto* result = dynamic_cast<const vulkan_stream*>(&stream);
    OPENVINO_ASSERT(result != nullptr, "[GPU][Vulkan] Memory operation received a stream from another backend");
    return *result;
}

}  // namespace

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

vulkan_buffer_allocation::vulkan_buffer_allocation(std::shared_ptr<vulkan_device> device_owner,
                                                   VkDevice device,
                                                   VkBuffer buffer,
                                                   VkDeviceMemory memory,
                                                   void* mapped_data,
                                                   VkDeviceSize size,
                                                   VkMemoryPropertyFlags memory_flags)
    : device(device),
      buffer(buffer),
      memory(memory),
      mapped_data(mapped_data),
      size(size),
      memory_flags(memory_flags),
      _device_owner(std::move(device_owner)) {}

vulkan_buffer_allocation::~vulkan_buffer_allocation() {
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

void vulkan_buffer_allocation::flush() const {
    OPENVINO_ASSERT(is_host_visible(), "[GPU][Vulkan] Cannot flush non-host-visible memory");
    if ((memory_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(_visibility_mutex);
    VkMappedMemoryRange range{};
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = memory;
    range.offset = 0;
    range.size = VK_WHOLE_SIZE;
    check_vk_result(vkFlushMappedMemoryRanges(device, 1, &range), "vkFlushMappedMemoryRanges");
}

void vulkan_buffer_allocation::invalidate() const {
    OPENVINO_ASSERT(is_host_visible(), "[GPU][Vulkan] Cannot invalidate non-host-visible memory");
    if ((memory_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(_visibility_mutex);
    VkMappedMemoryRange range{};
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = memory;
    range.offset = 0;
    range.size = VK_WHOLE_SIZE;
    check_vk_result(vkInvalidateMappedMemoryRanges(device, 1, &range), "vkInvalidateMappedMemoryRanges");
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
        allocation = allocate_buffer(*_engine, static_cast<size_t>(block_size), vulkan_buffer_memory_usage::device);
    } catch (const std::bad_alloc&) {
        if (block_size == requested_size) {
            throw;
        }
        block_size = requested_size;
        reserved_size = requested_size;
        allocation = allocate_buffer(*_engine, size, vulkan_buffer_memory_usage::device);
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

vulkan_buffer::vulkan_buffer(vulkan_engine* engine,
                             const layout& layout,
                             vulkan_buffer_region::ptr region,
                             VkDeviceSize view_offset,
                             std::shared_ptr<MemoryTracker> memory_tracker)
    : memory(engine, layout, allocation_type::vulkan_buffer, std::move(memory_tracker)),
      _region(std::move(region)),
      _view_offset(view_offset) {
    OPENVINO_ASSERT(_region != nullptr && _view_offset <= _region->get_size() && _bytes_count <= _region->get_size() - _view_offset,
                    "[GPU][Vulkan] Reinterpreted buffer range exceeds the underlying allocation");
}

void vulkan_buffer::validate_range(size_t offset, size_t size, const char* operation) const {
    OPENVINO_ASSERT(offset <= _bytes_count && size <= _bytes_count - offset, "[GPU][Vulkan] ", operation, " range exceeds buffer size");
}

void* vulkan_buffer::mapped_data() const {
    OPENVINO_ASSERT(get_allocation()->is_host_visible() && get_allocation()->mapped_data != nullptr,
                    "[GPU][Vulkan] Buffer memory is not directly host visible");
    return static_cast<unsigned char*>(get_allocation()->mapped_data) + get_offset();
}

vulkan_buffer_allocation::ptr vulkan_buffer::allocate_staging(size_t size) const {
    auto* engine = dynamic_cast<vulkan_engine*>(_engine);
    OPENVINO_ASSERT(engine != nullptr, "[GPU][Vulkan] Staging allocation requires a Vulkan engine");
    return allocate_buffer(*engine, size, vulkan_buffer_memory_usage::host_staging);
}

void* vulkan_buffer::lock(const stream& stream, mem_lock_type type) {
    const auto& vulkan_stream = validate_stream(stream);
    vulkan_stream.finish();
    std::lock_guard<std::mutex> lock(_lock_mutex);
    if (_lock_count == 0) {
        if (get_allocation()->is_host_visible()) {
            if (type != mem_lock_type::write) {
                get_allocation()->invalidate();
            }
        } else {
            if (_lock_staging == nullptr || _lock_staging->size < std::max<size_t>(_bytes_count, 1)) {
                _lock_staging = allocate_staging(_bytes_count);
            }
            if (type != mem_lock_type::write && _bytes_count > 0) {
                vulkan_stream.copy_buffer(get_buffer(), get_offset(), _lock_staging->buffer, 0, _bytes_count);
                _lock_staging->invalidate();
            }
        }
    }
    _write_access = _write_access || type != mem_lock_type::read;
    ++_lock_count;
    return get_allocation()->is_host_visible() ? mapped_data() : _lock_staging->mapped_data;
}

void vulkan_buffer::unlock(const stream& stream) {
    const auto& vulkan_stream = validate_stream(stream);
    std::lock_guard<std::mutex> lock(_lock_mutex);
    OPENVINO_ASSERT(_lock_count > 0, "[GPU][Vulkan] Attempt to unlock a buffer that is not locked");
    --_lock_count;
    if (_lock_count == 0 && _write_access) {
        if (get_allocation()->is_host_visible()) {
            get_allocation()->flush();
        } else if (_bytes_count > 0) {
            OPENVINO_ASSERT(_lock_staging != nullptr, "[GPU][Vulkan] Locked device-local buffer has no staging memory");
            _lock_staging->flush();
            vulkan_stream.copy_buffer(_lock_staging->buffer, 0, get_buffer(), get_offset(), _bytes_count);
        }
        _write_access = false;
    }
}

event::ptr vulkan_buffer::fill(stream& stream, unsigned char pattern, const std::vector<event::ptr>& dep_events, bool) {
    const auto& vulkan_stream = validate_stream(stream);
    stream.wait_for_events(dep_events);
    stream.finish();
    if (_bytes_count > 0) {
        if (get_allocation()->is_host_visible()) {
            std::memset(mapped_data(), pattern, _bytes_count);
            get_allocation()->flush();
        } else {
            auto staging = allocate_staging(_bytes_count);
            std::memset(staging->mapped_data, pattern, _bytes_count);
            staging->flush();
            vulkan_stream.copy_buffer(staging->buffer, 0, get_buffer(), get_offset(), _bytes_count);
        }
    }
    return stream.create_user_event(true);
}

shared_mem_params vulkan_buffer::get_internal_params(runtime_types runtime_type) const {
    OPENVINO_ASSERT(runtime_type == runtime_types::vulkan, "[GPU][Vulkan] Cannot provide internal params for a non-Vulkan runtime");
    return {shared_mem_type::shared_mem_empty,
            nullptr,
            nullptr,
            nullptr,
#ifdef _WIN32
            nullptr,
#else
            0,
#endif
            0};
}

event::ptr vulkan_buffer::copy_from(stream& stream, const void* source, size_t source_offset, size_t destination_offset, size_t size, bool) {
    const auto& vulkan_stream = validate_stream(stream);
    validate_range(destination_offset, size, "copy_from(host)");
    OPENVINO_ASSERT(source != nullptr || size == 0, "[GPU][Vulkan] Source pointer is null");
    stream.finish();
    if (size > 0) {
        const auto* source_bytes = static_cast<const unsigned char*>(source) + source_offset;
        if (get_allocation()->is_host_visible()) {
            auto* destination_bytes = static_cast<unsigned char*>(mapped_data()) + destination_offset;
            std::memcpy(destination_bytes, source_bytes, size);
            get_allocation()->flush();
        } else {
            auto staging = allocate_staging(size);
            std::memcpy(staging->mapped_data, source_bytes, size);
            staging->flush();
            vulkan_stream.copy_buffer(staging->buffer, 0, get_buffer(), get_offset() + destination_offset, size);
        }
    }
    return stream.create_user_event(true);
}

event::ptr vulkan_buffer::copy_from(stream& stream, const memory& source, size_t source_offset, size_t destination_offset, size_t size, bool) {
    const auto& vulkan_stream = validate_stream(stream);
    const auto* source_buffer = dynamic_cast<const vulkan_buffer*>(&source);
    OPENVINO_ASSERT(source_buffer != nullptr && source.get_engine() == _engine, "[GPU][Vulkan] Device copy source is not a Vulkan buffer from this engine");
    source_buffer->validate_range(source_offset, size, "copy_from(device source)");
    validate_range(destination_offset, size, "copy_from(device destination)");
    stream.finish();
    if (size > 0) {
        const auto source_host_visible = source_buffer->get_allocation()->is_host_visible();
        const auto destination_host_visible = get_allocation()->is_host_visible();
        if (source_host_visible && destination_host_visible) {
            source_buffer->get_allocation()->invalidate();
            const auto* source_bytes = static_cast<const unsigned char*>(source_buffer->mapped_data()) + source_offset;
            auto* destination_bytes = static_cast<unsigned char*>(mapped_data()) + destination_offset;
            std::memmove(destination_bytes, source_bytes, size);
            get_allocation()->flush();
        } else {
            if (source_host_visible) {
                source_buffer->get_allocation()->flush();
            }
            const auto absolute_source_offset = source_buffer->get_offset() + source_offset;
            const auto absolute_destination_offset = get_offset() + destination_offset;
            const auto same_buffer = source_buffer->get_buffer() == get_buffer();
            const auto ranges_overlap =
                same_buffer && absolute_source_offset < absolute_destination_offset + size && absolute_destination_offset < absolute_source_offset + size;
            if (same_buffer && absolute_source_offset == absolute_destination_offset) {
                return stream.create_user_event(true);
            }
            if (ranges_overlap) {
                auto staging = allocate_staging(size);
                vulkan_stream.copy_buffer(source_buffer->get_buffer(), absolute_source_offset, staging->buffer, 0, size);
                vulkan_stream.copy_buffer(staging->buffer, 0, get_buffer(), absolute_destination_offset, size);
            } else {
                vulkan_stream.copy_buffer(source_buffer->get_buffer(), absolute_source_offset, get_buffer(), absolute_destination_offset, size);
            }
            if (destination_host_visible) {
                get_allocation()->invalidate();
            }
        }
    }
    return stream.create_user_event(true);
}

event::ptr vulkan_buffer::copy_to(stream& stream, void* destination, size_t source_offset, size_t destination_offset, size_t size, bool) const {
    const auto& vulkan_stream = validate_stream(stream);
    validate_range(source_offset, size, "copy_to(host)");
    OPENVINO_ASSERT(destination != nullptr || size == 0, "[GPU][Vulkan] Destination pointer is null");
    stream.finish();
    if (size > 0) {
        const unsigned char* source_bytes = nullptr;
        vulkan_buffer_allocation::ptr staging;
        if (get_allocation()->is_host_visible()) {
            get_allocation()->invalidate();
            source_bytes = static_cast<const unsigned char*>(mapped_data()) + source_offset;
        } else {
            staging = allocate_staging(size);
            vulkan_stream.copy_buffer(get_buffer(), get_offset() + source_offset, staging->buffer, 0, size);
            staging->invalidate();
            source_bytes = static_cast<const unsigned char*>(staging->mapped_data);
        }
        auto* destination_bytes = static_cast<unsigned char*>(destination) + destination_offset;
        std::memcpy(destination_bytes, source_bytes, size);
    }
    return stream.create_user_event(true);
}

}  // namespace vulkan
}  // namespace cldnn
