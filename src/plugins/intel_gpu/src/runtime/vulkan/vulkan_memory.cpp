// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vulkan_memory.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
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

uint32_t select_memory_type(VkPhysicalDevice physical_device, uint32_t allowed_types) {
    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties(physical_device, &properties);

    std::optional<uint32_t> selected;
    int selected_score = -1;
    for (uint32_t index = 0; index < properties.memoryTypeCount; ++index) {
        if ((allowed_types & (1U << index)) == 0) {
            continue;
        }

        const auto flags = properties.memoryTypes[index].propertyFlags;
        if ((flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0) {
            continue;
        }

        const int score = ((flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0 ? 2 : 0) + ((flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0 ? 1 : 0);
        if (score > selected_score) {
            selected = index;
            selected_score = score;
        }
    }

    OPENVINO_ASSERT(selected.has_value(), "[GPU][Vulkan] No host-visible memory type is available for a storage buffer");
    return *selected;
}

vulkan_buffer_allocation::ptr allocate_buffer(vulkan_engine& engine, size_t requested_size) {
    const VkDeviceSize buffer_size = std::max<VkDeviceSize>(requested_size, 1);

    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = buffer_size;
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer = VK_NULL_HANDLE;
    check_vk_result(vkCreateBuffer(engine.get_device_handle(), &buffer_info, nullptr, &buffer), "vkCreateBuffer");

    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(engine.get_device_handle(), buffer, &requirements);
    const auto memory_type = select_memory_type(engine.get_physical_device(), requirements.memoryTypeBits);

    VkPhysicalDeviceMemoryProperties memory_properties{};
    vkGetPhysicalDeviceMemoryProperties(engine.get_physical_device(), &memory_properties);
    const auto memory_flags = memory_properties.memoryTypes[memory_type].propertyFlags;

    VkMemoryAllocateInfo allocation_info{};
    allocation_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocation_info.allocationSize = requirements.size;
    allocation_info.memoryTypeIndex = memory_type;

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
    const auto map_result = vkMapMemory(engine.get_device_handle(), device_memory, 0, VK_WHOLE_SIZE, 0, &mapped_data);
    if (map_result != VK_SUCCESS) {
        vkDestroyBuffer(engine.get_device_handle(), buffer, nullptr);
        vkFreeMemory(engine.get_device_handle(), device_memory, nullptr);
        check_vk_result(map_result, "vkMapMemory");
    }

    return std::make_shared<vulkan_buffer_allocation>(engine.get_vulkan_device_object(),
                                                      engine.get_device_handle(),
                                                      buffer,
                                                      device_memory,
                                                      mapped_data,
                                                      requirements.size,
                                                      (memory_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0);
}

const vulkan_stream& validate_stream(const stream& stream) {
    const auto* result = dynamic_cast<const vulkan_stream*>(&stream);
    OPENVINO_ASSERT(result != nullptr, "[GPU][Vulkan] Memory operation received a stream from another backend");
    return *result;
}

}  // namespace

vulkan_buffer_allocation::vulkan_buffer_allocation(std::shared_ptr<vulkan_device> device_owner,
                                                   VkDevice device,
                                                   VkBuffer buffer,
                                                   VkDeviceMemory memory,
                                                   void* mapped_data,
                                                   VkDeviceSize size,
                                                   bool host_coherent)
    : device(device),
      buffer(buffer),
      memory(memory),
      mapped_data(mapped_data),
      size(size),
      host_coherent(host_coherent),
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
    if (host_coherent) {
        return;
    }
    VkMappedMemoryRange range{};
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = memory;
    range.offset = 0;
    range.size = VK_WHOLE_SIZE;
    check_vk_result(vkFlushMappedMemoryRanges(device, 1, &range), "vkFlushMappedMemoryRanges");
}

void vulkan_buffer_allocation::invalidate() const {
    if (host_coherent) {
        return;
    }
    VkMappedMemoryRange range{};
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = memory;
    range.offset = 0;
    range.size = VK_WHOLE_SIZE;
    check_vk_result(vkInvalidateMappedMemoryRanges(device, 1, &range), "vkInvalidateMappedMemoryRanges");
}

vulkan_buffer::vulkan_buffer(vulkan_engine* engine, const layout& layout)
    : memory(engine, layout, allocation_type::vulkan_buffer, nullptr),
      _allocation(allocate_buffer(*engine, _bytes_count)) {
    m_mem_tracker = std::make_shared<MemoryTracker>(engine, _allocation->mapped_data, _bytes_count, allocation_type::vulkan_buffer);
}

vulkan_buffer::vulkan_buffer(vulkan_engine* engine,
                             const layout& layout,
                             vulkan_buffer_allocation::ptr allocation,
                             VkDeviceSize offset,
                             std::shared_ptr<MemoryTracker> memory_tracker)
    : memory(engine, layout, allocation_type::vulkan_buffer, std::move(memory_tracker)),
      _allocation(std::move(allocation)),
      _offset(offset) {
    OPENVINO_ASSERT(_allocation != nullptr && _offset <= _allocation->size && _bytes_count <= _allocation->size - _offset,
                    "[GPU][Vulkan] Reinterpreted buffer range exceeds the underlying allocation");
}

void vulkan_buffer::validate_range(size_t offset, size_t size, const char* operation) const {
    OPENVINO_ASSERT(offset <= _bytes_count && size <= _bytes_count - offset, "[GPU][Vulkan] ", operation, " range exceeds buffer size");
}

void* vulkan_buffer::mapped_data() const {
    return static_cast<unsigned char*>(_allocation->mapped_data) + _offset;
}

void* vulkan_buffer::lock(const stream& stream, mem_lock_type type) {
    validate_stream(stream).finish();
    std::lock_guard<std::mutex> lock(_lock_mutex);
    if (_lock_count == 0 && type != mem_lock_type::write) {
        _allocation->invalidate();
    }
    _write_access = _write_access || type != mem_lock_type::read;
    ++_lock_count;
    return mapped_data();
}

void vulkan_buffer::unlock(const stream& stream) {
    validate_stream(stream);
    std::lock_guard<std::mutex> lock(_lock_mutex);
    OPENVINO_ASSERT(_lock_count > 0, "[GPU][Vulkan] Attempt to unlock a buffer that is not locked");
    --_lock_count;
    if (_lock_count == 0 && _write_access) {
        _allocation->flush();
        _write_access = false;
    }
}

event::ptr vulkan_buffer::fill(stream& stream, unsigned char pattern, const std::vector<event::ptr>& dep_events, bool) {
    validate_stream(stream);
    stream.wait_for_events(dep_events);
    stream.finish();
    if (_bytes_count > 0) {
        std::memset(mapped_data(), pattern, _bytes_count);
        _allocation->flush();
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
    validate_stream(stream);
    validate_range(destination_offset, size, "copy_from(host)");
    OPENVINO_ASSERT(source != nullptr || size == 0, "[GPU][Vulkan] Source pointer is null");
    stream.finish();
    if (size > 0) {
        const auto* source_bytes = static_cast<const unsigned char*>(source) + source_offset;
        auto* destination_bytes = static_cast<unsigned char*>(mapped_data()) + destination_offset;
        std::memcpy(destination_bytes, source_bytes, size);
        _allocation->flush();
    }
    return stream.create_user_event(true);
}

event::ptr vulkan_buffer::copy_from(stream& stream, const memory& source, size_t source_offset, size_t destination_offset, size_t size, bool) {
    validate_stream(stream);
    const auto* source_buffer = dynamic_cast<const vulkan_buffer*>(&source);
    OPENVINO_ASSERT(source_buffer != nullptr, "[GPU][Vulkan] Device copy source is not a Vulkan buffer");
    source_buffer->validate_range(source_offset, size, "copy_from(device source)");
    validate_range(destination_offset, size, "copy_from(device destination)");
    stream.finish();
    if (size > 0) {
        source_buffer->_allocation->invalidate();
        const auto* source_bytes = static_cast<const unsigned char*>(source_buffer->mapped_data()) + source_offset;
        auto* destination_bytes = static_cast<unsigned char*>(mapped_data()) + destination_offset;
        std::memmove(destination_bytes, source_bytes, size);
        _allocation->flush();
    }
    return stream.create_user_event(true);
}

event::ptr vulkan_buffer::copy_to(stream& stream, void* destination, size_t source_offset, size_t destination_offset, size_t size, bool) const {
    validate_stream(stream);
    validate_range(source_offset, size, "copy_to(host)");
    OPENVINO_ASSERT(destination != nullptr || size == 0, "[GPU][Vulkan] Destination pointer is null");
    stream.finish();
    if (size > 0) {
        _allocation->invalidate();
        const auto* source_bytes = static_cast<const unsigned char*>(mapped_data()) + source_offset;
        auto* destination_bytes = static_cast<unsigned char*>(destination) + destination_offset;
        std::memcpy(destination_bytes, source_bytes, size);
    }
    return stream.create_user_event(true);
}

}  // namespace vulkan
}  // namespace cldnn
