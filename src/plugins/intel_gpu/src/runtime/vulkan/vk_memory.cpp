// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vk_memory.hpp"

#include "runtime_common.hpp"

#include "openvino/core/except.hpp"

#include <cstdint>
#include <cstring>
#include <vector>

namespace cldnn {
namespace vk {
namespace {

void* map_memory(VkDevice device, VkDeviceMemory memory) {
    void* ptr = nullptr;
    VK_CALL(vkMapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, &ptr), "vkMapMemory");
    return ptr;
}

void unmap_memory(VkDevice device, VkDeviceMemory memory) {
    vkUnmapMemory(device, memory);
}

uint32_t find_memory_type(VkPhysicalDevice phys_device, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties mem_properties;
    vkGetPhysicalDeviceMemoryProperties(phys_device, &mem_properties);
    for (uint32_t i = 0; i < mem_properties.memoryTypeCount; ++i) {
        if ((mem_properties.memoryTypes[i].propertyFlags & properties) == properties)
            return i;
    }
    OPENVINO_THROW("[GPU] Failed to find suitable Vulkan memory type");
}

uint32_t find_queue_family(VkPhysicalDevice phys_device) {
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys_device, &count, nullptr);
    std::vector<VkQueueFamilyProperties> properties(count);
    vkGetPhysicalDeviceQueueFamilyProperties(phys_device, &count, properties.data());
    // TODO: accept explicit queue family index from vk_engine instead of the first transfer-capable family
    for (uint32_t i = 0; i < count; ++i) {
        if (properties[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_TRANSFER_BIT))
            return i;
    }
    OPENVINO_THROW("[GPU] Failed to find suitable Vulkan queue family");
}

struct staging_buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
};

staging_buffer allocate_staging(VkDevice device, VkPhysicalDevice phys_device, VkDeviceSize size) {
    staging_buffer staging;

    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CALL(vkCreateBuffer(device, &buffer_info, nullptr, &staging.buffer), "vkCreateBuffer");

    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements(device, staging.buffer, &mem_reqs);

    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_reqs.size;
    alloc_info.memoryTypeIndex = find_memory_type(phys_device, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VK_CALL(vkAllocateMemory(device, &alloc_info, nullptr, &staging.memory), "vkAllocateMemory");
    VK_CALL(vkBindBufferMemory(device, staging.buffer, staging.memory, 0), "vkBindBufferMemory");

    staging.mapped = map_memory(device, staging.memory);
    return staging;
}

void free_staging(VkDevice device, staging_buffer& staging) {
    unmap_memory(device, staging.memory);
    vkDestroyBuffer(device, staging.buffer, nullptr);
    vkFreeMemory(device, staging.memory, nullptr);
}

}  // namespace

vk_memory::vk_memory(engine* engine,
                     const layout& layout,
                     allocation_type type,
                     VkDevice device,
                     VkPhysicalDevice phys_device,
                     VkQueue queue)
    : memory(engine, layout, type, nullptr)
    , _device(device)
    , _phys_device(phys_device)
    , _queue(queue) {
    _host_visible = (type == allocation_type::usm_host || type == allocation_type::usm_shared);
    allocate();
    m_mem_tracker = std::make_shared<MemoryTracker>(engine, static_cast<void*>(_buffer), _bytes_count, type);
}

vk_memory::~vk_memory() {
    if (_mapped_ptr != nullptr) {
        vkUnmapMemory(_device, _memory);
        _mapped_ptr = nullptr;
    }
    deallocate();
}

void vk_memory::allocate() {
    _allocated_size = _bytes_count == 0 ? 1 : static_cast<VkDeviceSize>(_bytes_count);
    // vkCmdFillBuffer requires dstOffset and size to be a multiple of 4
    _allocated_size = (_allocated_size + 3) & ~static_cast<VkDeviceSize>(3);

    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = _allocated_size;
    buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CALL(vkCreateBuffer(_device, &buffer_info, nullptr, &_buffer), "vkCreateBuffer");

    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements(_device, _buffer, &mem_reqs);

    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_reqs.size;
    alloc_info.memoryTypeIndex = find_memory_type(_phys_device,
                                                  _host_visible ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                                                                : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CALL(vkAllocateMemory(_device, &alloc_info, nullptr, &_memory), "vkAllocateMemory");
    VK_CALL(vkBindBufferMemory(_device, _buffer, _memory, 0), "vkBindBufferMemory");
}

void vk_memory::deallocate() {
    if (_cmd_pool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(_device, _cmd_pool, nullptr);
        _cmd_pool = VK_NULL_HANDLE;
    }
    if (_memory != VK_NULL_HANDLE) {
        vkFreeMemory(_device, _memory, nullptr);
        _memory = VK_NULL_HANDLE;
    }
    if (_buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(_device, _buffer, nullptr);
        _buffer = VK_NULL_HANDLE;
    }
}

void vk_memory::ensure_command_pool() const {
    if (_cmd_pool != VK_NULL_HANDLE)
        return;

    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    pool_info.queueFamilyIndex = find_queue_family(_phys_device);
    VK_CALL(vkCreateCommandPool(_device, &pool_info, nullptr, &_cmd_pool), "vkCreateCommandPool");
}

void vk_memory::one_shot_submit(const std::function<void(VkCommandBuffer)>& record) const {
    std::lock_guard<std::mutex> locker(_transfer_mutex);
    ensure_command_pool();

    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = _cmd_pool;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;

    VkCommandBuffer command_buffer;
    VK_CALL(vkAllocateCommandBuffers(_device, &alloc_info, &command_buffer), "vkAllocateCommandBuffers");

    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence;
    VK_CALL(vkCreateFence(_device, &fence_info, nullptr, &fence), "vkCreateFence");

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CALL(vkBeginCommandBuffer(command_buffer, &begin_info), "vkBeginCommandBuffer");
    record(command_buffer);
    VK_CALL(vkEndCommandBuffer(command_buffer), "vkEndCommandBuffer");

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;
    VK_CALL(vkQueueSubmit(_queue, 1, &submit_info, fence), "vkQueueSubmit");
    VK_CALL(vkWaitForFences(_device, 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");

    vkDestroyFence(_device, fence, nullptr);
    vkFreeCommandBuffers(_device, _cmd_pool, 1, &command_buffer);
}

void* vk_memory::lock(const stream&, mem_lock_type) {
    // TODO: support locking device-local memory via a staging buffer
    if (!_host_visible) {
        OPENVINO_THROW("vk_memory::lock on device-local memory");
    }
    std::lock_guard<std::mutex> locker(_mutex);
    if (0 == _lock_count) {
        _mapped_ptr = map_memory(_device, _memory);
    }
    _lock_count++;
    return _mapped_ptr;
}

void vk_memory::unlock(const stream&) {
    std::lock_guard<std::mutex> locker(_mutex);
    OPENVINO_ASSERT(_lock_count != 0, "[GPU] Trying to unlock an already unlocked buffer");
    _lock_count--;
    if (0 == _lock_count) {
        vkUnmapMemory(_device, _memory);
        _mapped_ptr = nullptr;
    }
}

event::ptr vk_memory::fill(stream&, unsigned char pattern, const std::vector<event::ptr>&, bool) {
    if (_bytes_count == 0)
        return nullptr;

    // TODO: support non-blocking fill by returning a Vulkan event
    if (_host_visible) {
        void* ptr = map_memory(_device, _memory);
        memset(ptr, pattern, static_cast<size_t>(_allocated_size));
        unmap_memory(_device, _memory);
    } else {
        one_shot_submit([&](VkCommandBuffer command_buffer) {
            vkCmdFillBuffer(command_buffer, _buffer, 0, _allocated_size, pattern);
        });
    }
    return nullptr;
}

event::ptr vk_memory::copy_from(stream&, const void* data_ptr, size_t src_offset, size_t dst_offset, size_t size, bool) {
    if (size == 0)
        return nullptr;

    check_boundaries(SIZE_MAX, src_offset, _bytes_count, dst_offset, size, "vk_memory::copy_from(void*)");

    // TODO: support non-blocking copies by returning a Vulkan event
    auto src_ptr = static_cast<const char*>(data_ptr) + src_offset;
    if (_host_visible) {
        void* dst_ptr = map_memory(_device, _memory);
        memcpy(static_cast<char*>(dst_ptr) + dst_offset, src_ptr, size);
        unmap_memory(_device, _memory);
    } else {
        auto staging = allocate_staging(_device, _phys_device, size);
        memcpy(staging.mapped, src_ptr, size);
        one_shot_submit([&](VkCommandBuffer command_buffer) {
            VkBufferCopy region{0, dst_offset, size};
            vkCmdCopyBuffer(command_buffer, staging.buffer, _buffer, 1, &region);
        });
        free_staging(_device, staging);
    }
    return nullptr;
}

event::ptr vk_memory::copy_from(stream& stream, const memory& src_mem, size_t src_offset, size_t dst_offset, size_t size, bool blocking) {
    if (size == 0)
        return nullptr;

    check_boundaries(src_mem.size(), src_offset, _bytes_count, dst_offset, size, "vk_memory::copy_from(memory&)");

    auto vk_src = dynamic_cast<const vk_memory*>(&src_mem);
    if (vk_src == nullptr) {
        // TODO: use direct copy when both memories belong to the Vulkan backend
        std::vector<char> tmp_buf(size);
        src_mem.copy_to(stream, tmp_buf.data(), src_offset, 0, size, true);
        return copy_from(stream, tmp_buf.data(), 0, dst_offset, size, true);
    }

    if (_host_visible && vk_src->_host_visible) {
        void* dst_ptr = map_memory(_device, _memory);
        void* src_ptr = map_memory(vk_src->_device, vk_src->_memory);
        memcpy(static_cast<char*>(dst_ptr) + dst_offset, static_cast<const char*>(src_ptr) + src_offset, size);
        unmap_memory(vk_src->_device, vk_src->_memory);
        unmap_memory(_device, _memory);
    } else if (vk_src->_host_visible) {
        auto staging = allocate_staging(_device, _phys_device, size);
        void* src_ptr = map_memory(vk_src->_device, vk_src->_memory);
        memcpy(staging.mapped, static_cast<const char*>(src_ptr) + src_offset, size);
        unmap_memory(vk_src->_device, vk_src->_memory);
        one_shot_submit([&](VkCommandBuffer command_buffer) {
            VkBufferCopy region{0, dst_offset, size};
            vkCmdCopyBuffer(command_buffer, staging.buffer, _buffer, 1, &region);
        });
        free_staging(_device, staging);
    } else {
        one_shot_submit([&](VkCommandBuffer command_buffer) {
            VkBufferCopy region{src_offset, dst_offset, size};
            vkCmdCopyBuffer(command_buffer, vk_src->_buffer, _buffer, 1, &region);
        });
    }
    return nullptr;
}

event::ptr vk_memory::copy_to(stream&, void* data_ptr, size_t src_offset, size_t dst_offset, size_t size, bool) const {
    if (size == 0)
        return nullptr;

    check_boundaries(_bytes_count, src_offset, SIZE_MAX, dst_offset, size, "vk_memory::copy_to(void*)");

    // TODO: support non-blocking copies by returning a Vulkan event
    auto dst_ptr = static_cast<char*>(data_ptr) + dst_offset;
    if (_host_visible) {
        void* src_ptr = map_memory(_device, _memory);
        memcpy(dst_ptr, static_cast<const char*>(src_ptr) + src_offset, size);
        unmap_memory(_device, _memory);
    } else {
        auto staging = allocate_staging(_device, _phys_device, size);
        one_shot_submit([&](VkCommandBuffer command_buffer) {
            VkBufferCopy region{src_offset, 0, size};
            vkCmdCopyBuffer(command_buffer, _buffer, staging.buffer, 1, &region);
        });
        memcpy(dst_ptr, staging.mapped, size);
        free_staging(_device, staging);
    }
    return nullptr;
}

shared_mem_params vk_memory::get_internal_params(runtime_types) const {
    return {shared_mem_type::shared_mem_usm,
            nullptr,
            nullptr,
            reinterpret_cast<shared_handle>(_buffer),
#ifdef _WIN32
            nullptr,
#else
            0,
#endif
            0};
}

}  // namespace vk
}  // namespace cldnn
