// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vk_memory.hpp"

#include "runtime/except.hpp"

#include <cstring>
#include <limits>

namespace ov::core::vulkan {
namespace cross_platform {
namespace {

void* map_memory(VkDevice device, VkDeviceMemory memory) {
    void* ptr = nullptr;
    VK_CALL(vkMapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, &ptr), "vkMapMemory");
    return ptr;
}

void unmap_memory(VkDevice device, VkDeviceMemory memory) {
    vkUnmapMemory(device, memory);
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

void check_boundaries(size_t src_size, size_t src_offset, size_t dst_size, size_t dst_offset, size_t size, const char* label) {
    OPENVINO_ASSERT(src_offset <= src_size && size <= src_size - src_offset,
                    "[GPU] ", label, ": copy range [", src_offset, ", ", src_offset + size,
                    ") exceeds the source buffer size (", src_size, ")");
    OPENVINO_ASSERT(dst_offset <= dst_size && size <= dst_size - dst_offset,
                    "[GPU] ", label, ": copy range [", dst_offset, ", ", dst_offset + size,
                    ") exceeds the destination buffer size (", dst_size, ")");
}

}  // namespace

vk_memory::vk_memory(VkDevice device,
                     VkPhysicalDevice phys_device,
                     VkQueue queue,
                     const layout& layout,
                     allocation_type type)
    : _device(device)
    , _phys_device(phys_device)
    , _queue(queue)
    , _layout(layout)
    , _type(type) {
    _host_visible = (type == allocation_type::usm_host || type == allocation_type::usm_shared);
    allocate();
}

vk_memory::vk_memory(const std::shared_ptr<vk_memory>& base, const layout& new_layout)
    : _device(base->_device)
    , _phys_device(base->_phys_device)
    , _queue(base->_queue)
    , _layout(new_layout)
    , _type(base->_type)
    , _base(base) {
    _host_visible = base->_host_visible;
    _buffer = base->_buffer;
    _memory = base->_memory;
    _allocated_size = base->_allocated_size;
    if (_host_visible)
        _mapped_ptr = base->_mapped_ptr;
}

vk_memory::~vk_memory() {
    if (_base)
        return;  // view — the base owns the allocation
    if (_mapped_ptr != nullptr) {
        vkUnmapMemory(_device, _memory);
        _mapped_ptr = nullptr;
    }
    deallocate();
}

void vk_memory::allocate() {
    _allocated_size = byte_size() == 0 ? 1 : static_cast<VkDeviceSize>(byte_size());
    // vkCmdFillBuffer requires dstOffset and size to be a multiple of 4
    _allocated_size = (_allocated_size + 3) & ~static_cast<VkDeviceSize>(3);

    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = _allocated_size;
    // Kernels bind these buffers as STORAGE_BUFFER descriptors; omitting the
    // usage bit is a spec violation and makes kernel access undefined.
    buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
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
    // Host-visible allocations are kept permanently mapped for their lifetime,
    // so buffer_ptr() and lock() return a valid pointer.
    if (_host_visible)
        _mapped_ptr = map_memory(_device, _memory);
}

void vk_memory::deallocate() {
    if (_base)
        return;  // view — the base owns the allocation
    if (_mapped_ptr != nullptr) {
        vkUnmapMemory(_device, _memory);
        _mapped_ptr = nullptr;
    }
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
    pool_info.queueFamilyIndex = find_compute_queue_family(_phys_device);
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

void* vk_memory::lock() {
    if (_base)
        return _base->lock();
    // TODO: support locking device-local memory via a staging buffer
    if (!_host_visible) {
        OPENVINO_THROW("vk_memory::lock on device-local memory");
    }
    std::lock_guard<std::mutex> locker(_mutex);
    if (0 == _lock_count) {
        // Host-visible allocations are pre-mapped in allocate(); map lazily as a fallback.
        if (_mapped_ptr == nullptr)
            _mapped_ptr = map_memory(_device, _memory);
    }
    _lock_count++;
    return _mapped_ptr;
}

void vk_memory::unlock() {
    if (_base) {
        _base->unlock();
        return;
    }
    std::lock_guard<std::mutex> locker(_mutex);
    OPENVINO_ASSERT(_lock_count != 0, "[GPU] Trying to unlock an already unlocked buffer");
    _lock_count--;
    // The mapping is kept for the lifetime of the allocation; it is released
    // in deallocate()/~vk_memory().
}

void vk_memory::fill(unsigned char pattern) {
    if (byte_size() == 0)
        return;

    if (_host_visible) {
        void* ptr = map_memory(_device, _memory);
        std::memset(ptr, pattern, static_cast<size_t>(_allocated_size));
        unmap_memory(_device, _memory);
    } else {
        one_shot_submit([&](VkCommandBuffer command_buffer) {
            vkCmdFillBuffer(command_buffer, _buffer, 0, _allocated_size, pattern);
        });
    }
}

void vk_memory::copy_from(const void* data_ptr, size_t src_offset, size_t dst_offset, size_t size) {
    if (size == 0)
        return;

    check_boundaries(std::numeric_limits<size_t>::max(), src_offset, byte_size(), dst_offset, size, "vk_memory::copy_from(void*)");

    auto src_ptr = static_cast<const char*>(data_ptr) + src_offset;
    if (_host_visible) {
        void* dst_ptr = map_memory(_device, _memory);
        std::memcpy(static_cast<char*>(dst_ptr) + dst_offset, src_ptr, size);
        unmap_memory(_device, _memory);
    } else {
        auto staging = allocate_staging(_device, _phys_device, size);
        std::memcpy(staging.mapped, src_ptr, size);
        one_shot_submit([&](VkCommandBuffer command_buffer) {
            VkBufferCopy region{0, dst_offset, size};
            vkCmdCopyBuffer(command_buffer, staging.buffer, _buffer, 1, &region);
        });
        free_staging(_device, staging);
    }
}

void vk_memory::copy_from(const vk_memory& src_mem, size_t src_offset, size_t dst_offset, size_t size) {
    if (size == 0)
        return;

    check_boundaries(src_mem.byte_size(), src_offset, byte_size(), dst_offset, size, "vk_memory::copy_from(memory&)");

    if (_host_visible && src_mem._host_visible) {
        void* dst_ptr = map_memory(_device, _memory);
        void* src_ptr = map_memory(src_mem._device, src_mem._memory);
        std::memcpy(static_cast<char*>(dst_ptr) + dst_offset, static_cast<const char*>(src_ptr) + src_offset, size);
        unmap_memory(src_mem._device, src_mem._memory);
        unmap_memory(_device, _memory);
    } else if (src_mem._host_visible) {
        auto staging = allocate_staging(_device, _phys_device, size);
        void* src_ptr = map_memory(src_mem._device, src_mem._memory);
        std::memcpy(staging.mapped, static_cast<const char*>(src_ptr) + src_offset, size);
        unmap_memory(src_mem._device, src_mem._memory);
        one_shot_submit([&](VkCommandBuffer command_buffer) {
            VkBufferCopy region{0, dst_offset, size};
            vkCmdCopyBuffer(command_buffer, staging.buffer, _buffer, 1, &region);
        });
        free_staging(_device, staging);
    } else {
        one_shot_submit([&](VkCommandBuffer command_buffer) {
            VkBufferCopy region{src_offset, dst_offset, size};
            vkCmdCopyBuffer(command_buffer, src_mem._buffer, _buffer, 1, &region);
        });
    }
}

void vk_memory::copy_to(void* data_ptr, size_t src_offset, size_t dst_offset, size_t size) const {
    if (size == 0)
        return;

    check_boundaries(byte_size(), src_offset, std::numeric_limits<size_t>::max(), dst_offset, size, "vk_memory::copy_to(void*)");

    auto dst_ptr = static_cast<char*>(data_ptr) + dst_offset;
    if (_host_visible) {
        void* src_ptr = map_memory(_device, _memory);
        std::memcpy(dst_ptr, static_cast<const char*>(src_ptr) + src_offset, size);
        unmap_memory(_device, _memory);
    } else {
        auto staging = allocate_staging(_device, _phys_device, size);
        one_shot_submit([&](VkCommandBuffer command_buffer) {
            VkBufferCopy region{src_offset, 0, size};
            vkCmdCopyBuffer(command_buffer, _buffer, staging.buffer, 1, &region);
        });
        std::memcpy(dst_ptr, staging.mapped, size);
        free_staging(_device, staging);
    }
}

}  // namespace cross_platform
}  // namespace ov::core::vulkan
