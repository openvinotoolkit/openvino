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

#if defined(__linux__)
#    include <fcntl.h>
#    include <sys/stat.h>
#    include <unistd.h>
#endif
#if defined(__APPLE__)
#    include <vulkan/vulkan_metal.h>
#endif
#if defined(__ANDROID__)
#    include <vulkan/vulkan_android.h>
#endif

#include "intel_gpu/runtime/stream.hpp"
#include "openvino/core/except.hpp"
#include "vulkan_engine.hpp"
#include "vulkan_stream.hpp"

namespace cldnn {
namespace vulkan {
namespace {

constexpr VkDeviceSize buffer_transfer_alignment = sizeof(uint32_t);

bool is_transfer_aligned(VkDeviceSize offset, VkDeviceSize size) {
    return offset % buffer_transfer_alignment == 0 && size % buffer_transfer_alignment == 0;
}

uint32_t make_fill_pattern(unsigned char pattern) {
    uint32_t result = 0;
    std::memset(&result, pattern, sizeof(result));
    return result;
}

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

bool transition_foreign_buffer_ownership(const std::shared_ptr<vulkan_device>& device_owner, VkBuffer buffer, bool acquire) noexcept {
    if (device_owner == nullptr || buffer == VK_NULL_HANDLE) {
        return false;
    }

    const auto device = device_owner->get_device();
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

struct vulkan_external_buffer_import {
    VkExternalMemoryHandleTypeFlagBits handle_type{};
    const void* allocation_pnext = nullptr;
    uint32_t memory_type_bits = 0;
    VkDeviceSize allocation_size = 0;
    bool map_memory = false;
    bool foreign_owned = false;
    int imported_fd = -1;
};

void close_imported_fd(vulkan_external_buffer_import& import) noexcept {
#if defined(__linux__)
    if (import.imported_fd >= 0) {
        close(import.imported_fd);
        import.imported_fd = -1;
    }
#else
    static_cast<void>(import);
#endif
}

vulkan_buffer_allocation::ptr import_external_buffer(vulkan_engine& engine,
                                                     size_t requested_size,
                                                     vulkan_buffer_memory_usage usage,
                                                     vulkan_external_buffer_import import) {
    const auto device_owner = engine.get_vulkan_device_object();
    OPENVINO_ASSERT(device_owner->get_external_memory_capabilities().supports(import.handle_type),
                    "[GPU][Vulkan] Requested external buffer handle type is not importable");
    const VkDeviceSize buffer_size = std::max<VkDeviceSize>(requested_size, 1);

    VkExternalMemoryBufferCreateInfo external_info{};
    external_info.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    external_info.handleTypes = import.handle_type;

    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.pNext = &external_info;
    buffer_info.size = buffer_size;
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer = VK_NULL_HANDLE;
    const auto create_result = vkCreateBuffer(engine.get_device_handle(), &buffer_info, nullptr, &buffer);
    if (create_result != VK_SUCCESS) {
        close_imported_fd(import);
        check_vk_result(create_result, "vkCreateBuffer(external)");
    }

    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(engine.get_device_handle(), buffer, &requirements);
    const auto allowed_types = requirements.memoryTypeBits & import.memory_type_bits;
    vulkan_memory_type_selection memory_type{};
    try {
        VkPhysicalDeviceMemoryProperties memory_properties{};
        vkGetPhysicalDeviceMemoryProperties(engine.get_physical_device(), &memory_properties);
        const bool prefer_unified_memory = engine.get_device_info().dev_type != device_type::discrete_gpu;
        memory_type = select_vulkan_memory_type(memory_properties, allowed_types, usage, prefer_unified_memory);
    } catch (...) {
        vkDestroyBuffer(engine.get_device_handle(), buffer, nullptr);
        close_imported_fd(import);
        throw;
    }

    const auto allocation_size = import.allocation_size == 0 ? requirements.size : import.allocation_size;
    if (allocation_size < requirements.size) {
        vkDestroyBuffer(engine.get_device_handle(), buffer, nullptr);
        close_imported_fd(import);
        OPENVINO_THROW("[GPU][Vulkan] External allocation is smaller than Vulkan buffer requirements: ", allocation_size, " < ", requirements.size);
    }

    VkMemoryDedicatedAllocateInfo dedicated_info{};
    dedicated_info.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
    dedicated_info.pNext = import.allocation_pnext;
    dedicated_info.buffer = buffer;

    VkMemoryAllocateInfo allocation_info{};
    allocation_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocation_info.pNext = &dedicated_info;
    allocation_info.allocationSize = allocation_size;
    allocation_info.memoryTypeIndex = memory_type.index;

    VkDeviceMemory device_memory = VK_NULL_HANDLE;
    const auto allocation_result = vkAllocateMemory(engine.get_device_handle(), &allocation_info, nullptr, &device_memory);
    if (allocation_result != VK_SUCCESS) {
        vkDestroyBuffer(engine.get_device_handle(), buffer, nullptr);
        close_imported_fd(import);
        check_vk_result(allocation_result, "vkAllocateMemory(external)");
    }
    import.imported_fd = -1;

    const auto bind_result = vkBindBufferMemory(engine.get_device_handle(), buffer, device_memory, 0);
    if (bind_result != VK_SUCCESS) {
        vkDestroyBuffer(engine.get_device_handle(), buffer, nullptr);
        vkFreeMemory(engine.get_device_handle(), device_memory, nullptr);
        check_vk_result(bind_result, "vkBindBufferMemory(external)");
    }

    void* mapped_data = nullptr;
    if (import.map_memory) {
        const auto map_result = vkMapMemory(engine.get_device_handle(), device_memory, 0, allocation_size, 0, &mapped_data);
        if (map_result != VK_SUCCESS) {
            vkDestroyBuffer(engine.get_device_handle(), buffer, nullptr);
            vkFreeMemory(engine.get_device_handle(), device_memory, nullptr);
            check_vk_result(map_result, "vkMapMemory(external)");
        }
    }

    if (import.foreign_owned && !transition_foreign_buffer_ownership(device_owner, buffer, true)) {
        if (mapped_data != nullptr) {
            vkUnmapMemory(engine.get_device_handle(), device_memory);
        }
        vkDestroyBuffer(engine.get_device_handle(), buffer, nullptr);
        vkFreeMemory(engine.get_device_handle(), device_memory, nullptr);
        OPENVINO_THROW("[GPU][Vulkan] Failed to acquire external buffer ownership from a foreign queue family");
    }

    return std::make_shared<vulkan_buffer_allocation>(device_owner,
                                                      engine.get_device_handle(),
                                                      buffer,
                                                      device_memory,
                                                      mapped_data,
                                                      buffer_size,
                                                      allocation_size,
                                                      memory_type.flags,
                                                      engine.get_non_coherent_atom_size(),
                                                      import.foreign_owned);
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
                                                      allocation_info.allocationSize,
                                                      memory_type.flags,
                                                      engine.get_non_coherent_atom_size());
}

const vulkan_stream& validate_stream(const stream& stream) {
    const auto* result = dynamic_cast<const vulkan_stream*>(&stream);
    OPENVINO_ASSERT(result != nullptr, "[GPU][Vulkan] Memory operation received a stream from another backend");
    return *result;
}

}  // namespace

vulkan_buffer_region::ptr import_vulkan_host_buffer(vulkan_engine& engine, void* address, size_t data_size, size_t buffer_size) {
    const auto& capabilities = engine.get_vulkan_device_object()->get_external_memory_capabilities();
    OPENVINO_ASSERT(capabilities.supports(VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT),
                    "[GPU][Vulkan] VK_EXT_external_memory_host is unavailable for this device");
    OPENVINO_ASSERT(address != nullptr, "[GPU][Vulkan] Imported host pointer must not be null");
    OPENVINO_ASSERT(data_size >= buffer_size, "[GPU][Vulkan] Imported host allocation is smaller than the tensor: ", data_size, " < ", buffer_size);
    const auto alignment = capabilities.min_imported_host_pointer_alignment;
    OPENVINO_ASSERT(reinterpret_cast<std::uintptr_t>(address) % alignment == 0, "[GPU][Vulkan] Imported host pointer must be ", alignment, "-byte aligned");
    OPENVINO_ASSERT(data_size > 0 && data_size % alignment == 0,
                    "[GPU][Vulkan] Imported host allocation size must be a positive multiple of ",
                    alignment,
                    " bytes");

    const auto get_properties =
        reinterpret_cast<PFN_vkGetMemoryHostPointerPropertiesEXT>(vkGetDeviceProcAddr(engine.get_device_handle(), "vkGetMemoryHostPointerPropertiesEXT"));
    OPENVINO_ASSERT(get_properties != nullptr, "[GPU][Vulkan] vkGetMemoryHostPointerPropertiesEXT is unavailable");
    VkMemoryHostPointerPropertiesEXT properties{};
    properties.sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT;
    check_vk_result(get_properties(engine.get_device_handle(), VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, address, &properties),
                    "vkGetMemoryHostPointerPropertiesEXT");

    VkImportMemoryHostPointerInfoEXT host_import{};
    host_import.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT;
    host_import.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    host_import.pHostPointer = address;
    auto allocation =
        import_external_buffer(engine,
                               buffer_size,
                               vulkan_buffer_memory_usage::host_staging,
                               {VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, &host_import, properties.memoryTypeBits, data_size, true, false, -1});
    OPENVINO_ASSERT(allocation->is_host_coherent(),
                    "[GPU][Vulkan] Imported host pointers require a host-coherent memory type because the public "
                    "CPU_VA API has no explicit Vulkan flush operation");
    return vulkan_buffer_region::create_external(std::move(allocation), buffer_size);
}

vulkan_buffer_region::ptr import_vulkan_os_buffer(vulkan_engine& engine, intptr_t external_handle, size_t buffer_size) {
#if defined(__linux__)
    OPENVINO_ASSERT(external_handle >= 0 && external_handle <= (std::numeric_limits<int>::max)(),
                    "[GPU][Vulkan] DMA-BUF handle must be a valid file descriptor");
    const auto& capabilities = engine.get_vulkan_device_object()->get_external_memory_capabilities();
    OPENVINO_ASSERT(capabilities.supports(VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT), "[GPU][Vulkan] DMA-BUF import is unavailable for this device");
    const auto source_fd = static_cast<int>(external_handle);
    struct stat source_status{};
    OPENVINO_ASSERT(fstat(source_fd, &source_status) == 0 && source_status.st_size > 0, "[GPU][Vulkan] Failed to query DMA-BUF allocation size");
    const auto allocation_size = static_cast<VkDeviceSize>(source_status.st_size);
    OPENVINO_ASSERT(allocation_size >= buffer_size, "[GPU][Vulkan] DMA-BUF allocation is smaller than the tensor: ", allocation_size, " < ", buffer_size);

    const auto get_properties = reinterpret_cast<PFN_vkGetMemoryFdPropertiesKHR>(vkGetDeviceProcAddr(engine.get_device_handle(), "vkGetMemoryFdPropertiesKHR"));
    OPENVINO_ASSERT(get_properties != nullptr, "[GPU][Vulkan] vkGetMemoryFdPropertiesKHR is unavailable");
    VkMemoryFdPropertiesKHR properties{};
    properties.sType = VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR;
    check_vk_result(get_properties(engine.get_device_handle(), VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, source_fd, &properties),
                    "vkGetMemoryFdPropertiesKHR");

    const auto imported_fd = fcntl(source_fd, F_DUPFD_CLOEXEC, 0);
    OPENVINO_ASSERT(imported_fd >= 0, "[GPU][Vulkan] Failed to duplicate DMA-BUF file descriptor before ownership transfer");
    VkImportMemoryFdInfoKHR fd_import{};
    fd_import.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR;
    fd_import.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    fd_import.fd = imported_fd;
    auto allocation = import_external_buffer(
        engine,
        buffer_size,
        vulkan_buffer_memory_usage::device,
        {VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, &fd_import, properties.memoryTypeBits, allocation_size, false, true, imported_fd});
    return vulkan_buffer_region::create_external(std::move(allocation), buffer_size);
#else
    static_cast<void>(engine);
    static_cast<void>(external_handle);
    static_cast<void>(buffer_size);
    OPENVINO_THROW("[GPU][Vulkan] OS-handle buffer import is only implemented for Linux DMA-BUF file descriptors");
#endif
}

vulkan_buffer_region::ptr import_vulkan_native_buffer(vulkan_engine& engine, void* external_handle, size_t buffer_size) {
    OPENVINO_ASSERT(external_handle != nullptr, "[GPU][Vulkan] Native external buffer handle must not be null");
#if defined(__ANDROID__)
    const auto& capabilities = engine.get_vulkan_device_object()->get_external_memory_capabilities();
    OPENVINO_ASSERT(capabilities.supports(VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID),
                    "[GPU][Vulkan] AHardwareBuffer import is unavailable for this device");
    const auto get_properties = reinterpret_cast<PFN_vkGetAndroidHardwareBufferPropertiesANDROID>(
        vkGetDeviceProcAddr(engine.get_device_handle(), "vkGetAndroidHardwareBufferPropertiesANDROID"));
    OPENVINO_ASSERT(get_properties != nullptr, "[GPU][Vulkan] vkGetAndroidHardwareBufferPropertiesANDROID is unavailable");
    auto* hardware_buffer = static_cast<AHardwareBuffer*>(external_handle);
    VkAndroidHardwareBufferPropertiesANDROID properties{};
    properties.sType = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID;
    check_vk_result(get_properties(engine.get_device_handle(), hardware_buffer, &properties), "vkGetAndroidHardwareBufferPropertiesANDROID");

    VkImportAndroidHardwareBufferInfoANDROID ahb_import{};
    ahb_import.sType = VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID;
    ahb_import.buffer = hardware_buffer;
    auto allocation = import_external_buffer(engine,
                                             buffer_size,
                                             vulkan_buffer_memory_usage::device,
                                             {VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID,
                                              &ahb_import,
                                              properties.memoryTypeBits,
                                              properties.allocationSize,
                                              false,
                                              true,
                                              -1});
    return vulkan_buffer_region::create_external(std::move(allocation), buffer_size);
#elif defined(__APPLE__)
    const auto& capabilities = engine.get_vulkan_device_object()->get_external_memory_capabilities();
    OPENVINO_ASSERT(capabilities.supports(VK_EXTERNAL_MEMORY_HANDLE_TYPE_MTLBUFFER_BIT_EXT), "[GPU][Vulkan] MTLBuffer import is unavailable for this device");
    const auto get_properties =
        reinterpret_cast<PFN_vkGetMemoryMetalHandlePropertiesEXT>(vkGetDeviceProcAddr(engine.get_device_handle(), "vkGetMemoryMetalHandlePropertiesEXT"));
    OPENVINO_ASSERT(get_properties != nullptr, "[GPU][Vulkan] vkGetMemoryMetalHandlePropertiesEXT is unavailable");
    VkMemoryMetalHandlePropertiesEXT properties{};
    properties.sType = VK_STRUCTURE_TYPE_MEMORY_METAL_HANDLE_PROPERTIES_EXT;
    check_vk_result(get_properties(engine.get_device_handle(), VK_EXTERNAL_MEMORY_HANDLE_TYPE_MTLBUFFER_BIT_EXT, external_handle, &properties),
                    "vkGetMemoryMetalHandlePropertiesEXT");

    VkImportMemoryMetalHandleInfoEXT metal_import{};
    metal_import.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_METAL_HANDLE_INFO_EXT;
    metal_import.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_MTLBUFFER_BIT_EXT;
    metal_import.handle = external_handle;
    auto allocation = import_external_buffer(engine,
                                             buffer_size,
                                             vulkan_buffer_memory_usage::device,
                                             {VK_EXTERNAL_MEMORY_HANDLE_TYPE_MTLBUFFER_BIT_EXT, &metal_import, properties.memoryTypeBits, 0, false, false, -1});
    return vulkan_buffer_region::create_external(std::move(allocation), buffer_size);
#else
    static_cast<void>(engine);
    static_cast<void>(buffer_size);
    OPENVINO_THROW("[GPU][Vulkan] Native external buffers are supported only on Android and macOS");
#endif
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
        transition_foreign_buffer_ownership(_device_owner, buffer, false);
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
    check_vk_result(vkFlushMappedMemoryRanges(device, 1, &range), "vkFlushMappedMemoryRanges");
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
            if (type != mem_lock_type::write || !get_allocation()->is_host_coherent()) {
                get_allocation()->invalidate(get_offset(), _bytes_count);
            }
        } else {
            const auto region_size = _region->get_size();
            if (_lock_staging == nullptr || _lock_staging->size < region_size) {
                _lock_staging = allocate_staging(static_cast<size_t>(region_size));
            }
            const auto covers_region = _view_offset == 0 && _bytes_count == region_size;
            if ((!covers_region || type != mem_lock_type::write) && region_size > 0) {
                vulkan_stream.enqueue_buffer_copy(get_allocation(), _region->get_offset(), _lock_staging, 0, region_size, true, {_region});
                _lock_staging->invalidate(0, region_size);
            }
        }
    }
    _write_access = _write_access || type != mem_lock_type::read;
    ++_lock_count;
    return get_allocation()->is_host_visible() ? mapped_data() : static_cast<unsigned char*>(_lock_staging->mapped_data) + _view_offset;
}

void vulkan_buffer::unlock(const stream& stream) {
    const auto& vulkan_stream = validate_stream(stream);
    std::lock_guard<std::mutex> lock(_lock_mutex);
    OPENVINO_ASSERT(_lock_count > 0, "[GPU][Vulkan] Attempt to unlock a buffer that is not locked");
    --_lock_count;
    if (_lock_count == 0 && _write_access) {
        if (get_allocation()->is_host_visible()) {
            get_allocation()->flush(get_offset(), _bytes_count);
        } else if (_bytes_count > 0) {
            OPENVINO_ASSERT(_lock_staging != nullptr, "[GPU][Vulkan] Locked device-local buffer has no staging memory");
            _lock_staging->flush(0, _region->get_size());
            vulkan_stream.enqueue_buffer_copy(_lock_staging, 0, get_allocation(), _region->get_offset(), _region->get_size(), true, {_region});
        }
        _write_access = false;
    }
}

event::ptr vulkan_buffer::fill(stream& stream, unsigned char pattern, const std::vector<event::ptr>& dep_events, bool blocking) {
    const auto& vulkan_stream = validate_stream(stream);
    if (_bytes_count == 0) {
        stream.wait_for_events(dep_events);
        return stream.create_user_event(true);
    }
    if (get_allocation()->is_host_visible()) {
        stream.wait_for_events(dep_events);
        stream.finish();
        if (!get_allocation()->is_host_coherent()) {
            get_allocation()->invalidate(get_offset(), _bytes_count);
        }
        std::memset(mapped_data(), pattern, _bytes_count);
        get_allocation()->flush(get_offset(), _bytes_count);
        return stream.create_user_event(true);
    }
    if (is_transfer_aligned(get_offset(), _bytes_count)) {
        return vulkan_stream.enqueue_buffer_fill(get_allocation(), get_offset(), _bytes_count, make_fill_pattern(pattern), _region, dep_events, blocking);
    }

    stream.wait_for_events(dep_events);
    stream.finish();
    const auto region_size = _region->get_size();
    auto staging = allocate_staging(static_cast<size_t>(region_size));
    vulkan_stream.enqueue_buffer_copy(get_allocation(), _region->get_offset(), staging, 0, region_size, true, {_region});
    staging->invalidate(0, region_size);
    std::memset(static_cast<unsigned char*>(staging->mapped_data) + _view_offset, pattern, _bytes_count);
    staging->flush(0, region_size);
    return vulkan_stream.enqueue_buffer_copy(staging, 0, get_allocation(), _region->get_offset(), region_size, blocking, {_region});
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

event::ptr vulkan_buffer::copy_from(stream& stream, const void* source, size_t source_offset, size_t destination_offset, size_t size, bool blocking) {
    const auto& vulkan_stream = validate_stream(stream);
    validate_range(destination_offset, size, "copy_from(host)");
    OPENVINO_ASSERT(source != nullptr || size == 0, "[GPU][Vulkan] Source pointer is null");
    if (size == 0) {
        return stream.create_user_event(true);
    }
    const auto* source_bytes = static_cast<const unsigned char*>(source) + source_offset;
    if (get_allocation()->is_host_visible()) {
        stream.finish();
        if (!get_allocation()->is_host_coherent()) {
            get_allocation()->invalidate(get_offset() + destination_offset, size);
        }
        auto* destination_bytes = static_cast<unsigned char*>(mapped_data()) + destination_offset;
        std::memcpy(destination_bytes, source_bytes, size);
        get_allocation()->flush(get_offset() + destination_offset, size);
        return stream.create_user_event(true);
    }

    const auto region_size = _region->get_size();
    auto staging = allocate_staging(static_cast<size_t>(region_size));
    const auto staging_offset = _view_offset + destination_offset;
    const auto replaces_region = staging_offset == 0 && size == region_size;
    if (!replaces_region) {
        vulkan_stream.enqueue_buffer_copy(get_allocation(), _region->get_offset(), staging, 0, region_size, true, {_region});
        staging->invalidate(0, region_size);
    }
    std::memcpy(static_cast<unsigned char*>(staging->mapped_data) + staging_offset, source_bytes, size);
    staging->flush(0, region_size);
    return vulkan_stream.enqueue_buffer_copy(staging, 0, get_allocation(), _region->get_offset(), region_size, blocking, {_region});
}

event::ptr vulkan_buffer::copy_from(stream& stream, const memory& source, size_t source_offset, size_t destination_offset, size_t size, bool blocking) {
    const auto& vulkan_stream = validate_stream(stream);
    const auto* source_buffer = dynamic_cast<const vulkan_buffer*>(&source);
    OPENVINO_ASSERT(source_buffer != nullptr && source.get_engine() == _engine, "[GPU][Vulkan] Device copy source is not a Vulkan buffer from this engine");
    source_buffer->validate_range(source_offset, size, "copy_from(device source)");
    validate_range(destination_offset, size, "copy_from(device destination)");
    if (size == 0) {
        return stream.create_user_event(true);
    }
    const auto absolute_source_offset = source_buffer->get_offset() + source_offset;
    const auto absolute_destination_offset = get_offset() + destination_offset;
    const auto same_allocation = source_buffer->get_allocation() == get_allocation();
    if (same_allocation && absolute_source_offset == absolute_destination_offset) {
        return stream.create_user_event(true);
    }
    const auto ranges_overlap =
        same_allocation && (absolute_source_offset < absolute_destination_offset ? size > absolute_destination_offset - absolute_source_offset
                                                                                 : size > absolute_source_offset - absolute_destination_offset);
    const auto both_host_visible = source_buffer->get_allocation()->is_host_visible() && get_allocation()->is_host_visible();
    const auto both_host_cached = source_buffer->get_allocation()->is_host_cached() && get_allocation()->is_host_cached();
    if (both_host_visible && both_host_cached) {
        stream.finish();
        source_buffer->get_allocation()->invalidate(absolute_source_offset, size);
        if (!get_allocation()->is_host_coherent()) {
            get_allocation()->invalidate(absolute_destination_offset, size);
        }
        const auto* source_bytes = static_cast<const unsigned char*>(source_buffer->mapped_data()) + source_offset;
        auto* destination_bytes = static_cast<unsigned char*>(mapped_data()) + destination_offset;
        std::memmove(destination_bytes, source_bytes, size);
        get_allocation()->flush(absolute_destination_offset, size);
        return stream.create_user_event(true);
    }
    if (!ranges_overlap && is_transfer_aligned(absolute_source_offset, size) && is_transfer_aligned(absolute_destination_offset, size)) {
        return vulkan_stream.enqueue_buffer_copy(source_buffer->get_allocation(),
                                                 absolute_source_offset,
                                                 get_allocation(),
                                                 absolute_destination_offset,
                                                 size,
                                                 blocking,
                                                 {source_buffer->get_region(), get_region()});
    }

    std::vector<unsigned char> temporary(size);
    source_buffer->copy_to(stream, temporary.data(), source_offset, 0, size, true);
    return copy_from(stream, temporary.data(), 0, destination_offset, size, blocking);
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
            get_allocation()->invalidate(get_offset() + source_offset, size);
            source_bytes = static_cast<const unsigned char*>(mapped_data()) + source_offset;
        } else {
            const auto region_size = _region->get_size();
            staging = allocate_staging(static_cast<size_t>(region_size));
            vulkan_stream.enqueue_buffer_copy(get_allocation(), _region->get_offset(), staging, 0, region_size, true, {_region});
            staging->invalidate(0, region_size);
            source_bytes = static_cast<const unsigned char*>(staging->mapped_data) + _view_offset + source_offset;
        }
        auto* destination_bytes = static_cast<unsigned char*>(destination) + destination_offset;
        std::memcpy(destination_bytes, source_bytes, size);
    }
    return stream.create_user_event(true);
}

}  // namespace vulkan
}  // namespace cldnn
