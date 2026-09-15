// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>

#include "vulkan_memory.hpp"

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

#include "openvino/core/except.hpp"
#include "vulkan_engine.hpp"
#include "vulkan_memory_internal.hpp"

namespace cldnn::vulkan {
namespace {

struct vulkan_external_buffer_import final {
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
        detail::check_vulkan_memory_result(create_result, "vkCreateBuffer(external)");
    }

    VkMemoryDedicatedRequirements dedicated_requirements{};
    dedicated_requirements.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS;
    VkMemoryRequirements2 requirements_info{};
    requirements_info.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
    requirements_info.pNext = &dedicated_requirements;
    VkBufferMemoryRequirementsInfo2 buffer_requirements_info{};
    buffer_requirements_info.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2;
    buffer_requirements_info.buffer = buffer;
    vkGetBufferMemoryRequirements2(engine.get_device_handle(), &buffer_requirements_info, &requirements_info);
    const auto& requirements = requirements_info.memoryRequirements;
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
    const bool requires_dedicated_allocation = dedicated_requirements.requiresDedicatedAllocation == VK_TRUE ||
                                               device_owner->get_external_memory_capabilities().requires_dedicated_allocation(import.handle_type);
    if (requires_dedicated_allocation && import.handle_type == VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT) {
        vkDestroyBuffer(engine.get_device_handle(), buffer, nullptr);
        close_imported_fd(import);
        OPENVINO_THROW("[GPU][Vulkan] Imported host allocation cannot satisfy a dedicated buffer requirement");
    }

    VkMemoryAllocateInfo allocation_info{};
    allocation_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocation_info.pNext = requires_dedicated_allocation ? static_cast<const void*>(&dedicated_info) : import.allocation_pnext;
    allocation_info.allocationSize = allocation_size;
    allocation_info.memoryTypeIndex = memory_type.index;

    VkDeviceMemory device_memory = VK_NULL_HANDLE;
    const auto allocation_result = vkAllocateMemory(engine.get_device_handle(), &allocation_info, nullptr, &device_memory);
    if (allocation_result != VK_SUCCESS) {
        vkDestroyBuffer(engine.get_device_handle(), buffer, nullptr);
        close_imported_fd(import);
        detail::check_vulkan_memory_result(allocation_result, "vkAllocateMemory(external)");
    }
    import.imported_fd = -1;

    const auto bind_result = vkBindBufferMemory(engine.get_device_handle(), buffer, device_memory, 0);
    if (bind_result != VK_SUCCESS) {
        vkDestroyBuffer(engine.get_device_handle(), buffer, nullptr);
        vkFreeMemory(engine.get_device_handle(), device_memory, nullptr);
        detail::check_vulkan_memory_result(bind_result, "vkBindBufferMemory(external)");
    }

    void* mapped_data = nullptr;
    if (import.map_memory) {
        const auto map_result = vkMapMemory(engine.get_device_handle(), device_memory, 0, allocation_size, 0, &mapped_data);
        if (map_result != VK_SUCCESS) {
            vkDestroyBuffer(engine.get_device_handle(), buffer, nullptr);
            vkFreeMemory(engine.get_device_handle(), device_memory, nullptr);
            detail::check_vulkan_memory_result(map_result, "vkMapMemory(external)");
        }
    }

    if (import.foreign_owned && !detail::transition_foreign_buffer_ownership(device_owner, buffer, true)) {
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
    detail::check_vulkan_memory_result(get_properties(engine.get_device_handle(), VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, address, &properties),
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
    struct stat source_status {};
    OPENVINO_ASSERT(fstat(source_fd, &source_status) == 0 && source_status.st_size > 0, "[GPU][Vulkan] Failed to query DMA-BUF allocation size");
    const auto allocation_size = static_cast<VkDeviceSize>(source_status.st_size);
    OPENVINO_ASSERT(allocation_size >= buffer_size, "[GPU][Vulkan] DMA-BUF allocation is smaller than the tensor: ", allocation_size, " < ", buffer_size);

    const auto get_properties = reinterpret_cast<PFN_vkGetMemoryFdPropertiesKHR>(vkGetDeviceProcAddr(engine.get_device_handle(), "vkGetMemoryFdPropertiesKHR"));
    OPENVINO_ASSERT(get_properties != nullptr, "[GPU][Vulkan] vkGetMemoryFdPropertiesKHR is unavailable");
    VkMemoryFdPropertiesKHR properties{};
    properties.sType = VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR;
    detail::check_vulkan_memory_result(get_properties(engine.get_device_handle(), VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, source_fd, &properties),
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
    detail::check_vulkan_memory_result(get_properties(engine.get_device_handle(), hardware_buffer, &properties), "vkGetAndroidHardwareBufferPropertiesANDROID");

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
    detail::check_vulkan_memory_result(
        get_properties(engine.get_device_handle(), VK_EXTERNAL_MEMORY_HANDLE_TYPE_MTLBUFFER_BIT_EXT, external_handle, &properties),
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

}  // namespace cldnn::vulkan
