// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vulkan_device.hpp"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "openvino/core/except.hpp"
#include "vulkan_pipeline_cache.hpp"
#include "vulkan_required_device_profile.hpp"

namespace cldnn {
namespace vulkan {
namespace {

constexpr const char* portability_subset_extension_name = "VK_KHR_portability_subset";
constexpr const char* external_memory_host_extension_name = "VK_EXT_external_memory_host";
constexpr const char* external_memory_fd_extension_name = "VK_KHR_external_memory_fd";
constexpr const char* external_memory_dma_buf_extension_name = "VK_EXT_external_memory_dma_buf";
constexpr const char* external_memory_ahb_extension_name = "VK_ANDROID_external_memory_android_hardware_buffer";
#if defined(__APPLE__)
constexpr const char* external_memory_metal_extension_name = "VK_EXT_external_memory_metal";
#endif
constexpr const char* queue_family_foreign_extension_name = "VK_EXT_queue_family_foreign";

void check_vk_result(VkResult result, const char* operation) {
    OPENVINO_ASSERT(result == VK_SUCCESS, "[GPU][Vulkan] ", operation, " failed with VkResult ", static_cast<int>(result));
}

std::string make_driver_version(const VkPhysicalDeviceProperties& properties) {
    std::ostringstream stream;
    stream << "Vulkan " << VK_API_VERSION_MAJOR(properties.apiVersion) << "." << VK_API_VERSION_MINOR(properties.apiVersion) << "."
           << VK_API_VERSION_PATCH(properties.apiVersion) << ", driver 0x" << std::hex << properties.driverVersion;
    return stream.str();
}

bool has_extension(const std::vector<VkExtensionProperties>& extensions, const char* name) {
    return std::any_of(extensions.begin(), extensions.end(), [name](const auto& extension) {
        return std::string(extension.extensionName) == name;
    });
}

VkExternalMemoryProperties query_external_buffer_properties(VkPhysicalDevice device, VkExternalMemoryHandleTypeFlagBits handle_type) {
    VkPhysicalDeviceExternalBufferInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO;
    buffer_info.flags = 0;
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer_info.handleType = handle_type;

    VkExternalBufferProperties properties{};
    properties.sType = VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES;
    vkGetPhysicalDeviceExternalBufferProperties(device, &buffer_info, &properties);
    return properties.externalMemoryProperties;
}

}  // namespace

vulkan_device::vulkan_device(vulkan_instance::ptr instance, VkPhysicalDevice physical_device, uint32_t compute_queue_family, bool initialize_device)
    : _instance(std::move(instance)),
      _physical_device(physical_device),
      _compute_queue_family(compute_queue_family) {
    OPENVINO_ASSERT(_instance != nullptr && _physical_device != VK_NULL_HANDLE,
                    "[GPU][Vulkan] Cannot create a device wrapper for a null instance or physical device");
    initialize_info();
    if (initialize_device) {
        initialize();
    }
}

vulkan_device::~vulkan_device() {
    if (_device != VK_NULL_HANDLE) {
        _pipeline_cache.reset();
        vkDestroyDevice(_device, nullptr);
    }
}

void vulkan_device::initialize_info() {
    VkPhysicalDeviceIDProperties id_properties{};
    id_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;

    VkPhysicalDeviceSubgroupProperties subgroup_properties{};
    subgroup_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    id_properties.pNext = &subgroup_properties;

    VkPhysicalDeviceProperties2 properties2{};
    properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties2.pNext = &id_properties;
    vkGetPhysicalDeviceProperties2(_physical_device, &properties2);
    const auto& properties = properties2.properties;

    VkPhysicalDeviceFeatures features{};
    vkGetPhysicalDeviceFeatures(_physical_device, &features);

    VkPhysicalDeviceMemoryProperties memory_properties{};
    vkGetPhysicalDeviceMemoryProperties(_physical_device, &memory_properties);

    uint64_t device_local_memory = 0;
    uint64_t total_memory = 0;
    for (uint32_t index = 0; index < memory_properties.memoryHeapCount; ++index) {
        const auto& heap = memory_properties.memoryHeaps[index];
        total_memory += heap.size;
        if ((heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) {
            device_local_memory += heap.size;
        }
    }
    _info.max_global_mem_size = device_local_memory != 0 ? device_local_memory : total_memory;
    _info.max_alloc_mem_size = std::min<uint64_t>(_info.max_global_mem_size, properties.limits.maxStorageBufferRange);
    _info.max_global_cache_size = 0;

    _info.execution_units_count = 0;
    _info.gpu_frequency = 0;
    _info.max_work_group_size = properties.limits.maxComputeWorkGroupInvocations;
    _info.max_local_mem_size = properties.limits.maxComputeSharedMemorySize;
    _info.max_image2d_width = properties.limits.maxImageDimension2D;
    _info.max_image2d_height = properties.limits.maxImageDimension2D;
    _max_push_constants_size = properties.limits.maxPushConstantsSize;
    _max_memory_allocation_count = std::max(properties.limits.maxMemoryAllocationCount, 1U);
    _non_coherent_atom_size = std::max<VkDeviceSize>(properties.limits.nonCoherentAtomSize, 1);

    // Eltwise uses a portable FP16 storage conversion path and performs arithmetic in FP32.
    _info.supports_fp16 = true;
    _info.supports_fp64 = features.shaderFloat64 == VK_TRUE;
    _info.supports_fp16_denorms = false;
    _info.supports_khr_subgroups = subgroup_properties.subgroupSize > 0 && (subgroup_properties.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0;
    _info.supports_intel_subgroups = false;
    _info.supports_intel_subgroups_short = false;
    _info.supports_intel_subgroups_char = false;
    _info.supports_intel_required_subgroup_size = false;
    _info.supports_queue_families = false;
    _info.supports_image = properties.limits.maxImageDimension2D > 0;
    _info.supports_intel_planar_yuv = false;
    _info.supports_work_group_collective_functions = false;
    _info.supports_non_uniform_work_group = false;
    _info.supports_imad = false;
    _info.supports_immad = false;
    _info.supports_mutable_command_list = false;
    _info.supports_usm = false;
    _info.has_separate_cache = properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
    _info.supports_cp_offload = false;
    _info.supports_counter_based_events = false;
    _info.supports_leo = false;

    if (_info.supports_khr_subgroups) {
        _info.supported_simd_sizes = {subgroup_properties.subgroupSize};
    }

    _info.vendor_id = properties.vendorID;
    _info.dev_name = properties.deviceName;
    _info.driver_version = make_driver_version(properties);
    _info.dev_type = properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? device_type::discrete_gpu : device_type::integrated_gpu;
    _info.gfx_ver = {0, 0, 0};
    _info.arch = gpu_arch::unknown;
    _info.ip_version = 0;
    _info.device_id = properties.deviceID;
    _info.num_slices = 0;
    _info.num_sub_slices_per_slice = 0;
    _info.num_eus_per_sub_slice = 0;
    _info.num_threads_per_eu = 0;
    _info.num_ccs = 1;
    _info.sub_device_idx = (std::numeric_limits<uint32_t>::max)();

    if (properties.limits.minStorageBufferOffsetAlignment <= (std::numeric_limits<uint32_t>::max)()) {
        _info.sub_buffer_base_alignment = static_cast<uint32_t>(properties.limits.minStorageBufferOffsetAlignment);
    }

    _info.timer_resolution = 0;
    _info.kernel_timestamp_valid_bits = 0;
    _info.compute_queue_group_ordinal = _compute_queue_family;
    _info.device_memory_ordinal = 0;

    static_assert(VK_UUID_SIZE == ov::device::UUID::MAX_UUID_SIZE, "Unexpected Vulkan UUID size");
    std::copy_n(id_properties.deviceUUID, VK_UUID_SIZE, _info.uuid.uuid.begin());
    std::fill(_info.luid.luid.begin(), _info.luid.luid.end(), 0);
    if (id_properties.deviceLUIDValid == VK_TRUE) {
        static_assert(VK_LUID_SIZE == ov::device::LUID::MAX_LUID_SIZE, "Unexpected Vulkan LUID size");
        std::copy_n(id_properties.deviceLUID, VK_LUID_SIZE, _info.luid.luid.begin());
    }
}

void vulkan_device::initialize() {
    std::lock_guard<std::mutex> lock(_init_mutex);
    if (_device != VK_NULL_HANDLE) {
        return;
    }

    uint32_t extension_count = 0;
    check_vk_result(vkEnumerateDeviceExtensionProperties(_physical_device, nullptr, &extension_count, nullptr), "vkEnumerateDeviceExtensionProperties(count)");
    std::vector<VkExtensionProperties> extensions(extension_count);
    if (extension_count > 0) {
        check_vk_result(vkEnumerateDeviceExtensionProperties(_physical_device, nullptr, &extension_count, extensions.data()),
                        "vkEnumerateDeviceExtensionProperties(list)");
    }

    std::vector<const char*> enabled_extensions;
    if (has_extension(extensions, portability_subset_extension_name)) {
        enabled_extensions.push_back(portability_subset_extension_name);
    }

    const auto enable_external_import = [&](const char* extension_name, VkExternalMemoryHandleTypeFlagBits handle_type) {
        if (!has_extension(extensions, extension_name)) {
            return false;
        }
        const auto external = query_external_buffer_properties(_physical_device, handle_type);
        if ((external.externalMemoryFeatures & VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT) == 0 || (external.compatibleHandleTypes & handle_type) == 0) {
            return false;
        }
        enabled_extensions.push_back(extension_name);
        _external_memory_capabilities.importable_buffer_handle_types |= handle_type;
        if ((external.externalMemoryFeatures & VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT) != 0) {
            _external_memory_capabilities.dedicated_buffer_handle_types |= handle_type;
        }
        return true;
    };

    if (enable_external_import(external_memory_host_extension_name, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT)) {
        VkPhysicalDeviceExternalMemoryHostPropertiesEXT host_properties{};
        host_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT;
        VkPhysicalDeviceProperties2 properties{};
        properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        properties.pNext = &host_properties;
        vkGetPhysicalDeviceProperties2(_physical_device, &properties);
        _external_memory_capabilities.min_imported_host_pointer_alignment = std::max<VkDeviceSize>(host_properties.minImportedHostPointerAlignment, 1);
    }

    const bool queue_family_foreign_available = has_extension(extensions, queue_family_foreign_extension_name);
    const bool dma_buf_enabled = queue_family_foreign_available && has_extension(extensions, external_memory_fd_extension_name) &&
                                 enable_external_import(external_memory_dma_buf_extension_name, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT);
    if (dma_buf_enabled) {
        enabled_extensions.push_back(external_memory_fd_extension_name);
    }

    const bool ahb_enabled = queue_family_foreign_available &&
                             enable_external_import(external_memory_ahb_extension_name, VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID);
#if defined(__APPLE__)
    enable_external_import(external_memory_metal_extension_name, VK_EXTERNAL_MEMORY_HANDLE_TYPE_MTLBUFFER_BIT_EXT);
#endif

    if (dma_buf_enabled || ahb_enabled) {
        enabled_extensions.push_back(queue_family_foreign_extension_name);
    }

    constexpr float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_create_info{};
    queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_create_info.queueFamilyIndex = _compute_queue_family;
    queue_create_info.queueCount = 1;
    queue_create_info.pQueuePriorities = &queue_priority;

    const auto required_profile = query_vulkan_required_device_profile(_physical_device);
    OPENVINO_ASSERT(required_profile.is_supported(), "[GPU][Vulkan] ", required_profile.incompatibility_reason());

    VkPhysicalDevice8BitStorageFeatures enabled_storage8{};
    enabled_storage8.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES;
    enabled_storage8.storageBuffer8BitAccess = VK_TRUE;
    VkPhysicalDeviceSynchronization2Features enabled_synchronization2{};
    enabled_synchronization2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES;
    enabled_synchronization2.synchronization2 = VK_TRUE;
    VkPhysicalDeviceTimelineSemaphoreFeatures enabled_timeline{};
    enabled_timeline.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
    enabled_timeline.timelineSemaphore = VK_TRUE;
    VkPhysicalDeviceMaintenance4Features enabled_maintenance4{};
    enabled_maintenance4.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES;
    enabled_maintenance4.maintenance4 = VK_TRUE;
    enabled_storage8.pNext = &enabled_synchronization2;
    enabled_synchronization2.pNext = &enabled_timeline;
    enabled_timeline.pNext = &enabled_maintenance4;

    VkDeviceCreateInfo device_create_info{};
    device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_create_info.pNext = &enabled_storage8;
    device_create_info.queueCreateInfoCount = 1;
    device_create_info.pQueueCreateInfos = &queue_create_info;
    device_create_info.enabledExtensionCount = static_cast<uint32_t>(enabled_extensions.size());
    device_create_info.ppEnabledExtensionNames = enabled_extensions.empty() ? nullptr : enabled_extensions.data();

    check_vk_result(vkCreateDevice(_physical_device, &device_create_info, nullptr, &_device), "vkCreateDevice");
    vkGetDeviceQueue(_device, _compute_queue_family, 0, &_compute_queue);
    OPENVINO_ASSERT(_compute_queue != VK_NULL_HANDLE, "[GPU][Vulkan] vkGetDeviceQueue returned a null compute queue");
    try {
        _pipeline_cache = std::make_unique<vulkan_pipeline_cache>(_device, _physical_device);
    } catch (...) {
        vkDestroyDevice(_device, nullptr);
        _compute_queue = VK_NULL_HANDLE;
        _device = VK_NULL_HANDLE;
        throw;
    }
}

bool vulkan_device::is_initialized() const {
    std::lock_guard<std::mutex> lock(_init_mutex);
    return _device != VK_NULL_HANDLE;
}

bool vulkan_device::is_same(const device::ptr other) {
    const auto other_vulkan_device = std::dynamic_pointer_cast<vulkan_device>(other);
    return other_vulkan_device != nullptr && (_physical_device == other_vulkan_device->_physical_device || _info.is_same_device(other_vulkan_device->_info));
}

void vulkan_device::set_mem_caps(const memory_capabilities& memory_capabilities) {
    _mem_caps = memory_capabilities;
}

vulkan_pipeline_cache& vulkan_device::get_pipeline_cache() const {
    OPENVINO_ASSERT(_pipeline_cache != nullptr, "[GPU][Vulkan] Pipeline cache requested before device initialization");
    return *_pipeline_cache;
}

}  // namespace vulkan
}  // namespace cldnn
