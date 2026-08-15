// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vk_device.hpp"

#include <algorithm>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

namespace ov::core::vulkan {
namespace cross_platform {

namespace {

device_type get_device_type(VkPhysicalDeviceType type) {
    return type == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? device_type::discrete_gpu
                                                        : device_type::integrated_gpu;
}

device_info init_device_info(VkPhysicalDevice physical_device) {
    device_info info;

    VkPhysicalDeviceProperties props = {};
    vkGetPhysicalDeviceProperties(physical_device, &props);

    info.vendor_id = props.vendorID;
    info.dev_name = props.deviceName;
    info.driver_version = std::to_string(VK_API_VERSION_MAJOR(props.driverVersion)) + "." +
                          std::to_string(VK_API_VERSION_MINOR(props.driverVersion)) + "." +
                          std::to_string(VK_API_VERSION_PATCH(props.driverVersion));
    info.dev_type = get_device_type(props.deviceType);

    info.max_work_group_size = props.limits.maxComputeWorkGroupInvocations;
    info.max_local_mem_size = props.limits.maxComputeSharedMemorySize;
    info.max_global_cache_size = 0;
    info.max_image2d_width = props.limits.maxImageDimension2D;
    info.max_image2d_height = props.limits.maxImageDimension2D;

    VkPhysicalDeviceMemoryProperties mem_props = {};
    vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_props);
    uint64_t global_mem_size = 0;
    uint64_t max_heap_size = 0;
    for (uint32_t i = 0; i < mem_props.memoryHeapCount; ++i) {
        global_mem_size += mem_props.memoryHeaps[i].size;
        max_heap_size = std::max(max_heap_size, static_cast<uint64_t>(mem_props.memoryHeaps[i].size));
    }
    info.max_global_mem_size = global_mem_size;
    info.max_alloc_mem_size = max_heap_size;

    VkPhysicalDeviceShaderFloat16Int8Features float16_int8_features = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
    VkPhysicalDeviceFeatures2 features2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features2.pNext = &float16_int8_features;
    vkGetPhysicalDeviceFeatures2(physical_device, &features2);

    info.supports_fp16 = float16_int8_features.shaderFloat16;
    info.supports_fp64 = features2.features.shaderFloat64;
    info.supports_fp16_denorms = false;

    info.supports_khr_subgroups = true;  // subgroup operations are core since Vulkan 1.1
    info.supports_intel_subgroups = false;
    info.supports_intel_subgroups_short = false;
    info.supports_intel_subgroups_char = false;
    info.supports_intel_required_subgroup_size = false;
    info.supports_queue_families = false;
    info.supports_image = true;
    info.supports_intel_planar_yuv = false;
    info.supports_work_group_collective_functions = false;
    info.supports_non_uniform_work_group = true;

    info.supports_imad = false;
    info.supports_immad = false;
    info.supports_mutable_command_list = false;
    info.supports_usm = false;
    info.has_separate_cache = false;
    info.supports_cp_offload = false;
    info.supports_counter_based_events = false;
    info.supports_leo = false;

    // Vulkan does not expose the number of execution units; approximate value for interface compatibility
    info.execution_units_count = (info.dev_type == device_type::discrete_gpu) ? 2048 : 512;

    info.gpu_frequency = 0;

    VkPhysicalDeviceSubgroupProperties subgroup_props = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceIDProperties id_props = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    VkPhysicalDeviceProperties2 props2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    props2.pNext = &subgroup_props;
    subgroup_props.pNext = &id_props;
    vkGetPhysicalDeviceProperties2(physical_device, &props2);

    info.supported_simd_sizes = {static_cast<size_t>(subgroup_props.subgroupSize)};

    static_assert(VK_UUID_SIZE == ov::device::UUID::MAX_UUID_SIZE, "Vulkan UUID size mismatch");
    std::memcpy(info.uuid.uuid.data(), id_props.deviceUUID, VK_UUID_SIZE);
    if (id_props.deviceLUIDValid) {
        static_assert(VK_LUID_SIZE == ov::device::LUID::MAX_LUID_SIZE, "Vulkan LUID size mismatch");
        std::memcpy(info.luid.luid.data(), id_props.deviceLUID, VK_LUID_SIZE);
    } else {
        std::fill_n(std::begin(info.luid.luid), ov::device::LUID::MAX_LUID_SIZE, 0);
    }

    // Intel-specific fields, not applicable to non-Intel GPUs
    info.gfx_ver = {0, 0, 0};
    info.arch = gpu_arch::unknown;
    info.ip_version = 0;
    info.device_id = props.deviceID;
    info.num_slices = 0;
    info.num_sub_slices_per_slice = 0;
    info.num_eus_per_sub_slice = 0;
    info.num_threads_per_eu = 0;
    info.num_ccs = 1;
    info.sub_device_idx = 0;

    info.timer_resolution = 0;
    info.kernel_timestamp_valid_bits = 0;
    info.compute_queue_group_ordinal = 0;
    info.device_memory_ordinal = 0;

    return info;
}

}  // namespace

vk_device::vk_device(const instance_ptr& instance, VkPhysicalDevice physical_device)
    : _instance(instance)
    , _physical_device(physical_device)
    , _info(init_device_info(physical_device)) {
    VkPhysicalDeviceProperties props = {};
    vkGetPhysicalDeviceProperties(physical_device, &props);
    _physical_device_type = props.deviceType;

    _queue_family_index = find_compute_queue_family(physical_device);

    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_create_info = {};
    queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_create_info.queueFamilyIndex = _queue_family_index;
    queue_create_info.queueCount = 1;
    queue_create_info.pQueuePriorities = &queue_priority;

    // Portability devices (MoltenVK on Apple platforms) expose
    // VK_KHR_portability_subset; it must be enabled at device creation or
    // vkCreateDevice fails. Not present on desktop drivers, so this is a
    // no-op there. (The name macro lives in vulkan_beta.h in newer headers,
    // so it is spelled out.)
    constexpr std::string_view k_portability_subset_extension = "VK_KHR_portability_subset";
    std::vector<const char*> enabled_extensions;
    uint32_t extension_count = 0;
    vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, nullptr);
    std::vector<VkExtensionProperties> extension_props(extension_count);
    vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, extension_props.data());
    for (const auto& ext : extension_props) {
        if (ext.extensionName == k_portability_subset_extension) {
            enabled_extensions.push_back(ext.extensionName);
            break;
        }
    }

    VkDeviceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    create_info.queueCreateInfoCount = 1;
    create_info.pQueueCreateInfos = &queue_create_info;
    if (!enabled_extensions.empty()) {
        create_info.enabledExtensionCount = static_cast<uint32_t>(enabled_extensions.size());
        create_info.ppEnabledExtensionNames = enabled_extensions.data();
    }

    VkDevice device = VK_NULL_HANDLE;
    VK_CALL(vkCreateDevice(physical_device, &create_info, nullptr, &device), "vkCreateDevice");
    _device = make_device(device);

    vkGetDeviceQueue(device, _queue_family_index, 0, &_queue);
    if (_queue == VK_NULL_HANDLE) {
        throw vulkan_error("vk_device: failed to get device queue");
    }
}

}  // namespace cross_platform
}  // namespace ov::core::vulkan
