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
#include "vulkan_api.hpp"
#include "vulkan_required_device_profile.hpp"

namespace cldnn {
namespace vulkan {
namespace {

constexpr const char* portability_subset_extension_name = "VK_KHR_portability_subset";

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
        vkDestroyDevice(_device, nullptr);
    }
}

void vulkan_device::initialize_info() {
    VkPhysicalDeviceIDProperties id_properties{};
    id_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;

    VkPhysicalDeviceSubgroupProperties subgroup_properties{};
    subgroup_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    id_properties.pNext = &subgroup_properties;

    VkPhysicalDeviceMaintenance3Properties maintenance3_properties{};
    maintenance3_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES;
    subgroup_properties.pNext = &maintenance3_properties;

    VkPhysicalDeviceMaintenance4Properties maintenance4_properties{};
    maintenance4_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_PROPERTIES;
    maintenance3_properties.pNext = &maintenance4_properties;

    VkPhysicalDeviceProperties2 properties2{};
    properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties2.pNext = &id_properties;
    vkGetPhysicalDeviceProperties2(_physical_device, &properties2);
    const auto& properties = properties2.properties;

    VkPhysicalDevice16BitStorageFeatures storage{};
    storage.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
    VkPhysicalDeviceFeatures2 features{};
    features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features.pNext = &storage;
    vkGetPhysicalDeviceFeatures2(_physical_device, &features);
    _info.supports_fp16 = storage.storageBuffer16BitAccess == VK_TRUE;

    vkGetPhysicalDeviceMemoryProperties(_physical_device, &_memory_properties);

    VkDeviceSize device_local_memory = 0;
    VkDeviceSize total_memory = 0;
    VkDeviceSize largest_device_local_heap = 0;
    VkDeviceSize largest_heap = 0;
    for (uint32_t index = 0; index < _memory_properties.memoryHeapCount; ++index) {
        const auto& heap = _memory_properties.memoryHeaps[index];
        total_memory += heap.size;
        largest_heap = std::max(largest_heap, heap.size);
        if ((heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) {
            device_local_memory += heap.size;
            largest_device_local_heap = std::max(largest_device_local_heap, heap.size);
        }
    }
    _info.max_global_mem_size = device_local_memory != 0 ? device_local_memory : total_memory;
    _max_buffer_size = maintenance4_properties.maxBufferSize;
    _max_memory_allocation_size = maintenance3_properties.maxMemoryAllocationSize;
    _non_coherent_atom_size = std::max<VkDeviceSize>(properties.limits.nonCoherentAtomSize, 1);
    // The Vulkan storage offset and non-coherent atom alignments are powers of two.
    _buffer_alignment = std::max({properties.limits.minStorageBufferOffsetAlignment, _non_coherent_atom_size, VkDeviceSize{sizeof(uint32_t)}});
    const auto storage_heap_size = largest_device_local_heap != 0 ? largest_device_local_heap : largest_heap;
    const auto storage_limit =
        std::min<VkDeviceSize>({storage_heap_size, properties.limits.maxStorageBufferRange, _max_memory_allocation_size, _max_buffer_size});
    _info.max_alloc_mem_size = storage_limit - storage_limit % _buffer_alignment;
    _info.max_global_cache_size = 0;

    _info.execution_units_count = 0;
    _info.gpu_frequency = 0;
    _info.max_work_group_size = properties.limits.maxComputeWorkGroupInvocations;
    _info.max_local_mem_size = properties.limits.maxComputeSharedMemorySize;
    _info.max_image2d_width = properties.limits.maxImageDimension2D;
    _info.max_image2d_height = properties.limits.maxImageDimension2D;
    _max_memory_allocation_count = std::max(properties.limits.maxMemoryAllocationCount, 1U);

    _info.supports_khr_subgroups = subgroup_properties.subgroupSize > 0 && (subgroup_properties.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0;
    _info.supports_intel_subgroups = false;
    _info.supports_intel_subgroups_short = false;
    _info.supports_intel_subgroups_char = false;
    _info.supports_intel_required_subgroup_size = false;
    _info.supports_queue_families = false;
    _info.supports_image = false;
    _info.supports_intel_planar_yuv = false;
    _info.supports_work_group_collective_functions = false;
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

    const auto extensions =
        enumerate_vulkan<VkExtensionProperties>("vkEnumerateDeviceExtensionProperties", [this](uint32_t* count, VkExtensionProperties* entries) {
            return vkEnumerateDeviceExtensionProperties(_physical_device, nullptr, count, entries);
        });

    std::vector<const char*> enabled_extensions;
    if (has_extension(extensions, portability_subset_extension_name)) {
        enabled_extensions.push_back(portability_subset_extension_name);
    }

    constexpr float queue_priority = 1.0F;
    VkDeviceQueueCreateInfo queue_create_info{};
    queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_create_info.queueFamilyIndex = _compute_queue_family;
    queue_create_info.queueCount = 1;
    queue_create_info.pQueuePriorities = &queue_priority;

    const auto required_profile = query_vulkan_required_device_profile(_physical_device);
    OPENVINO_ASSERT(required_profile.is_supported(), "[GPU][Vulkan] ", required_profile.incompatibility_reason());

    VkPhysicalDeviceSynchronization2Features enabled_synchronization2{};
    enabled_synchronization2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES;
    enabled_synchronization2.synchronization2 = VK_TRUE;

    VkDeviceCreateInfo device_create_info{};
    device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_create_info.pNext = &enabled_synchronization2;
    device_create_info.queueCreateInfoCount = 1;
    device_create_info.pQueueCreateInfos = &queue_create_info;
    device_create_info.enabledExtensionCount = static_cast<uint32_t>(enabled_extensions.size());
    device_create_info.ppEnabledExtensionNames = enabled_extensions.empty() ? nullptr : enabled_extensions.data();

    VkDevice device = VK_NULL_HANDLE;
    check_vk_result(vkCreateDevice(_physical_device, &device_create_info, nullptr, &device), "vkCreateDevice");
    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, _compute_queue_family, 0, &queue);
    if (queue == VK_NULL_HANDLE) {
        vkDestroyDevice(device, nullptr);
        OPENVINO_THROW("[GPU][Vulkan] vkGetDeviceQueue returned a null compute queue");
    }
    _device = device;
    _compute_queue = queue;
}

bool vulkan_device::is_initialized() const {
    std::lock_guard<std::mutex> lock(_init_mutex);
    return _device != VK_NULL_HANDLE;
}

bool vulkan_memory_budget::try_reserve(uint32_t heap_index, VkDeviceSize size, const limits& capacity) noexcept {
    if (heap_index >= _heap_usage.size() || size == 0) {
        return false;
    }
    const std::lock_guard<std::mutex> lock(_mutex);
    auto& heap_usage = _heap_usage[heap_index];
    if (_allocation_count >= capacity.allocations || size > capacity.heap - std::min(heap_usage, capacity.heap) ||
        size > capacity.total - std::min(_total_usage, capacity.total)) {
        return false;
    }
    heap_usage += size;
    _total_usage += size;
    ++_allocation_count;
    return true;
}

void vulkan_memory_budget::release(uint32_t heap_index, VkDeviceSize size) noexcept {
    const std::lock_guard<std::mutex> lock(_mutex);
    _heap_usage[heap_index] -= size;
    _total_usage -= size;
    --_allocation_count;
}

bool vulkan_device::try_reserve_memory(uint32_t heap_index, VkDeviceSize size, VkDeviceSize limit) noexcept {
    return heap_index < _memory_properties.memoryHeapCount &&
           _memory_budget.try_reserve(heap_index, size, {limit, _memory_properties.memoryHeaps[heap_index].size, _max_memory_allocation_count});
}

void vulkan_device::release_memory(uint32_t heap_index, VkDeviceSize size) noexcept {
    _memory_budget.release(heap_index, size);
}

bool vulkan_device::is_same(const device::ptr other) {
    const auto other_vulkan_device = std::dynamic_pointer_cast<vulkan_device>(other);
    return other_vulkan_device != nullptr && (_physical_device == other_vulkan_device->_physical_device || _info.is_same_device(other_vulkan_device->_info));
}

void vulkan_device::set_mem_caps(const memory_capabilities& memory_capabilities) {
    _mem_caps = memory_capabilities;
}

}  // namespace vulkan
}  // namespace cldnn
