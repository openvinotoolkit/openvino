// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ze_device.hpp"
#include "ze_common.hpp"

#include <ze_api.h>
#include <vector>
#include <algorithm>
#include <cassert>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <SetupAPI.h>
#include <devguid.h>
#include <cstring>
#else
#include <unistd.h>
#include <limits.h>
#include <link.h>
#include <dlfcn.h>
#endif

namespace cldnn {
namespace ze {

namespace {

bool supports_extension(const std::vector<ze_driver_extension_properties_t>& extensions, const std::string& ext_name, uint32_t ext_ver) {
    return std::find_if(extensions.begin(), extensions.end(), [&ext_name, &ext_ver](const ze_driver_extension_properties_t& ep) {
        return std::string(ep.name) == ext_name && ep.version == ext_ver;
    }) != extensions.end();
}

device_info init_device_info(ze_driver_handle_t driver, ze_device_handle_t device) {
    device_info info;

    uint32_t num_ext = 0;
    ZE_CHECK(zeDriverGetExtensionProperties(driver, &num_ext, nullptr));

    std::vector<ze_driver_extension_properties_t> extensions(num_ext);
    ZE_CHECK(zeDriverGetExtensionProperties(driver, &num_ext, &extensions[0]));

    ze_driver_properties_t driver_properties;
    ZE_CHECK(zeDriverGetProperties(driver, &driver_properties));

    bool supports_luid = supports_extension(extensions, ZE_DEVICE_LUID_EXT_NAME, ZE_DEVICE_LUID_EXT_VERSION_1_0);

    ze_device_ip_version_ext_t ip_version_properties = {ZE_STRUCTURE_TYPE_DEVICE_IP_VERSION_EXT, nullptr, 0};
    ze_device_properties_t device_properties{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2, supports_luid ? &ip_version_properties : nullptr};
    ZE_CHECK(zeDeviceGetProperties(device, &device_properties));

    ze_device_compute_properties_t device_compute_properties{ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES};
    ZE_CHECK(zeDeviceGetComputeProperties(device, &device_compute_properties));

    uint32_t queue_properties_count = 0;
    ZE_CHECK(zeDeviceGetCommandQueueGroupProperties(device, &queue_properties_count, nullptr));

    std::vector<ze_command_queue_group_properties_t> queue_properties(queue_properties_count);
    for (auto& mp : queue_properties) {
        mp.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
    }

    ZE_CHECK(zeDeviceGetCommandQueueGroupProperties(device, &queue_properties_count, &queue_properties[0]));

    auto compute_queue_props = std::find_if(queue_properties.begin(), queue_properties.end(), [](const ze_command_queue_group_properties_t& qp) {
        return (qp.flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) == true;
    });

    OPENVINO_ASSERT(compute_queue_props != queue_properties.end());

    uint32_t memory_properties_count = 0;
    ZE_CHECK(zeDeviceGetMemoryProperties(device, &memory_properties_count, nullptr));

    std::vector<ze_device_memory_properties_t> device_memory_properties(memory_properties_count);
    for (auto& mp : device_memory_properties) {
        mp.stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
    }
    ZE_CHECK(zeDeviceGetMemoryProperties(device, &memory_properties_count, &device_memory_properties[0]));

    ze_device_memory_access_properties_t device_memory_access_properties{ZE_STRUCTURE_TYPE_DEVICE_MEMORY_ACCESS_PROPERTIES};
    ZE_CHECK(zeDeviceGetMemoryAccessProperties(device, &device_memory_access_properties));

    auto ddr_properies = std::find_if(device_memory_properties.begin(), device_memory_properties.end(), [](const ze_device_memory_properties_t& p) {
        return std::string(p.name) == "DDR";
    });

    ze_device_module_properties_t device_module_properties{ZE_STRUCTURE_TYPE_DEVICE_MODULE_PROPERTIES};
    ZE_CHECK(zeDeviceGetModuleProperties(device, &device_module_properties));

    ze_device_image_properties_t device_image_properties{ZE_STRUCTURE_TYPE_DEVICE_IMAGE_PROPERTIES};
    ZE_CHECK(zeDeviceGetImageProperties(device, &device_image_properties));

    info.vendor_id = device_properties.vendorId;
    info.dev_name = device_properties.name;
    info.driver_version = std::to_string(driver_properties.driverVersion);
    info.dev_type = (device_properties.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED) ? device_type::integrated_gpu : device_type::discrete_gpu;

    info.execution_units_count = device_properties.numEUsPerSubslice * device_properties.numSubslicesPerSlice * device_properties.numSlices;

    info.gpu_frequency = device_properties.coreClockRate;

    info.supported_simd_sizes = {};
    info.has_separate_cache = true;

    info.max_work_group_size = device_compute_properties.maxTotalGroupSize;
    info.max_local_mem_size = device_compute_properties.maxSharedLocalMemory;

    if (ddr_properies != device_memory_properties.end())
        info.max_global_mem_size = ddr_properies->totalSize;
    else
        info.max_global_mem_size = 0;

    info.max_alloc_mem_size = device_properties.maxMemAllocSize;

    info.supports_image = device_image_properties.maxSamplers > 0;
    info.supports_intel_planar_yuv = false;
    info.max_image2d_width = device_image_properties.maxImageDims2D;
    info.max_image2d_height = device_image_properties.maxImageDims2D;

    info.supports_fp16 = (device_module_properties.flags & ZE_DEVICE_MODULE_FLAG_FP16) != 0;
    info.supports_fp64 = (device_module_properties.flags & ZE_DEVICE_MODULE_FLAG_FP64) != 0;
    info.supports_fp16_denorms = info.supports_fp16 && (device_module_properties.fp16flags & ZE_DEVICE_FP_FLAG_DENORM) != 0;

    info.supports_khr_subgroups = true;
    info.supports_intel_subgroups = true;
    info.supports_intel_subgroups_short = true;
    info.supports_intel_subgroups_char = true;
    info.supports_intel_required_subgroup_size = true;

    info.supports_imad = (device_module_properties.flags & ZE_DEVICE_MODULE_FLAG_DP4A) != 0;
    info.supports_immad = false; // FIXME

    info.supports_usm = device_memory_access_properties.hostAllocCapabilities && device_memory_access_properties.deviceAllocCapabilities;

    info.supports_local_block_io = true;

    info.gfx_ver = {0, 0, 0}; // could find how to retrieve this from L0 so far
    info.arch = gpu_arch::unknown;
    info.ip_version = ip_version_properties.ipVersion;

    info.device_id = device_properties.deviceId;
    info.num_slices = device_properties.numSlices;
    info.num_sub_slices_per_slice = device_properties.numSubslicesPerSlice;
    info.num_eus_per_sub_slice = device_properties.numEUsPerSubslice;
    info.num_threads_per_eu = device_properties.numThreadsPerEU;

    info.num_ccs = compute_queue_props->numQueues;
    info.supports_queue_families = true;

    info.kernel_timestamp_valid_bits  = device_properties.kernelTimestampValidBits;
    info.timer_resolution  = device_properties.timerResolution;

    static_assert(ZE_MAX_DEVICE_UUID_SIZE == ov::device::UUID::MAX_UUID_SIZE, "");
    static_assert(ZE_MAX_DEVICE_LUID_SIZE_EXT == ov::device::LUID::MAX_LUID_SIZE, "");
    std::copy_n(&device_properties.uuid.id[0], ZE_MAX_DEVICE_UUID_SIZE, info.uuid.uuid.begin());

    if (supports_luid) {
        ze_device_luid_ext_properties_t luid_props{ZE_STRUCTURE_TYPE_DEVICE_LUID_EXT_PROPERTIES, nullptr};
        ze_device_properties_t device_properties{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES, &luid_props};
        if (zeDeviceGetProperties(device, &device_properties) == ZE_RESULT_SUCCESS)
            std::copy_n(&luid_props.luid.id[0], ZE_MAX_DEVICE_LUID_SIZE_EXT, info.luid.luid.begin());
    }

    info.supports_mutable_command_list = supports_extension(extensions, ZE_MUTABLE_COMMAND_LIST_EXP_NAME, ZE_MUTABLE_COMMAND_LIST_EXP_VERSION_1_0);
    return info;
}

memory_capabilities init_memory_caps(ze_device_handle_t device, const device_info& info) {
    std::vector<allocation_type> memory_caps;

    ze_device_memory_access_properties_t device_memory_access_properties{ZE_STRUCTURE_TYPE_DEVICE_MEMORY_ACCESS_PROPERTIES};
    ZE_CHECK(zeDeviceGetMemoryAccessProperties(device, &device_memory_access_properties));

    if (info.supports_usm) {
        if (device_memory_access_properties.hostAllocCapabilities) {
            memory_caps.push_back(allocation_type::usm_host);
        }
        if (device_memory_access_properties.sharedSingleDeviceAllocCapabilities) {
            memory_caps.push_back(allocation_type::usm_shared);
        }
        if (device_memory_access_properties.deviceAllocCapabilities) {
            memory_caps.push_back(allocation_type::usm_device);
        }
    }

    return memory_capabilities(memory_caps);
}

}  // namespace


ze_device::ze_device(ze_driver_handle_t driver, ze_device_handle_t device)
: _driver(driver)
, _device(device)
, _info(init_device_info(driver, device))
, _mem_caps(init_memory_caps(device, _info)) {
    ze_context_desc_t context_desc = { ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0 };
    ZE_CHECK(zeContextCreate(driver, &context_desc, &_context));
}

bool ze_device::is_same(const device::ptr other) {
    auto casted = downcast<ze_device>(other.get());
    if (!casted)
        return false;

    return _context == casted->get_context() && _device == casted->get_device() && _driver == casted->get_driver();
}

ze_device::~ze_device() {
    zeContextDestroy(_context);
}

}  // namespace ze
}  // namespace cldnn
