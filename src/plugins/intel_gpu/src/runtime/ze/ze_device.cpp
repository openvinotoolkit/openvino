// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ze_device.hpp"
#include "ze_common.hpp"

#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <string>
#include <cassert>
#include <time.h>
#include <limits>
#include <chrono>
#include <fstream>
#include <iostream>
#include <utility>

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

gfx_version parse_version(uint32_t ver) {
    uint16_t major = ver >> 16;
    uint8_t minor = (ver >> 8) & 0xFF;
    uint8_t revision = ver & 0xFF;

    return {major, minor, revision};
}

device_info init_device_info(ze_driver_handle_t driver, ze_device_handle_t device) {
    device_info info;

    ze_driver_properties_t driver_properties;
    ZE_CHECK(zeDriverGetProperties(driver, &driver_properties));

    ze_device_properties_t device_properties;
    ZE_CHECK(zeDeviceGetProperties(device, &device_properties));

    ze_device_compute_properties_t device_compute_properties;
    ZE_CHECK(zeDeviceGetComputeProperties(device, &device_compute_properties));

    uint32_t memory_properties_count = 0;
    ZE_CHECK(zeDeviceGetMemoryProperties(device, &memory_properties_count, nullptr));

    std::vector<ze_device_memory_properties_t> device_memory_properties(memory_properties_count);
    ZE_CHECK(zeDeviceGetMemoryProperties(device, &memory_properties_count, &device_memory_properties[0]));

    uint32_t extension_properties_count = 0;
    ZE_CHECK(zeDriverGetExtensionProperties(driver, &extension_properties_count, nullptr));

    std::vector<ze_driver_extension_properties_t> driver_extension_properties(extension_properties_count);
    ZE_CHECK(zeDriverGetExtensionProperties(driver, &extension_properties_count, &driver_extension_properties[0]));

    ze_device_memory_access_properties_t device_memory_access_properties;
    ZE_CHECK(zeDeviceGetMemoryAccessProperties(device, &device_memory_access_properties));

    auto ddr_properies = std::find_if(device_memory_properties.begin(), device_memory_properties.end(), [](const ze_device_memory_properties_t& p) {
        return std::string(p.name) == "DDR";
    });

    //ze_device_module_properties_t device_module_properties;
    //ZE_CHECK(zeDeviceGetModuleProperties(device, &device_module_properties));

    ze_device_image_properties_t device_image_properties;
    ZE_CHECK(zeDeviceGetImageProperties(device, &device_image_properties));

    info.vendor_id = device_properties.vendorId;
    info.dev_name = device_properties.name;
    info.driver_version = std::to_string(driver_properties.driverVersion);
    info.dev_type = (device_properties.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED) ? device_type::integrated_gpu : device_type::discrete_gpu;

    info.execution_units_count = device_properties.numEUsPerSubslice * device_properties.numSubslicesPerSlice * device_properties.numSlices;

    info.gpu_frequency = device_properties.coreClockRate;

    info.max_work_group_size = device_compute_properties.maxTotalGroupSize;
    info.max_local_mem_size = device_compute_properties.maxSharedLocalMemory;

    if (ddr_properies != device_memory_properties.end())
        info.max_global_mem_size = ddr_properies->totalSize;
    else
        info.max_global_mem_size = 0;

    info.max_alloc_mem_size = device_properties.maxMemAllocSize;

    // TODO: check if any better propery exists
    // info.supports_image = device_image_properties.maxSamplers > 0;
    // info.max_image2d_width = ??;
    // info.max_image2d_height = ??;

    info.supports_fp16 = true;//(device_module_properties.flags & ZE_DEVICE_MODULE_FLAG_FP16) != 0;
    info.supports_fp64 = true;//(device_module_properties.flags & ZE_DEVICE_MODULE_FLAG_FP64) != 0;
    info.supports_fp16_denorms = true;//info.supports_fp16 && (device_module_properties.fp16flags & ZE_DEVICE_FP_FLAG_DENORM) != 0;

    info.supports_subgroups = true;
    info.supports_subgroups_short = true;
    info.supports_subgroups_char = true;

    info.supports_imad = false;//(device_module_properties.flags & ZE_DEVICE_MODULE_FLAG_DP4A) != 0;;
    info.supports_immad = false;

    info.num_threads_per_eu = device_properties.numThreadsPerEU;

    info.supports_usm = device_memory_access_properties.hostAllocCapabilities && device_memory_access_properties.deviceAllocCapabilities;

    info.supports_local_block_io = true;

    info.gfx_ver = {0, 0, 0}; // could find how to retrieve this from L0 so far
    info.device_id = device_properties.deviceId;
    info.num_slices = device_properties.numSlices;
    info.num_sub_slices_per_slice = device_properties.numSubslicesPerSlice;
    info.num_eus_per_sub_slice = device_properties.numEUsPerSubslice;
    info.num_threads_per_eu = device_properties.numThreadsPerEU;

    return info;
}

memory_capabilities init_memory_caps(ze_device_handle_t device, const device_info& info) {
    std::vector<allocation_type> memory_caps;

    ze_device_memory_access_properties_t device_memory_access_properties;
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
