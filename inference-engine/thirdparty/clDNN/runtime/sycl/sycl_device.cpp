// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sycl_device.hpp"

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
namespace sycl {

namespace {
int driver_dev_id() {
    const std::vector<int> unused_ids = {
        0x4905, 0x4906, 0x4907, 0x4908
    };
    std::vector<int> result;

#ifdef _WIN32
    {
        HDEVINFO device_info_set = SetupDiGetClassDevsA(&GUID_DEVCLASS_DISPLAY, NULL, NULL, DIGCF_PRESENT);
        if (device_info_set == INVALID_HANDLE_VALUE)
            return 0;

        SP_DEVINFO_DATA devinfo_data;
        std::memset(&devinfo_data, 0, sizeof(devinfo_data));
        devinfo_data.cbSize = sizeof(devinfo_data);

        for (DWORD dev_idx = 0; SetupDiEnumDeviceInfo(device_info_set, dev_idx, &devinfo_data); dev_idx++) {
            const size_t kBufSize = 512;
            char buf[kBufSize];
            if (!SetupDiGetDeviceInstanceIdA(device_info_set, &devinfo_data, buf, kBufSize, NULL)) {
                continue;
            }

            char* vendor_pos = std::strstr(buf, "VEN_");
            if (vendor_pos != NULL && std::stoi(vendor_pos + 4, NULL, 16) == 0x8086) {
                char* device_pos = strstr(vendor_pos, "DEV_");
                if (device_pos != NULL) {
                    result.push_back(std::stoi(device_pos + 4, NULL, 16));
                }
            }
        }

        if (device_info_set) {
            SetupDiDestroyDeviceInfoList(device_info_set);
        }
    }
#elif defined(__linux__)
    {
        std::string dev_base{ "/sys/devices/pci0000:00/0000:00:02.0/" };
        std::ifstream ifs(dev_base + "vendor");
        if (ifs.good()) {
            int ven_id;
            ifs >> std::hex >> ven_id;
            ifs.close();
            if (ven_id == 0x8086) {
                ifs.open(dev_base + "device");
                if (ifs.good()) {
                    int res = 0;
                    ifs >> std::hex >> res;
                    result.push_back(res);
                }
            }
        }
    }
#endif

    auto id_itr = result.begin();
    while (id_itr != result.end()) {
        if (std::find(unused_ids.begin(), unused_ids.end(), *id_itr) != unused_ids.end())
            id_itr = result.erase(id_itr);
        else
            id_itr++;
    }

    if (result.empty())
        return 0;
    else
        return result.back();
}

device_type get_device_type(const cl::sycl::device& device) {
    auto unified_mem = device.get_info<cl::sycl::info::device::host_unified_memory>();

    return unified_mem ? device_type::integrated_gpu : device_type::discrete_gpu;
}

bool get_imad_support(const cl::sycl::device& device) {
    std::string dev_name = device.get_info<cl::sycl::info::device::name>();

    if (dev_name.find("Gen12") != std::string::npos ||
        dev_name.find("Xe") != std::string::npos)
        return true;

    if (get_device_type(device) == device_type::integrated_gpu) {
        const std::vector<int> imad_ids = {
            0x9A40, 0x9A49, 0x9A59, 0x9AD9,
            0x9A60, 0x9A68, 0x9A70, 0x9A78,
            0x9A7F, 0x9AF8, 0x9AC0, 0x9AC9
        };
        int dev_id = driver_dev_id();
        if (dev_id == 0)
            return false;

        if (std::find(imad_ids.begin(), imad_ids.end(), dev_id) != imad_ids.end())
            return true;
    } else {
        return true;
    }

    return false;
}

device_info init_device_info(const cl::sycl::device& device) {
    device_info info;
    auto platform = device.get_platform();
    info.vendor_id = static_cast<uint32_t>(device.get_info<cl::sycl::info::device::vendor_id>());
    info.dev_name = device.get_info<cl::sycl::info::device::name>() + " " + platform.get_info<cl::sycl::info::platform::name>();
    info.driver_version = device.get_info<cl::sycl::info::device::driver_version>();
    info.dev_type = get_device_type(device);

    info.execution_units_count = device.get_info<cl::sycl::info::device::max_compute_units>();

    info.gpu_frequency = static_cast<uint32_t>(device.get_info<cl::sycl::info::device::max_clock_frequency>());

    info.max_work_group_size = static_cast<uint64_t>(device.get_info<cl::sycl::info::device::max_work_group_size>());

    info.max_local_mem_size = static_cast<uint64_t>(device.get_info<cl::sycl::info::device::local_mem_size>());
    info.max_global_mem_size = static_cast<uint64_t>(device.get_info<cl::sycl::info::device::global_mem_size>());
    info.max_alloc_mem_size = static_cast<uint64_t>(device.get_info<cl::sycl::info::device::max_mem_alloc_size>());

    info.supports_image = static_cast<uint8_t>(device.get_info<cl::sycl::info::device::image_support>());
    info.max_image2d_width = static_cast<uint64_t>(device.get_info<cl::sycl::info::device::image2d_max_width>());
    info.max_image2d_height = static_cast<uint64_t>(device.get_info<cl::sycl::info::device::image2d_max_height>());

    auto extensions = device.get_info<cl::sycl::info::device::extensions>();
    info.supports_fp16 = std::find(extensions.begin(), extensions.end(), "cl_khr_fp16") != extensions.end();
    info.supports_fp64 = std::find(extensions.begin(), extensions.end(), "cl_khr_fp64") != extensions.end();
    std::vector<cl::sycl::info::fp_config> half_config = device.get_info<cl::sycl::info::device::half_fp_config>();
    info.supports_fp16_denorms = info.supports_fp16 &&
                                 std::find(half_config.begin(), half_config.end(), cl::sycl::info::fp_config::denorm) != half_config.end();

    info.supports_subgroups = std::find(extensions.begin(), extensions.end(), "cl_intel_subgroups") != extensions.end();
    info.supports_subgroups_short = std::find(extensions.begin(), extensions.end(), "cl_intel_subgroups_short") != extensions.end();
    info.supports_subgroups_char = std::find(extensions.begin(), extensions.end(), "cl_intel_subgroups_char") != extensions.end();

    info.supports_imad = get_imad_support(device);
    info.supports_immad = false;

    info.max_threads_per_execution_unit = 7;
    info.max_threads_per_device = static_cast<uint32_t>(info.execution_units_count * info.max_threads_per_execution_unit);

    info.supports_usm = false;

    info.supports_local_block_io = std::find(extensions.begin(), extensions.end(), "cl_intel_subgroup_local_block_io") != extensions.end();

    return info;
}

memory_capabilities init_memory_caps(const cl::sycl::device& device, const device_info& info) {
    std::vector<allocation_type> memory_caps;
    if (info.supports_usm) {
        if (device.get_info<cl::sycl::info::device::usm_host_allocations>()) {
            memory_caps.push_back(allocation_type::usm_host);
        }
        // Do we need resticted shared allocations?
        if (device.get_info<cl::sycl::info::device::usm_shared_allocations>()) {
            memory_caps.push_back(allocation_type::usm_shared);
        }
        if (device.get_info<cl::sycl::info::device::usm_device_allocations>()) {
            memory_caps.push_back(allocation_type::usm_device);
        }
    }

    return memory_capabilities(memory_caps);
}

}  // namespace


sycl_device::sycl_device(const cl::sycl::device& dev, const cl::sycl::context& ctx)
: _context(ctx)
, _device(dev)
, _info(init_device_info(dev))
, _mem_caps(init_memory_caps(dev, _info)) { }

}  // namespace sycl
}  // namespace cldnn
