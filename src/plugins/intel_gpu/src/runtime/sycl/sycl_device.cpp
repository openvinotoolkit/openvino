// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef ENABLE_ONEDNN_FOR_GPU
#ifndef NOMINMAX
# define NOMINMAX
#endif
#include "gpu/intel/jit/generator.hpp"
#endif  // ENABLE_ONEDNN_FOR_GPU

#include "sycl_device.hpp"
#include "sycl_common.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"

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
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <setupapi.h>
#include <devguid.h>
#include <cstring>
#else
#include <unistd.h>
#include <limits.h>
#include <link.h>
#include <dlfcn.h>
#endif

#if SYCL_EXT_ONEAPI_MATRIX
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#endif // SYCL_EXT_ONEAPI_MATRIX

namespace cldnn {
namespace sycl {

namespace {

#ifdef ENABLE_ONEDNN_FOR_GPU
gpu_arch convert_ngen_arch(ngen::HW gpu_arch) {
    switch (gpu_arch) {
        case ngen::HW::Gen9: return gpu_arch::gen9;
        case ngen::HW::Gen11: return gpu_arch::gen11;
        case ngen::HW::XeLP: return gpu_arch::xe_lp;
        case ngen::HW::XeHP: return gpu_arch::xe_hp;
        case ngen::HW::XeHPG: return gpu_arch::xe_hpg;
        case ngen::HW::XeHPC: return gpu_arch::xe_hpc;
        case ngen::HW::Xe2: return gpu_arch::xe2;
        case ngen::HW::Xe3: return gpu_arch::xe3;
        case ngen::HW::Gen10:
        case ngen::HW::Unknown: return gpu_arch::unknown;
    }
    return gpu_arch::unknown;
}
#endif

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

device_type get_device_type(const ::sycl::device& device) {
    auto unified_mem = device.has(::sycl::aspect::usm_host_allocations);

    return unified_mem ? device_type::integrated_gpu : device_type::discrete_gpu;
}

bool get_imad_support(const ::sycl::device& device) {
    std::string dev_name = device.get_info<::sycl::info::device::name>();

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

device_info init_device_info(const ::sycl::device& device, const ::sycl::context& context) {
    device_info info = {};
    info.vendor_id = device.get_info<::sycl::info::device::vendor_id>();
    info.dev_name = device.get_info<::sycl::info::device::name>();
    info.driver_version = device.get_info<::sycl::info::device::driver_version>();
    info.dev_type = get_device_type(device);

    info.execution_units_count = device.get_info<::sycl::info::device::max_compute_units>();

    info.gpu_frequency = device.get_info<::sycl::info::device::max_clock_frequency>();

    info.max_work_group_size = static_cast<uint64_t>(device.get_info<::sycl::info::device::max_work_group_size>());

    // For some reason nvidia runtime throws an exception (CL_INVALID_KERNEL_ARGS) for WG as follows:
    // global: < 1 x 32 x 5184 >
    // local: < 1 x 1 x 576 >
    // While local  < 1 x 1 x 36 > works fine
    // So below we limit max WG size by 64 which was selected based on few experiments.
    if (info.vendor_id == NVIDIA_VENDOR_ID) {
        info.max_work_group_size = 64;
    }

    info.max_local_mem_size = device.get_info<::sycl::info::device::local_mem_size>();
    info.max_global_mem_size = device.get_info<::sycl::info::device::global_mem_size>();
    info.max_alloc_mem_size = device.get_info<::sycl::info::device::max_mem_alloc_size>();
    info.max_global_cache_size = device.get_info<::sycl::info::device::global_mem_cache_size>();

    info.supports_image = device.has(::sycl::aspect::ext_intel_legacy_image);
    info.max_image2d_width = static_cast<uint64_t>(device.get_info<::sycl::info::device::image2d_max_width>());
    info.max_image2d_height = static_cast<uint64_t>(device.get_info<::sycl::info::device::image2d_max_height>());

    info.supports_intel_planar_yuv = false; // no corresponding aspect in SYCL
    info.supports_fp16 = device.has(::sycl::aspect::fp16);
    info.supports_fp64 = device.has(::sycl::aspect::fp64);
    if (info.supports_fp16) {
        auto half_fp_config = device.get_info<::sycl::info::device::half_fp_config>();
        info.supports_fp16_denorms = std::find(half_fp_config.begin(), half_fp_config.end(),
                                               ::sycl::info::fp_config::denorm) != half_fp_config.end();
    } else {
        info.supports_fp16_denorms = false;
    }

    info.supports_khr_subgroups = true; // SYCL 2020 requires subgroups support
    info.supports_intel_subgroups = false; // no corresponding aspect in SYCL
    info.supports_intel_subgroups_short = false; // no corresponding aspect in SYCL
    info.supports_intel_subgroups_char = false; // no corresponding aspect in SYCL
    info.supports_intel_required_subgroup_size = true; // to use sub_group_sizes as SIMD sizes

    info.supports_imad = get_imad_support(device);
    info.supports_immad = false;

    info.supports_usm = device.has(::sycl::aspect::usm_host_allocations) ||
                        device.has(::sycl::aspect::usm_shared_allocations) ||
                        device.has(::sycl::aspect::usm_device_allocations);

    info.supports_usm = false;  // currently USM is disabled for SYCL backend

    info.supports_queue_families = false; // no corresponding aspect in SYCL

    if (info.supports_intel_required_subgroup_size) {
        info.supported_simd_sizes = device.get_info<::sycl::info::device::sub_group_sizes>();
    } else {
        // Set these values as reasonable default for most of the supported platforms
        info.supported_simd_sizes = {8, 16, 32};
    }

    bool device_uuid_supported = device.has(::sycl::aspect::ext_intel_device_info_uuid);
    if (device_uuid_supported) {
        auto uuid = device.get_info<::sycl::ext::intel::info::device::uuid>();
        static_assert(uuid.size() == ov::device::UUID::MAX_UUID_SIZE, "");
        info.uuid.uuid = uuid;
    } else {
        std::fill_n(std::begin(info.uuid.uuid), ov::device::UUID::MAX_UUID_SIZE, 0);
    }

    // sycl::aspect::ext_intel_device_info_luid is described in
    // https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_intel_device_info.md
    // DPC++ 6.2.0 and later will support this aspect.
    // if (device.has(::sycl::aspect::ext_intel_device_info_luid)) {
    //     auto luid = device.get_info<::sycl::ext::intel::info::device::luid>();
    //     static_assert(luid.size() == ov::device::LUID::MAX_LUID_SIZE, "");
    //     info.luid.luid = luid;
    // } else {
        std::fill_n(std::begin(info.luid.luid), ov::device::LUID::MAX_LUID_SIZE, 0);
    // }

    info.has_separate_cache = false;
    if (info.vendor_id == INTEL_VENDOR_ID) {
        info.ip_version = 0; // no corresponding info in SYCL
        info.gfx_ver = {0, 0, 0}; // no corresponding info in SYCL
        if (device.has(::sycl::aspect::ext_intel_device_id)) {
            info.device_id = device.get_info<::sycl::ext::intel::info::device::device_id>();
        }
        if (device.has(::sycl::aspect::ext_intel_gpu_slices)) {
            info.num_slices = device.get_info<::sycl::ext::intel::info::device::gpu_slices>();
        }
        if (device.has(::sycl::aspect::ext_intel_gpu_subslices_per_slice)) {
            info.num_sub_slices_per_slice = device.get_info<::sycl::ext::intel::info::device::gpu_subslices_per_slice>();
        }
        if (device.has(::sycl::aspect::ext_intel_gpu_eu_count_per_subslice)) {
            info.num_eus_per_sub_slice = device.get_info<::sycl::ext::intel::info::device::gpu_eu_count_per_subslice>();
        }
        if (device.has(::sycl::aspect::ext_intel_gpu_hw_threads_per_eu)) {
            info.num_threads_per_eu = device.get_info<::sycl::ext::intel::info::device::gpu_hw_threads_per_eu>();
        }

        info.supports_imad = info.supports_imad;
#if SYCL_EXT_ONEAPI_MATRIX
        auto matrix_combinations = device.get_info<::sycl::ext::oneapi::experimental::info::device::matrix_combinations>();
        info.supports_immad = matrix_combinations.size() > 0;
#endif // SYCL_EXT_ONEAPI_MATRIX
        info.has_separate_cache = false; // no corresponding info in SYCL
    } else if (info.vendor_id == NVIDIA_VENDOR_ID) {
        info.gfx_ver = {0, 0, 0}; // no corresponding info in SYCL
    } else {
        info.gfx_ver = {0, 0, 0};
        info.device_id = driver_dev_id();
        info.num_slices = 0;
        info.num_sub_slices_per_slice = 0;
        info.num_eus_per_sub_slice = 0;
        info.num_threads_per_eu = 0;
    }

    info.num_ccs = 1;


#ifdef ENABLE_ONEDNN_FOR_GPU
    using namespace dnnl::impl::gpu::intel::jit;
    ngen::Product product = generator_t<ngen::HW::Unknown>::detectHWInfo(context, device);
    info.arch = convert_ngen_arch(ngen::getCore(product.family));
    // We change the value of this flag to avoid OneDNN usage for the platforms unknown to OneDNN
    // This is required to guarantee some level of forward compatibility for the new HW generations
    // as OneDNN code generators are not generic and typically requires some updates for the new architectures
    // Ideally, we shouldn't do that as OCL impls sometimes also check this flag, but in order to avoid that
    // we need to ensure that graph transformations are not relying on this flag as indicator that onednn will be used
    if (product.family == ngen::ProductFamily::Unknown) {
        info.supports_immad = false;
    }
#else  // ENABLE_ONEDNN_FOR_GPU
    info.arch = gpu_arch::unknown;
#endif  // ENABLE_ONEDNN_FOR_GPU

    return info;
}

memory_capabilities init_memory_caps(const ::sycl::device& device, const device_info& info) {
    std::vector<allocation_type> memory_caps;
    if (info.supports_usm) {
        if (device.has(::sycl::aspect::usm_host_allocations)) {
            memory_caps.push_back(allocation_type::usm_host);
        }
        if (device.has(::sycl::aspect::usm_shared_allocations)) {
            memory_caps.push_back(allocation_type::usm_shared);
        }
        if (device.has(::sycl::aspect::usm_device_allocations)) {
            memory_caps.push_back(allocation_type::usm_device);
        }
    }

    return memory_capabilities(memory_caps);
}

}  // namespace


sycl_device::sycl_device(const ::sycl::device dev, const ::sycl::context& ctx, const ::sycl::platform& platform)
: _context(ctx)
, _device(dev)
, _platform(platform)
, _info(init_device_info(dev, ctx))
, _mem_caps(init_memory_caps(dev, _info))
, _is_initialized(true){
// , _usm_helper(new cl::UsmHelper(_context, _device, use_unified_shared_memory()))
}

bool sycl_device::is_same(const device::ptr other) {
    auto casted = downcast<sycl_device>(other.get());
    if (!casted)
        return false;

    return _device == casted->get_device() && _platform == casted->get_platform();
}

void sycl_device::set_mem_caps(const memory_capabilities& memory_capabilities) {
    _mem_caps = memory_capabilities;
}

void sycl_device::initialize() {
    if (_is_initialized)
        return;
}

}  // namespace sycl
}  // namespace cldnn
