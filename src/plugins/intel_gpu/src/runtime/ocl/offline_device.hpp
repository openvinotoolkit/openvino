// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/device.hpp"
#include "intel_gpu/runtime/device_info.hpp"
#include "intel_gpu/runtime/memory_caps.hpp"
#include "openvino/core/except.hpp"

#include <iostream>
#include <string>
#include <vector>

namespace cldnn {
namespace ocl {

// HW-free device holding a fabricated device_info for offline compile-only. No real cl::Device: it is
// never enumerated by device_query and never initialized against hardware. Mirrors the dummy_device
// test helper (device_test.cpp). Used by offline_engine so the compile passes read device metadata
// without a GPU. device_info is NOT serialized into the blob, so the values are hardcoded here.
class offline_device : public device {
public:
    explicit offline_device(const device_info& info)
        : _info(info)
        , _mem_caps(info.supports_usm
                        ? std::vector<allocation_type>{allocation_type::cl_mem, allocation_type::usm_host,
                                                       allocation_type::usm_shared, allocation_type::usm_device}
                        : std::vector<allocation_type>{allocation_type::cl_mem}) {}

    const device_info& get_info() const override { return _info; }
    memory_capabilities get_mem_caps() const override { return _mem_caps; }
    void initialize() override {}
    bool is_initialized() const override { return true; }
    bool is_same(const device::ptr other) override { return this == other.get(); }
    void set_mem_caps(const memory_capabilities& memory_capabilities) override { _mem_caps = memory_capabilities; }

private:
    device_info _info;
    memory_capabilities _mem_caps;
};

// Returns a fabricated device_info for the offline compile target. The field values are captured once
// from a real device (Intel UHD Graphics 770, 0x4680, driver 32.0.101.6129) and hardcoded, because
// device_info is never serialized. Any device target (from EP metadata) is accepted: the
// 0x4680-derived profile is used as the base and, when device_arg is a 0xXXXX hex id, device_id is
// overridden. For targets other than 0x4680 a one-time warning is logged -- kernel/layout selection
// fidelity is only guaranteed for 0x4680 until per-device profiles are added.
inline device_info make_offline_device_info(const std::string& device_arg) {
    device_info info{};
    // Captured from a real 0x4680 device.
    info.execution_units_count = 32;
    info.gpu_frequency = 1500;
    info.max_work_group_size = 512;
    info.max_local_mem_size = 65536;
    info.max_global_mem_size = 31714639872ULL;
    info.max_alloc_mem_size = 4294959104ULL;
    info.max_global_cache_size = 1966080;
    info.max_image2d_width = 16384;
    info.max_image2d_height = 16384;
    info.supports_fp16 = true;
    info.supports_fp64 = false;
    info.supports_fp16_denorms = true;
    info.supports_khr_subgroups = false;
    info.supports_intel_subgroups = true;
    info.supports_intel_subgroups_short = true;
    info.supports_intel_subgroups_char = true;
    info.supports_intel_required_subgroup_size = true;
    info.supports_queue_families = true;
    info.supports_image = true;
    info.supports_intel_planar_yuv = true;
    info.supports_work_group_collective_functions = true;
    info.supports_non_uniform_work_group = true;
    info.supports_imad = true;
    info.supports_immad = false;
    info.supports_mutable_command_list = false;
    info.supports_usm = true;
    info.has_separate_cache = false;
    info.supports_cp_offload = false;
    info.supports_counter_based_events = false;
    info.supports_leo = false;
    info.supported_simd_sizes = {8, 16, 32};
    info.vendor_id = INTEL_VENDOR_ID;  // 0x8086
    info.dev_name = "Intel(R) UHD Graphics 770";
    info.driver_version = "32.0.101.6129";
    info.dev_type = device_type::integrated_gpu;
    info.gfx_ver = {12, 2, 0};
    info.arch = gpu_arch::xe_lp;  // enum value 3
    info.ip_version = 50364416;
    info.device_id = 0x4680;
    info.num_slices = 1;
    info.num_sub_slices_per_slice = 4;
    info.num_eus_per_sub_slice = 8;
    info.num_threads_per_eu = 7;
    info.num_ccs = 1;
    info.sub_device_idx = 0xFFFFFFFF;  // captured 4294967295 (UINT32_MAX = no sub-device)
    info.pci_info = {0, 0, 2, 0};  // domain, bus, device, function
    info.timer_resolution = 0;
    info.kernel_timestamp_valid_bits = 0;
    info.compute_queue_group_ordinal = 0;
    info.device_memory_ordinal = 0;
    // uuid / luid left default (not used by compile-time passes).

    // Accept any device target. If device_arg is a 0xXXXX hex id, override device_id with it. For any
    // target other than 0x4680, the fields above are a 0x4680-derived approximation: warn once that
    // kernel/layout fidelity is only guaranteed for 0x4680 (add per-device profiles when needed).
    const bool is_4680 = (device_arg == "0x4680" || device_arg == "0X4680");
    if (!device_arg.empty() && (device_arg.rfind("0x", 0) == 0 || device_arg.rfind("0X", 0) == 0)) {
        try {
            info.device_id = static_cast<uint32_t>(std::stoul(device_arg, nullptr, 16));
        } catch (...) {
            // leave the captured default device_id
        }
    }
    if (!is_4680) {
        std::cerr << "[GPU offline] make_offline_device_info: using 0x4680-derived device_info profile "
                     "for target " << device_arg << "; kernel/layout fidelity only guaranteed for 0x4680"
                  << std::endl;
    }
    return info;
}

}  // namespace ocl
}  // namespace cldnn
