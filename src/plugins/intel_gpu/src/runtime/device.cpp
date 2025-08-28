// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/device.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include <cmath>
#include <limits>
namespace cldnn {


struct DeviceOps {
    DeviceOps(std::string platform,
              std::vector<uint32_t> dev_id_list,
              std::vector<gfx_version> gfx_version_list,
              std::vector<float> ops): platform(platform), dev_id_list(dev_id_list), gfx_version_list(gfx_version_list) {
        auto add_ops_to_map = [](std::map<data_types, float>& out_map, std::vector<float> ops, data_types dt, size_t index) -> void {
            if (!is_zero(ops[index])) {
                out_map[dt] = ops[index];
            }
        };

        add_ops_to_map(ops_mad,  ops, data_types::f64, 0);
        add_ops_to_map(ops_mad,  ops, data_types::f32, 1);
        add_ops_to_map(ops_mad,  ops, data_types::f16, 2);
        add_ops_to_map(ops_dpas, ops, data_types::f16, 3);
        add_ops_to_map(ops_dpas, ops, data_types::i8,  4);
        add_ops_to_map(ops_dpas, ops, data_types::i4,  5);
        add_ops_to_map(ops_dp4a, ops, data_types::f16, 6);
        add_ops_to_map(ops_dp4a, ops, data_types::i8,  7);
    }

    std::string platform;
    std::vector<uint32_t> dev_id_list;
    std::vector<gfx_version> gfx_version_list;
    std::map<data_types, float> ops_mad;
    std::map<data_types, float> ops_dpas;
    std::map<data_types, float> ops_dp4a;

    void print_summary() const {
        printf("platform: %s\n", platform.c_str());

        if (gfx_version_list.size() == 1) {
            printf("gfx_version: %d.%d.%d\n", gfx_version_list[0].major, gfx_version_list[0].minor, gfx_version_list[0].revision);
        } else if (gfx_version_list.size() == 2) {
            printf("gfx_version: %d.%d.%d ~ %d.%d.%d\n",
                    gfx_version_list[0].major, gfx_version_list[0].minor, gfx_version_list[0].revision,
                    gfx_version_list[1].major, gfx_version_list[1].minor, gfx_version_list[1].revision);
        }

        for (const auto& pair : ops_mad) {
            std::cout << "\tops(mad): " << pair.first << ", " << pair.second << std::endl;
        }
        for (const auto& pair : ops_dpas) {
            std::cout << "\tops(dpas): " << pair.first << ", " << pair.second << std::endl;
        }
        for (const auto& pair : ops_dp4a) {
            std::cout << "\tops(dp4a): " << pair.first << ", " << pair.second << std::endl;
        }
    }

    static bool is_zero(float value) {
        return std::abs(value) <= std::numeric_limits<float>::epsilon();
    }

    bool is_support(device_info& info) const {
        if (gfx_version_list.size() == 1) {
            return info.gfx_ver == gfx_version_list[0];
        } else if (gfx_version_list.size() == 2) {
            return info.gfx_ver >= gfx_version_list[0] && info.gfx_ver <= gfx_version_list[1];
        }
        return false;
    }

    float get_ops(device_info& info, data_types dt) const {
        if (info.supports_immad) {
            auto it = ops_dp4a.find(dt);
            if (it != ops_dp4a.end()) {
                return it->second;
            }
        }

        if (info.supports_imad) {
            auto it = ops_dpas.find(dt);
            if (it != ops_dpas.end()) {
                return it->second;
            }
        }

        auto it = ops_mad.find(dt);
        if (it != ops_mad.end()) {
            return it->second;
        } else {
            return 0;
        }
    }
};

const uint8_t MAX_REVISION = 0xff;

const std::vector<DeviceOps> device_ops_table = {
//    | platform | device_id | gfx_ver                                 | MAD                   | DPAS (imad)           | DP4A (immad)  |
//    |          |           |                                         |  fp64 |  fp32 |  fp16 |  fp16 |  int8 |  int4 |  fp16 |  int8 |
    { "Legacy",   {},         { { 0,  0,  0}, {12,  9, MAX_REVISION} }, {     0,     16,     32,      0,      0,      0,      0,      0 } },    // TGL, ADL-S
    { "DG1",      {},         { {12, 10,  0}                         }, {     0,     16,     32,      0,      0,      0,      0,      0 } },
    { "DG2",      {},         { {12, 55,  0}, {12, 57, MAX_REVISION} }, {     0,     16,     32,      0,      0,      0,    128,    256 } },
    { "PVC_XL",   {},         { {12, 60,  0}, {12, 60, 1}            }, {    16,     32,     64,      0,      0,      0,    512,   1024 } },
    { "PVC_XT",   {},         { {12, 60,  3}, {12, 61, 7}            }, {    32,     32,     64,      0,      0,      0,    512,   1024 } },
    { "MTL/ARL",  {},         { {12, 70,  0}, {12, 74, MAX_REVISION} }, {   0.5,     16,     32,    128,    256,      0,      0,     64 } },
    { "ATS",      { 0x020A, 0x0210 },
                              { {20,  1,  0}, {20,  2, MAX_REVISION} }, {     0,     16,     32,    128,    256,    512,      0,      0 } },
    { "BMG",      {},         { {20,  1,  0}, {20,  2, MAX_REVISION} }, {     1,     16,     32,    128,    256,    512,      0,      0 } },
    { "LNL/PTL",  {},         { {20,  4,  0}, {30,  1, MAX_REVISION} }, {     1,     16,     32,    128,    256,    512,      0,      0 } },
};

float device::get_gops(data_types dt) const {
    // WA: The u8 type isn't accounted for in the device_ops_table, since it's the same as i8.
    dt = dt == data_types::u8 ? data_types::i8 : dt;

    auto info = get_info();
    if (info.vendor_id != INTEL_VENDOR_ID) {
        // GOPS calculation is not supported for non Intel GPUs
        return 0.0f;
    }

    auto freqGHz = info.gpu_frequency / 1000.f;
    auto numEUs = info.execution_units_count * (info.arch < gpu_arch::xe2 ? 1 : 2);
    auto opsPerEU = 0.f;

    for (auto& device_ops : device_ops_table) {
        if (device_ops.is_support(info)) {
            opsPerEU = device_ops.get_ops(info, dt);
            break;
        }
    }

    return freqGHz * numEUs * opsPerEU;
}

bool device::use_unified_shared_memory() const {
    GPU_DEBUG_IF(ExecutionConfig::get_disable_usm()) {
        return false;
    }
    if (get_mem_caps().supports_usm()) {
        return true;
    }
    return false;
}

}  // namespace cldnn
