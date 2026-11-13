// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "intel_gpu/runtime/device_info.hpp"

namespace ov::intel_gpu::ocl::paged_gated_delta_net_jit {

inline size_t get_subgroup_size(cldnn::gpu_arch arch) {
    switch (arch) {
    case cldnn::gpu_arch::gen9:
    case cldnn::gpu_arch::gen11:
    case cldnn::gpu_arch::xe_lp:
    case cldnn::gpu_arch::xe_hp:
    case cldnn::gpu_arch::xe_hpg:
        return 8;
    case cldnn::gpu_arch::xe2:
    case cldnn::gpu_arch::xe3:
    default:
        return 16;
    }
}

// The 256-register mode exists from Xe-HP on. Earlier parts run with 128.
inline bool supports_large_grf(cldnn::gpu_arch arch) {
    switch (arch) {
    case cldnn::gpu_arch::gen9:
    case cldnn::gpu_arch::gen11:
    case cldnn::gpu_arch::xe_lp:
        return false;
    default:
        return true;
    }
}

inline size_t get_grf_bytes(cldnn::gpu_arch arch) {
    switch (arch) {
    case cldnn::gpu_arch::gen9:
    case cldnn::gpu_arch::gen11:
    case cldnn::gpu_arch::xe_lp:
    case cldnn::gpu_arch::xe_hp:
    case cldnn::gpu_arch::xe_hpg:
        return 32;
    default:
        return 64;
    }
}

// Floats per lane in a thread's register file: its size divided by the subgroup
// width.
inline size_t get_lane_float_capacity(cldnn::gpu_arch arch) {
    const size_t grf_regs = supports_large_grf(arch) ? 256 : 128;
    return grf_regs * get_grf_bytes(arch) / (get_subgroup_size(arch) * sizeof(float));
}

// d_v-on-lanes keeps k_head_dims floats of state per lane. It needs the large
// register file and applies while those are at most half of the lane's floats
// in it; above that the k axis goes on the lanes instead.
inline bool uses_d_on_lanes(size_t k_head_dims, cldnn::gpu_arch arch) {
    return !supports_large_grf(arch) || k_head_dims > get_lane_float_capacity(arch) / 2;
}

// v rows a subgroup covers under d-on-lanes, where the count can be chosen
// (under d_v-on-lanes it is the subgroup width).
//
// The floats a lane needs are estimated as the sum of
//   v_rows * lane_elems   for the state,
//   4 * lane_elems        for q and k of this token and the next,
//   32                    a margin, set from spill audits, for everything else.
// v_rows starts at 8, so that few subgroups read the same q and k rows, and is
// halved, down to 1, while the estimate reaches the lane's floats in the
// register file.
inline size_t get_d_on_lanes_v_rows(size_t k_head_dims, cldnn::gpu_arch arch) {
    const size_t lane_elems = k_head_dims / get_subgroup_size(arch);
    const size_t capacity = get_lane_float_capacity(arch);
    size_t v_rows = 8;
    while (v_rows > 1 && (v_rows + 4) * lane_elems + 32 >= capacity) {
        v_rows /= 2;
    }
    return v_rows;
}

// v rows a subgroup covers. Under d_v-on-lanes each lane holds one v row, so
// this is the subgroup width.
inline size_t get_v_rows_per_thread(size_t k_head_dims, cldnn::gpu_arch arch) {
    return uses_d_on_lanes(k_head_dims, arch) ? get_d_on_lanes_v_rows(k_head_dims, arch) : get_subgroup_size(arch);
}

}  // namespace ov::intel_gpu::ocl::paged_gated_delta_net_jit
