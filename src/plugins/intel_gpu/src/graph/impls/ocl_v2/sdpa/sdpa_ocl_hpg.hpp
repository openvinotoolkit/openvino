// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// The host-side view of the sdpa_ocl tilings (Xe2 and the xe_hpg SG8 ones) and their device limits, without
// kernel_impl_params. Light enough for the unit tests to include.

#pragma once

#include <cstddef>
#include <cstdint>

#include "intel_gpu/runtime/device_info.hpp"

namespace ov::intel_gpu::ocl {

// What a tiling costs on a device: the host-side view of the kernel's local memory and workgroup size.
struct SDPAOclTilingInfo {
    int subgroup_size = 0;
    int sg_per_wg = 0;
    int wg_size = 0;  // sg_per_wg * subgroup_size work-items
    int kq_sg_tile_keys = 0;
    int kq_sg_tile_queries = 0;
    int kq_sg_per_wg_keys = 0;
    int kq_sg_per_wg_queries = 0;
    int sv_sg_tile_values = 0;
    int sv_sg_tile_scores = 0;
    int sv_sg_per_wg_values = 0;
    int sv_sg_per_wg_scores = 0;
    int kq_wg_tile_keys = 0;
    int kq_wg_tile_queries = 0;
    size_t slm_bytes = 0;
};

// The tiling sdpa_ocl uses for these head sizes on `arch`, with the SDPA_OCL_KQ_* overrides applied like choose_config() does;
// false when there is none or it does not fit the device (local memory, workgroup size). Never throws.
bool sdpa_ocl_describe_tiling(cldnn::gpu_arch arch, size_t k_head_size, size_t v_head_size, SDPAOclTilingInfo& info);

// Local memory (bytes) and work-group size sdpa_ocl plans for, per arch (not the queried device, so a dump with a forged arch
// is judged by the arch it stands for).
size_t sdpa_ocl_max_slm_bytes(cldnn::gpu_arch arch);
size_t sdpa_ocl_max_wg_size(cldnn::gpu_arch arch);

}  // namespace ov::intel_gpu::ocl
