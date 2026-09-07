// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstddef>

#include "intel_gpu/runtime/device_info.hpp"
#include "intel_gpu/runtime/utils.hpp"

namespace ov::intel_gpu::ocl::selective_ssm_jit {

enum class device_kind { integrated, discrete };

inline constexpr size_t short_sequence_private_value_budget = 12;
inline constexpr size_t long_sequence_private_value_budget = 24;
inline constexpr size_t paged_private_value_budget = 48;
inline constexpr size_t default_discrete_head_dim_block = 4;
// With block 4, cap the private recurrence state at 32 values on SIMD16 and 16 values on SIMD32.
inline constexpr size_t max_common_discrete_private_state_size = 128;

inline bool supports_simd_size(const cldnn::device_info& info, const size_t simd_size) {
    return std::find(info.supported_simd_sizes.begin(), info.supported_simd_sizes.end(), simd_size) != info.supported_simd_sizes.end();
}

inline size_t get_subgroup_size(const cldnn::device_info& info, const device_kind kind) {
    if ((!info.supports_khr_subgroups && !info.supports_intel_subgroups) || !info.supports_intel_required_subgroup_size)
        return 0;

    const size_t preferred = kind == device_kind::discrete && info.arch >= cldnn::gpu_arch::xe2 ? 32 : 16;
    if (supports_simd_size(info, preferred) && preferred <= info.max_work_group_size)
        return preferred;
    if (supports_simd_size(info, 16) && 16 <= info.max_work_group_size)
        return 16;
    if (supports_simd_size(info, 8) && 8 <= info.max_work_group_size)
        return 8;
    return 0;
}

inline bool supports_common_discrete_private_state(const cldnn::device_info& info, const size_t state_size) {
    const bool has_supported_arch = info.arch == cldnn::gpu_arch::xe_hpg || info.arch >= cldnn::gpu_arch::xe2;
    return has_supported_arch && state_size <= max_common_discrete_private_state_size;
}

inline size_t get_head_dim_block(const size_t head_dim,
                                 const size_t state_size,
                                 const size_t subgroup_size,
                                 const cldnn::device_info& info,
                                 const device_kind kind,
                                 const size_t max_private_values,
                                 const size_t discrete_target = default_discrete_head_dim_block) {
    if (head_dim == 0 || state_size == 0 || subgroup_size == 0)
        return 0;

    const size_t state_iterations = cldnn::ceil_div(state_size, subgroup_size);
    const size_t target = kind == device_kind::discrete ? discrete_target : (info.arch >= cldnn::gpu_arch::xe2 ? 8 : 4);
    size_t block = std::min(head_dim, target);
    if (kind == device_kind::integrated) {
        // Bound the scalarized recurrence state together with the two per-head-dimension temporaries.
        while (block > 1 && block * (state_iterations + 2) > max_private_values)
            --block;
        return block * (state_iterations + 2) <= max_private_values ? block : 0;
    }
    const size_t local_capacity = info.max_local_mem_size / sizeof(float);
    while (block > 1 && block * state_size > local_capacity)
        --block;
    return block * state_size <= local_capacity ? block : 0;
}

inline bool matches_device_kind(const cldnn::device_info& info, const device_kind kind) {
    const auto expected = kind == device_kind::integrated ? cldnn::device_type::integrated_gpu : cldnn::device_type::discrete_gpu;
    return info.dev_type == expected;
}

}  // namespace ov::intel_gpu::ocl::selective_ssm_jit
