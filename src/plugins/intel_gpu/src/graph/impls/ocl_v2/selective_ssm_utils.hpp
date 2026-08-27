// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "common_utils/kernel_generator_base.hpp"
#include "intel_gpu/runtime/device_info.hpp"
#include "intel_gpu/runtime/utils.hpp"

namespace ov::intel_gpu::ocl::selective_ssm_utils {

inline constexpr size_t max_head_dim_block = 4;

inline size_t get_lws(const size_t state_size, const cldnn::device_info& info) {
    const size_t limit = std::min<size_t>(32, info.max_work_group_size);
    const size_t target = std::min(std::max<size_t>(state_size, 1), limit);
    size_t lws = 1;
    while (lws * 2 <= target)
        lws *= 2;
    return lws;
}

inline bool requires_global_state(const size_t state_size, const size_t lws, const cldnn::device_info& info) {
    const size_t local_capacity = info.max_local_mem_size / sizeof(float);
    return state_size > std::numeric_limits<uint32_t>::max() || state_size > local_capacity || lws > local_capacity - state_size;
}

inline size_t get_head_dim_block(const size_t head_dim, const size_t state_size, const size_t lws, const cldnn::device_info& info) {
    const size_t local_capacity = info.max_local_mem_size / sizeof(float);
    const size_t state_and_reduction = state_size + lws;
    size_t block = std::min(std::max<size_t>(head_dim, 1), max_head_dim_block);
    while (block > 1 && state_and_reduction > local_capacity / block)
        --block;
    return block;
}

inline size_t get_head_dim_groups(const size_t head_dim, const size_t head_dim_block) {
    return cldnn::ceil_div(head_dim, head_dim_block);
}

inline void set_dispatch_scalars(KernelData& kd, const size_t block, const cldnn::device_info& info) {
    kd.params.scalars.clear();
    cldnn::scalar_desc block_desc;
    block_desc.t = cldnn::scalar_desc::Types::INT32;
    block_desc.v.s32 = static_cast<int32_t>(block);
    kd.params.scalars.push_back(block_desc);

    const bool use_subgroup_reduction = info.dev_type == cldnn::device_type::integrated_gpu || info.gfx_ver.major >= 20;
    cldnn::scalar_desc reduction_desc;
    reduction_desc.t = cldnn::scalar_desc::Types::UINT32;
    reduction_desc.v.u32 = use_subgroup_reduction ? 1 : 0;
    kd.params.scalars.push_back(reduction_desc);
}

}  // namespace ov::intel_gpu::ocl::selective_ssm_utils
