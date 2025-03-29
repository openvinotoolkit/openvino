/*******************************************************************************
* Copyright (c) 2022-2023 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/// @file
/// C++ API

#pragma once

#include "group/gemm/api.hpp"
#include "group/gemm/compute_policy.hpp"

namespace gpu::xetla::group {
namespace detail {

template <typename dtype_a, typename dtype_b, int alignment_a, int alignment_b,
        gpu_arch arch_tag>
class check_2d_block_pitch_alignment {
    using load_store_attr = typename arch_attr_t<
            arch_tag>::template load_store_attr<msg_type::block_2d>;
    static constexpr int alignment_bytes = load_store_attr::alignment_in_bytes;
    static constexpr int alignment_bytes_a = alignment_a * sizeof(dtype_a);
    static constexpr int alignment_bytes_b = alignment_b * sizeof(dtype_b);

public:
    static constexpr bool value = (alignment_bytes_a % alignment_bytes == 0)
            && (alignment_bytes_b % alignment_bytes == 0);
};

} // namespace detail

/// @addtogroup xetla_gemm
/// @{

/// @brief Selects 2d block && xmx based gemm.
template <typename dtype_a, typename dtype_b, mem_layout mem_layout_a,
        mem_layout mem_layout_b, mem_space mem_space_a, mem_space mem_space_b,
        int alignment_a, int alignment_b, typename dtype_acc,
        typename tile_shape, int k_stride, gpu_arch arch_tag, int stages,
        int sync_freq>
class gemm_selector_t<dtype_a, dtype_b, mem_layout_a, mem_layout_b, mem_space_a,
        mem_space_b, alignment_a, alignment_b, dtype_acc, tile_shape, k_stride,
        mma_engine::xmx, arch_tag, stages, sync_freq,
        std::enable_if_t<detail::check_2d_block_pitch_alignment<dtype_a,
                dtype_b, alignment_a, alignment_b, arch_tag>::value>> {
    using mem_desc_a
            = mem_desc_t<dtype_a, mem_layout_a, mem_space_a, alignment_a>;
    using mem_desc_b
            = mem_desc_t<dtype_b, mem_layout_b, mem_space_b, alignment_b>;
    using compute_attr = compute_attr_t<dtype_a, dtype_b, dtype_acc>;
    using perf_tuning_knob = perf_tuning_knob_t<k_stride, stages, sync_freq>;
    using compute_policy = compute_policy_default_xmx<compute_attr,
            perf_tuning_knob, arch_tag>;
    using pre_processing = pre_processing_default_t<tile_shape, arch_tag>;

public:
    using gemm = gemm_t<compute_policy, tile_shape, mem_desc_a, mem_desc_b,
            pre_processing>;
};

/// @} xetla_gemm
} // namespace gpu::xetla::group
