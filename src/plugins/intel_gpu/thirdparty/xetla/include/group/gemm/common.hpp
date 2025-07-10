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

#include "common/common.hpp"
#include "subgroup/subgroup.hpp"

namespace gpu::xetla::group {

/// @brief Compute attribute for gemm.
/// @tparam dtype_a_ Is the memory data type of matA.
/// @tparam dtype_b_ Is the memory data type of matB.
/// @tparam dtype_acc_ Is the compute data type.
template <typename dtype_a_, typename dtype_b_, typename dtype_acc_,
        typename dtype_aci_ = void, typename dtype_as_ = void,
        typename dtype_bs_ = void, typename dtype_cs_ = void>
struct compute_attr_t {
    using dtype_a = dtype_a_;
    using dtype_b = dtype_b_;
    using dtype_acc = dtype_acc_;
    using dtype_aci = dtype_aci_;
    using dtype_as = dtype_as_;
    using dtype_bs = dtype_bs_;
    using dtype_cs = dtype_cs_;
};

/// @brief Fine-tune knobs for gemm.
/// @tparam k_stride_ Is the accumulate stride along k-dim.
/// @tparam stages_ Is the prefetch distance.
/// @tparam sync_freq_ Is the group sync frequency.
template <int k_stride_ = 8, int stages_ = 3, int sync_freq_ = 0>
struct perf_tuning_knob_t {
    static constexpr int k_stride = k_stride_;
    static constexpr int stages = stages_;
    static constexpr int sync_freq = sync_freq_;
};

template <typename mem_desc_t_a, typename tile_shape_t,
        typename perf_tuning_knob_>
constexpr bool is_slm_trans_a() {
    return (mem_desc_t_a::layout == mem_layout::col_major)
            //only for 8bit and 16bit
            && (sizeof(typename mem_desc_t_a::dtype) <= sizeof(uint16_t)
                    && (tile_shape_t::sg_tile_size_y % 16 == 0)
                    && (perf_tuning_knob_::is_slm_transa)
                    && (tile_shape_t::wg_size_x > 1));
}

} // namespace gpu::xetla::group
