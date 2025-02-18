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

#include "group/gemm/common.hpp"

namespace gpu::xetla::group {

/// @addtogroup xetla_gemm
/// @{

/// @brief Gemm default pre_processing functor.
/// @tparam tile_shape Is the workgroup-level tile shape.
/// @tparam arch_tag Is the HW architecture.
template <typename tile_shape, gpu_arch arch_tag, class enable = void>
struct pre_processing_default_t {};

/// @brief Gemm pre_processing functor with applying relu op to matA.
/// @tparam tile_shape Is the workgroup-level tile shape.
/// @tparam arch_tag Is the HW architecture.
template <typename tile_shape, gpu_arch arch_tag, class enable = void>
struct pre_processing_matA_neg_filter_t {};

/// @brief Gemm functor.
/// @tparam compute_policy Is the compute algorithm of gemm implementation.
/// @tparam tile_shape Is the workgroup-level tile shape.
/// @tparam mem_desc_a Is the memory descriptor of matA.
/// @tparam mem_desc_b Is the memory descriptor of matB.
/// @tparam pre_processing Is the pre_processing functor.
template <typename compute_policy, typename tile_shape, typename mem_desc_a,
        typename mem_desc_b,
        typename pre_processing
        = pre_processing_default_t<tile_shape, compute_policy::arch_tag>,
        typename mem_desc_as = void, typename mem_desc_bs = void,
        class enable = void>
class gemm_t {};

/// @brief Gemm selection functor.
/// @tparam dtype_a Is the memory data type of matA.
/// @tparam dtype_b Is the memory data type of matB.
/// @tparam mem_layout_a Is the memory layout of matA.
/// @tparam mem_layout_b Is the memory layout of matB.
/// @tparam mem_space_a Is the memory space of matA.
/// @tparam mem_space_b Is the memory space of matB.
/// @tparam alignment_a Is the memory alignment of matA.
/// @tparam alignment_b Is the memory alignment of matB.
/// @tparam dtype_acc Is the compute data type.
/// @tparam tile_shape Is the workgroup-level tile shape.
/// @tparam k_stride Is the accumulate stride along k-dim.
/// @tparam engine Is the compute engine type.
/// @tparam arch_tag Is the HW architecture.
/// @tparam stages Is the prefetch distance.
/// @tparam sync_freq Is the group sync frequency.
template <typename dtype_a, typename dtype_b, mem_layout mem_layout_a,
        mem_layout mem_layout_b, mem_space mem_space_a, mem_space mem_space_b,
        int alignment_a, int alignment_b, typename dtype_acc,
        typename tile_shape, int k_stride, mma_engine engine, gpu_arch arch_tag,
        int stages = 3, int sync_freq = 0, class enable = void>
class gemm_selector_t {};

/// @} xetla_gemm

} // namespace gpu::xetla::group
