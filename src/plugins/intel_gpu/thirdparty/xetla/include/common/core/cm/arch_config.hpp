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

#ifdef _WIN32
#include "../../../common/core/cm/common.hpp"
#else
#include "common/core/cm/common.hpp"
#endif

namespace gpu::xetla {

/// @addtogroup xetla_core_arch_config
/// @{

template <msg_type message_type, gpu_arch arch_tag>
struct load_store_attr_t {};
template <>
struct load_store_attr_t<msg_type::block_2d, gpu_arch::Xe> {
    static constexpr uint32_t max_load_height_in_elem = 32;
    static constexpr uint32_t max_load_width_in_bytes = 64;
    static constexpr uint32_t max_trans_load_width_in_bytes = 32;
    static constexpr uint32_t special_trans_load_width_in_bytes = 32;
    static constexpr uint32_t max_vnni_load_width_in_elems = 16;
    static constexpr uint32_t min_vnni_load_height_in_bytes = 4;

    static constexpr uint32_t max_store_height_in_elem = 8;
    static constexpr uint32_t max_store_width_in_bytes = 64;

    static constexpr uint32_t max_load_size_in_bytes = 2048;
    static constexpr uint32_t max_store_size_in_bytes = 512;

    static constexpr uint32_t special_prefetch_width_in_bytes = 64;

    static constexpr uint32_t cache_line_size_in_bytes = 64;
    static constexpr uint32_t alignment_in_bytes = 8;
};

template <gpu_arch arch_tag>
struct mma_attr_t {};
template <>
struct mma_attr_t<gpu_arch::Xe> {
    static constexpr uint32_t mma_m_in_elem = 8;
    static constexpr uint32_t mma_n_in_elem = 16;
    static constexpr uint32_t mma_k_in_bytes = 32;
};

template <grf_mode grf_num_mode, gpu_arch arch_tag>
struct register_attr_t {};
template <grf_mode grf_num_mode>
struct register_attr_t<grf_num_mode, gpu_arch::Xe> {
    static constexpr uint32_t acc_reg_in_bytes
            = (grf_num_mode == grf_mode::normal) ? 4 * 64 : 8 * 64;
    static constexpr uint32_t grf_in_bytes
            = (grf_num_mode == grf_mode::normal) ? 128 * 64 : 256 * 64;
    static constexpr uint32_t reg_in_bytes = 64;
};

template <gpu_arch arch_tag>
struct arch_attr_t {};
template <>
struct arch_attr_t<gpu_arch::Xe> {
    template <msg_type message_type = msg_type::block_2d>
    using load_store_attr = load_store_attr_t<message_type, gpu_arch::Xe>;

    template <grf_mode grf_num_mode = grf_mode::double_grf>
    using register_attr = register_attr_t<grf_num_mode, gpu_arch::Xe>;

    using mma_attr = mma_attr_t<gpu_arch::Xe>;

    static constexpr uint32_t max_wg_num = 64;
};

/// @} xetla_core_arch_config

} // namespace gpu::xetla
