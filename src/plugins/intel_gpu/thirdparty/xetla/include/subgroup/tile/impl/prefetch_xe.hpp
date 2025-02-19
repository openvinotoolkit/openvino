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

#include "subgroup/tile/api.hpp"
#include "subgroup/tile/impl/op_function.hpp"
#include "subgroup/tile/impl/payload_xe.hpp"

namespace gpu::xetla::subgroup {
namespace detail {
template <typename payload_t>
struct check_prefetch_type {
    static constexpr bool is_global_2d_xe
            = ((payload_t::memory_space == mem_space::global)
                    && (payload_t::tile_desc::tile_size_y != 1)
                    && (payload_t::arch_tag == gpu_arch::Xe));

    static constexpr bool is_global_block_1d_xe
            = ((payload_t::memory_space == mem_space::global)
                    && (payload_t::tile_desc::tile_size_y == 1)
                    && (payload_t::arch_tag == gpu_arch::Xe));

    static constexpr bool is_local_xe
            = ((payload_t::memory_space == mem_space::local)
                    && (payload_t::arch_tag == gpu_arch::Xe));
};

} // namespace detail

/// @brief Is prefetch data func, which data located in global memory is prefetched to
/// cache, where has higher bandwidth. e.g. In gemm, prefetch next iteration data for mma
/// consumption. This func is specicalized for block 2d scenario.
/// @tparam payload_t Is the mem_payload_t struct illustrating memory info
/// payload indicates the source of prefetch operation.
/// @tparam L1 Is cache hint for L1 cache.
/// @tparam L2 Is cache hint for L2 cache.
/// @param payload Is the payload object with type payload_t. Contains all the information for prefetches.
template <cache_hint L1 = cache_hint::cached,
        cache_hint L2 = cache_hint::cached, typename payload_t>
__XETLA_API typename std::enable_if_t<
        detail::check_prefetch_type<payload_t>::is_global_2d_xe>
tile_prefetch(payload_t &payload, uint16_t pred = 1) {
    using dtype = typename payload_t::dtype;
    static constexpr uint32_t num_tdesc = payload_t::num_tdesc;
    auto tdesc_2d
            = payload.tdesc_prefetch.xetla_format<uint32_t, num_tdesc, 16>();

#pragma unroll
    for (int i = 0; i < num_tdesc; i++) {
        xetla_mask<16> mask = pred;
        xetla_tprefetch_global<dtype, L1, L2, payload_t::arch_tag>(
                tdesc_2d.row(i), pred);
    }
}

/// @brief Is prefetch data func, which data located in global memory is prefetched to
/// cache, where has higher bandwidth. e.g. In gemm, prefetch next iteration data for mma
/// consumption. This func is specicalized for block 1d scenario.
/// @tparam payload_t Is the mem_payload_t struct illustrating memory info
/// payload indicates the source of prefetch operation
/// @tparam L1 Is cache hint for L1 cache.
/// @tparam L2 Is cache hint for L2 cache.
/// @param payload Is the payload object with type payload_t. Contains all the information for prefetches.
template <cache_hint L1 = cache_hint::cached,
        cache_hint L2 = cache_hint::cached, typename payload_t>
__XETLA_API typename std::enable_if_t<
        detail::check_prefetch_type<payload_t>::is_global_block_1d_xe>
tile_prefetch(payload_t &payload) {
    using dtype = typename payload_t::dtype;
    using tile_desc = typename payload_t::tile_desc;
    using prefetch_dtype = typename payload_t::prefetch_dtype;
    constexpr uint32_t prefetch_len
            = tile_desc::tile_size_x / payload_t::scale_factor;
    if constexpr (prefetch_len >= 64) {
#pragma unroll
        for (int j = 0; j < prefetch_len / 64; j++) {
            uint32_t offset_x = j * 64 * payload_t::scale_factor;
            uint32_t address_offset = offset_x * sizeof(dtype);
            xetla_prefetch_global<prefetch_dtype, 64, data_size::default_size,
                    L1, L2>(
                    payload.base_ptr, payload.base_offset + address_offset);
        }
    }
    constexpr uint32_t tail_len = prefetch_len % 64;
    uint32_t tail_offset = prefetch_len / 64 * 64 * payload_t::scale_factor;
    detail::process_1d_tail<tail_len, 32, L1, L2, payload_t>(
            payload, tail_offset);
}

/// @brief Is prefetch data func.
/// Current shared local memory prefetch is not supported yet. Only used to keep the consistency with global prefetch.
/// @tparam payload_t Is the mem_payload_t struct illustrating memory info.
/// @tparam L1 Is cache hint for L1 cache.
/// @tparam L2 Is cache hint for L2 cache.
/// @param payload Is the payload object with type payload_t. Contains all the information for prefetches.
template <cache_hint L1 = cache_hint::cached,
        cache_hint L2 = cache_hint::cached, typename payload_t>
__XETLA_API typename std::enable_if_t<
        detail::check_prefetch_type<payload_t>::is_local_xe>
tile_prefetch(payload_t &payload) {}

} // namespace gpu::xetla::subgroup
