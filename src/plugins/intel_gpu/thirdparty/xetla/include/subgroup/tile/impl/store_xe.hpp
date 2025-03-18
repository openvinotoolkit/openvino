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
template <typename tile_t, typename payload_t>
struct check_store_type {
    static constexpr bool is_global_2d_xe
            = (payload_t::memory_space == mem_space::global
                    && (payload_t::message_type == msg_type::block_2d)
                    && (payload_t::arch_tag == gpu_arch::Xe));

    static constexpr bool is_global_block_1d_xe
            = ((payload_t::memory_space == mem_space::global)
                    && (tile_t::tile_size_y == 1) && (tile_t::block_size_y == 1)
                    && (payload_t::message_type == msg_type::block_1d)
                    && (payload_t::arch_tag == gpu_arch::Xe));

    static constexpr bool is_global_unaligned_2d_xe
            = (payload_t::memory_space == mem_space::global
                    && (payload_t::message_type == msg_type::unaligned_2d)
                    && (payload_t::arch_tag == gpu_arch::Xe));

    static constexpr bool is_global_atomic_xe
            = ((payload_t::memory_space == mem_space::global)
                    && (payload_t::message_type == msg_type::atomic_add)
                    && (payload_t::arch_tag == gpu_arch::Xe));

    static constexpr bool is_local_scatter_xe = ((payload_t::memory_space
                                                         == mem_space::local)
            && (payload_t::message_type == msg_type::scatter)
            && (payload_t::arch_tag == gpu_arch::Xe)
            && (payload_t::tile_desc::register_layout == reg_layout::tiled
                    || payload_t::tile_desc::register_layout
                            == reg_layout::vnni_tiled));

    static constexpr bool is_local_scatter_vnni_col_xe
            = ((payload_t::memory_space == mem_space::local)
                    && (payload_t::message_type == msg_type::scatter)
                    && (payload_t::arch_tag == gpu_arch::Xe)
                    && (payload_t::tile_desc::register_layout
                            == reg_layout::vnni_tiled_col_major));

    static constexpr bool is_local_block_1d_xe = ((payload_t::memory_space
                                                          == mem_space::local)
            && (payload_t::message_type == msg_type::block_1d)
            && (payload_t::arch_tag == gpu_arch::Xe)
            && (payload_t::tile_desc::register_layout == reg_layout::tiled));
};

} // namespace detail

/// @brief Is the func storing data from register file to global memory.
/// store a rectangular region (X,Y)..(X+W,Y+H) into memory from registers.
/// @tparam tile_t Is the tile_t struct contains registers
/// These registers will be the source of store operation.
/// @tparam payload_t Is the mem_payload_t struct describing the memory info
/// payload indicates the destination of store operation.
/// @tparam L1 Is the cache hint for L1 cache.
/// @tparam L2 Is the cache hint for L2 cache.
/// @param tile Is the tile object with type tile_t, contains the data to be stored.
/// @param payload Is the payload object with type payload_t. Contains all the information for stores.
template <cache_hint L1 = cache_hint::write_back,
        cache_hint L2 = cache_hint::write_back, typename tile_t,
        typename payload_t>
__XETLA_API typename std::enable_if_t<
        detail::check_store_type<tile_t, payload_t>::is_global_2d_xe>
tile_store(tile_t &tile, payload_t &payload) {
    using dtype = typename tile_t::dtype;
    using tile_desc = typename tile_t::tile_desc;
    using store_dtype = typename payload_t::mem_dtype;
    using check_store =
            typename subgroup::check_store<gpu_arch::Xe, dtype, store_dtype>::
                    template global_2d<payload_t::tile_desc::block_size_x>;

    static constexpr uint32_t tile_size_x = tile_desc::tile_size_x;
    static constexpr uint32_t tile_size_y = tile_desc::tile_size_y;
    static constexpr uint32_t block_size_x = tile_desc::block_size_x;
    static constexpr uint32_t block_size_y = tile_desc::block_size_y;
    static constexpr uint32_t remained_size_y = tile_desc::remained_size_y;

    static constexpr uint32_t block_elems = tile_desc::block_elems;

    static constexpr uint32_t num_block_x = tile_desc::num_block_x;
    static constexpr uint32_t num_block_y = tile_desc::num_block_y;
    static constexpr uint32_t num_block = tile_desc::num_block;

    using load_store_attr = typename arch_attr_t<
            payload_t::arch_tag>::template load_store_attr<msg_type::block_2d>;

    static constexpr int32_t max_block_width
            = load_store_attr::max_load_width_in_bytes / sizeof(dtype);
    static constexpr int32_t max_store_block_height
            = load_store_attr::max_store_height_in_elem;
    static_assert((max_block_width % block_size_x) == 0,
            "max_block_width should be a multiply of block size x.");
    static constexpr uint32_t elems_per_CL
            = load_store_attr::cache_line_size_in_bytes / sizeof(dtype);
    static constexpr uint32_t st_blk_size_y
            = block_size_y > max_store_block_height ? max_store_block_height
                                                    : block_size_y;
    // to make sure full CL store
    static constexpr uint32_t st_block_x = ((tile_size_x % elems_per_CL) == 0)
            ? elems_per_CL
            : (((elems_per_CL % tile_size_x) == 0) ? tile_size_x
                                                   : block_size_x);

    static constexpr uint8_t arr_len_candidate = st_block_x / block_size_x;
    static constexpr bool is_valid_arr_len_candidate = (arr_len_candidate == 1)
            || (arr_len_candidate == 2) || (arr_len_candidate == 4);

    static constexpr uint8_t arr_len
            = is_valid_arr_len_candidate ? arr_len_candidate : 1;

    auto payload_2d = payload.payloads.xetla_format<uint32_t, num_block, 16>();
    uint32_t base_offset_y = 0;
#pragma unroll
    for (int i = 0; i < num_block_y; ++i) {
        constexpr uint32_t store_block_elems = block_elems * arr_len;
        auto payload_row = payload_2d.xetla_select<num_block_x, 1, 16, 1>(
                i * num_block_x, 0);
        detail::reset_tile_desc_core<num_block_x, block_size_x * arr_len,
                st_blk_size_y, 1, 1, false>(payload_row);
        base_offset_y += block_size_y;
#pragma unroll
        for (int j = 0; j < num_block_x; j += arr_len) {
            xetla_tdescriptor tdesc = payload_row.row(j);
            auto reg_blk = tile.reg.xetla_select<store_block_elems, 1>(
                    (i * num_block_x + j) * block_elems);
            xetla_vector<dtype, store_block_elems> combine_blk;
            auto combine_blk_2d = combine_blk.xetla_format<native_type_t<dtype>,
                    block_size_y, block_size_x * arr_len>();
#pragma unroll
            for (int combine_i = 0; combine_i < arr_len; ++combine_i) {
                combine_blk_2d.xetla_select<block_size_y, 1, block_size_x, 1>(
                        0, combine_i * block_size_x)
                        = reg_blk.xetla_select<block_elems, 1>(
                                combine_i * block_elems);
            }
#pragma unroll
            for (int ii = 0; ii < block_size_y / st_blk_size_y; ++ii) {
                constexpr uint32_t store_elems
                        = st_blk_size_y * block_size_x * arr_len;
                auto st_blk = combine_blk.xetla_select<store_elems, 1>(
                        ii * store_elems);
                xetla_tstore_global<dtype, store_elems, L1, L2,
                        payload_t::arch_tag>(tdesc, st_blk);
                xetla_update_tdesc_offsety(
                        tdesc.xetla_format<uint32_t>(), st_blk_size_y);
            }
            // exceed hardware limitation
            if constexpr ((block_size_y % st_blk_size_y) != 0) {
                constexpr uint32_t blk_remained_start = block_size_y
                        / st_blk_size_y * st_blk_size_y * block_size_x
                        * arr_len;
                constexpr uint8_t blk_remained_y = block_size_y % st_blk_size_y;
                constexpr uint8_t blk_remained_elems
                        = blk_remained_y * block_size_x * arr_len;
                auto st_blk = combine_blk.xetla_select<blk_remained_elems, 1>(
                        blk_remained_start);
                constexpr uint32_t block_widthx_widthy_arrlen
                        = (block_size_x * arr_len - 1)
                        | ((blk_remained_y - 1) << 8);
                gpu::xetla::detail::xetla_set_block_widthx_widthy_arrlen(
                        tdesc.xetla_format<uint32_t>(),
                        block_widthx_widthy_arrlen);
                xetla_tstore_global<dtype, blk_remained_elems, L1, L2,
                        payload_t::arch_tag>(tdesc, st_blk);
            }
        }
    }
    // process tail
    if constexpr (remained_size_y > 0) {
        constexpr uint32_t remained_block_elems
                = block_size_x * remained_size_y;
        constexpr uint32_t processed_elems
                = num_block_y * num_block_x * block_elems;
        constexpr uint32_t remained_st_blk_size_y
                = st_blk_size_y > remained_size_y ? remained_size_y
                                                  : st_blk_size_y;
        auto payload_row = payload_2d.xetla_select<num_block_x, 1, 16, 1>(
                num_block_y * num_block_x, 0);
        detail::reset_tile_desc_core<num_block_x, block_size_x * arr_len,
                remained_st_blk_size_y, 1, 1, false>(payload_row);
#pragma unroll
        for (int j = 0; j < num_block_x; j += arr_len) {
            xetla_tdescriptor tdesc = payload_row.row(j);
            auto reg_blk
                    = tile.reg.xetla_select<remained_block_elems * arr_len, 1>(
                            processed_elems + j * remained_block_elems);
            // Do combination
            xetla_vector<dtype, remained_block_elems * arr_len> combine_blk;
            auto combine_blk_2d = combine_blk.xetla_format<native_type_t<dtype>,
                    remained_size_y, block_size_x * arr_len>();
#pragma unroll
            for (int combine_i = 0; combine_i < arr_len; ++combine_i) {
                combine_blk_2d
                        .xetla_select<remained_size_y, 1, block_size_x, 1>(
                                0, combine_i * block_size_x)
                        = reg_blk.xetla_select<remained_block_elems, 1>(
                                combine_i * remained_block_elems);
            }
#pragma unroll
            for (int ii = 0; ii < remained_size_y / remained_st_blk_size_y;
                    ++ii) {
                constexpr uint32_t store_elems
                        = remained_st_blk_size_y * block_size_x * arr_len;
                auto st_blk = combine_blk.xetla_select<store_elems, 1>(
                        ii * store_elems);
                xetla_tstore_global<dtype, store_elems, L1, L2,
                        payload_t::arch_tag>(tdesc, st_blk);
                xetla_update_tdesc_offsety(
                        tdesc.xetla_format<uint32_t>(), remained_st_blk_size_y);
            }
            constexpr uint32_t final_st_blk_size_y
                    = remained_size_y % remained_st_blk_size_y;
            if constexpr (final_st_blk_size_y != 0) {
                constexpr uint32_t final_start = remained_size_y
                        / remained_st_blk_size_y * remained_st_blk_size_y
                        * block_size_x * arr_len;
                constexpr uint32_t final_store_elems
                        = final_st_blk_size_y * block_size_x * arr_len;
                auto st_blk = combine_blk.xetla_select<final_store_elems, 1>(
                        final_start);
                constexpr uint32_t block_widthx_widthy_arrlen
                        = (block_size_x * arr_len - 1)
                        | ((final_st_blk_size_y - 1) << 8);
                gpu::xetla::detail::xetla_set_block_widthx_widthy_arrlen(
                        tdesc.xetla_format<uint32_t>(),
                        block_widthx_widthy_arrlen);
                xetla_tstore_global<dtype, final_store_elems, L1, L2,
                        payload_t::arch_tag>(tdesc, st_blk);
            }
        }
    }
}

/// @brief Is the func storing data from register file to global memory.
/// For each enabled SIMT lane, a vector is written into memory from registers.
/// @tparam tile_t Is the tile_t struct contains registers
/// These registers will be the source of store operation.
/// @tparam payload_t Is the mem_payload_t struct describing the memory info
/// payload indicates the destination of store operation
/// @tparam L1 Is the cache hint for L1 cache.
/// @tparam L2 Is the cache hint for L2 cache.
/// @param tile Is the tile object with type tile_t, contains the data to be stored.
/// @param payload Is the payload object with type payload_t. Contains all the information for stores.
template <cache_hint L1 = cache_hint::write_back,
        cache_hint L2 = cache_hint::write_back, typename tile_t,
        typename payload_t>
__XETLA_API typename std::enable_if_t<
        detail::check_store_type<tile_t, payload_t>::is_global_block_1d_xe>
tile_store(tile_t &tile, payload_t &payload) {
    using dtype = typename tile_t::dtype;
    using store_dtype = typename payload_t::mem_dtype;
    using check_store = typename subgroup::check_store<gpu_arch::Xe, dtype,
            store_dtype>::global_1d;

    static constexpr uint32_t tile_size_x = tile_t::tile_size_x;
    static constexpr uint32_t scale_factor = payload_t::scale_factor;

    constexpr uint32_t store_len = tile_size_x / scale_factor;
    if constexpr (store_len >= 64) {
#pragma unroll
        for (int i = 0; i < store_len / 64; i++) {
            uint32_t offset_x = i * 64 * scale_factor;
            auto reg_sub
                    = tile.reg.xetla_select<64 * scale_factor, 1>(offset_x);
            uint32_t address_offset = offset_x * sizeof(dtype);

            xetla_store_global<store_dtype, 64, data_size::default_size, L1,
                    L2>(payload.base_ptr, payload.base_offset + address_offset,
                    reg_sub.xetla_format<store_dtype>());
        }
    }
    constexpr uint32_t tail_len = store_len % 64;
    uint32_t tail_offset = store_len / 64 * 64 * scale_factor;
    detail::process_1d_tail<tail_len, 32, detail::process_flag::store, L1, L2>(
            tile, payload, tail_offset);
}

/// @brief Is the func storing data from register file to unaligned global memory surface.
/// store a rectangular region (X,Y)..(X+W,Y+H) into memory from registers.
/// @tparam tile_t Is the tile_t struct contains registers
/// These registers will be the source of store operation.
/// @tparam payload_t Is the mem_payload_t struct describing the memory info
/// payload indicates the destination of store operation.
/// @tparam L1 Is the cache hint for L1 cache.
/// @tparam L3 Is the cache hint for L3 cache.
/// @param tile Is the tile object with type tile_t, contains the data to be stored.
/// @param payload Is the payload object with type payload_t. Contains all the information for stores.
/// @return No return, update in place.
template <cache_hint L1 = cache_hint::write_back,
        cache_hint L3 = cache_hint::write_back, typename tile_t,
        typename payload_t,
        typename oob_check_tag = global_atomic_oob_check_on_tag>
__XETLA_API typename std::enable_if_t<
        detail::check_store_type<tile_t, payload_t>::is_global_unaligned_2d_xe>
tile_store(tile_t &tile, payload_t &payload, oob_check_tag tag = {}) {
    constexpr bool oob_check = std::is_same<oob_check_tag,
            global_atomic_oob_check_on_tag>::value;
    using dtype = typename payload_t::dtype;
    using tile_desc = typename payload_t::tile_desc;
    using store_dtype = typename payload_t::mem_dtype;
    using check_store =
            typename subgroup::check_store<gpu_arch::Xe, dtype, store_dtype>::
                    template unaligned_2d<payload_t::tile_desc::block_size_x>;
    static constexpr uint32_t block_size_x = tile_desc::block_size_x;
    static constexpr uint32_t block_size_y = tile_desc::block_size_y;
    // using num_block_x = tile_desc::num_block_x;
    constexpr uint32_t num_channel_y = payload_t::num_channel_y;
    constexpr uint32_t load_elems = num_channel_y * payload_t::num_channel_x;
    constexpr uint32_t scale_factor = payload_t::scale_factor;

#pragma unroll
    for (int i = 0; i < tile_desc::tile_size_y / tile_desc::block_size_y; i++) {
        uint32_t offset_y = i * tile_desc::block_size_y;
#pragma unroll
        for (int j = 0; j < tile_desc::num_block_x; j++) {
            uint32_t offset_x = j * tile_desc::block_size_x;
            auto reg_sub = tile.reg.xetla_select<tile_desc::block_elems, 1>(
                    (i * tile_desc::num_block_x + j) * tile_desc::block_elems);
            xetla_mask<load_elems> pred_x = oob_check
                    ? payload.step_x + payload.base_x + offset_x
                            < payload.width_in_elems
                    : 1;
#pragma unroll
            for (int sub_block_y = 0; sub_block_y < tile_desc::block_size_y;
                    sub_block_y += num_channel_y) {
                xetla_mask<load_elems> pred_y = oob_check
                        ? payload.step_y + payload.base_y + offset_y
                                        + sub_block_y
                                < payload.height_in_elems
                        : 1;

                uint32_t address_offset = offset_x * sizeof(dtype)
                        + (offset_y + sub_block_y) * payload.pitch_in_bytes;
                xetla_store_global<store_dtype, 1, data_size::default_size, L1,
                        L3, load_elems>(payload.base_ptr,
                        (payload.base_offset + address_offset
                                + payload.channel_offset),
                        reg_sub.xetla_select<load_elems * scale_factor, 1>(
                                       sub_block_y * tile_desc::block_size_x)
                                .xetla_format<store_dtype>(),
                        (pred_x && pred_y));
            }
        }
    }
    // process the tail
    if constexpr ((tile_desc::tile_size_y % tile_desc::block_size_y) != 0) {
        constexpr uint32_t remained_size_y = tile_desc::remained_size_y;
        constexpr uint32_t offset_y = tile_desc::tile_size_y - remained_size_y;
        constexpr uint32_t processed_elems = offset_y * tile_desc::tile_size_x;
        constexpr uint32_t remain_block_elems
                = remained_size_y * tile_desc::block_size_x;
#pragma unroll
        for (int j = 0; j < tile_desc::num_block_x; j++) {
            uint32_t offset_x = j * tile_desc::block_size_x;
            auto reg_sub = tile.reg.xetla_select<remain_block_elems, 1>(
                    processed_elems + j * remain_block_elems);
            xetla_mask<load_elems> pred_x = oob_check
                    ? payload.step_x + payload.base_x + offset_x
                            < payload.width_in_elems
                    : 1;
#pragma unroll
            for (int sub_block_y = 0; sub_block_y < remained_size_y;
                    sub_block_y += num_channel_y) {
                xetla_mask<load_elems> pred_y = oob_check
                        ? payload.step_y + payload.base_y + offset_y
                                        + sub_block_y
                                < payload.height_in_elems
                        : 1;

                uint32_t address_offset = offset_x * sizeof(dtype)
                        + (offset_y + sub_block_y) * payload.pitch_in_bytes;
                xetla_store_global<store_dtype, 1, data_size::default_size, L1,
                        L3, load_elems>(payload.base_ptr,
                        (payload.base_offset + address_offset
                                + payload.channel_offset),
                        reg_sub.xetla_select<load_elems * scale_factor, 1>(
                                       sub_block_y * tile_desc::block_size_x)
                                .xetla_format<store_dtype>(),
                        (pred_x && pred_y));
            }
        }
    }
}

/// @brief Is the func storing data from register file to global memory
/// enable atomic adding data into the same buffer, but support float32, float64, uint32_t,
/// uint64_t and int type
/// @tparam tile_t Is the tile_t struct contains registers
/// These registers will be the source of store operation.
/// @tparam payload_t Is the mem_payload_t struct describing the memory info
/// payload indicates the destination of store operation
/// @tparam L1 Is the cache hint for L1 cache.
/// @tparam L2 Is the cache hint for L2 cache.
/// @param tile Is the tile object with type tile_t, contains the data to be stored.
/// @param payload Is the payload object with type payload_t. Contains all the information for stores.
template <cache_hint L1 = cache_hint::uncached,
        cache_hint L2 = cache_hint::write_back, typename tile_t,
        typename payload_t,
        typename oob_check_tag = global_atomic_oob_check_on_tag>
__XETLA_API typename std::enable_if_t<
        detail::check_store_type<tile_t, payload_t>::is_global_atomic_xe>
tile_store(tile_t &tile, payload_t &payload, oob_check_tag tag = {}) {
    constexpr bool oob_check = std::is_same<oob_check_tag,
            global_atomic_oob_check_on_tag>::value;
    using dtype = typename tile_t::dtype;
    using tile_desc = typename tile_t::tile_desc;
    using check_store = typename subgroup::check_store<gpu_arch::Xe,
            dtype>::template global_atomic<payload_t::tile_bytes,
            payload_t::min_store_bytes, payload_t::block_bytes,
            payload_t::num_channel_x, payload_t::num_channel>;

    static constexpr uint32_t tile_size_x = tile_desc::tile_size_x;
    static constexpr uint32_t tile_size_y = tile_desc::tile_size_y;
    static constexpr uint32_t block_size_x = tile_desc::block_size_x;
    static constexpr uint32_t block_size_y = tile_desc::block_size_y;
    static constexpr uint32_t block_elems = tile_desc::block_elems;
    static constexpr uint32_t num_block_x = tile_desc::num_block_x;

    static constexpr atomic_op op_kind
            = (std::is_same<remove_const_t<dtype>, float>::value
                      || std::is_same<remove_const_t<dtype>, double>::value)
            ? atomic_op::fadd
            : atomic_op::iadd;

#pragma unroll
    for (int i = 0; i < tile_size_y / block_size_y; i++) {
        uint32_t offset_y = i * block_size_y;
#pragma unroll
        for (int j = 0; j < num_block_x; j++) {
            uint32_t offset_x = j * block_size_x;
            auto reg_sub = tile.reg.xetla_select<block_elems, 1>(
                    (i * num_block_x + j) * block_elems);
            xetla_mask<payload_t::num_channel> pred_x = oob_check
                    ? (payload.step_x + offset_x + payload.base_x)
                            < payload.width_in_elems
                    : 1;
#pragma unroll
            for (int sub_block_y = 0; sub_block_y < block_size_y;
                    sub_block_y += payload_t::num_channel_y) {
                xetla_mask<payload_t::num_channel> pred_y = oob_check
                        ? (payload.step_y + offset_y + payload.base_y
                                  + sub_block_y)
                                < payload.height_in_elems
                        : 1;
                uint64_t address_offset = offset_x * sizeof(dtype)
                        + (sub_block_y + offset_y) * payload.pitch_in_bytes;

                xetla_tatomic_store_global<dtype, payload_t::num_channel, L1,
                        L2, op_kind, payload_t::arch_tag,
                        typename payload_t::Toffset>(
                        payload.base_pointer + address_offset,
                        payload.channel_offset,
                        reg_sub.xetla_select<payload_t::store_elems, 1>(
                                sub_block_y * block_size_x),
                        pred_x & pred_y);
            }
        }
    }
    // process the tail
    if constexpr ((tile_size_y % block_size_y) != 0) {
        constexpr uint32_t remained_size_y = tile_desc::remained_size_y;
        constexpr uint32_t offset_y = tile_size_y - remained_size_y;
        constexpr uint32_t processed_elems = offset_y * tile_size_x;
        constexpr uint32_t remain_block_elems = remained_size_y * block_size_x;
#pragma unroll
        for (int j = 0; j < num_block_x; j++) {
            uint32_t offset_x = j * block_size_x;
            auto reg_sub = tile.reg.xetla_select<remain_block_elems, 1>(
                    processed_elems + j * remain_block_elems);
            xetla_mask<payload_t::num_channel> pred_x = oob_check
                    ? (payload.step_x + offset_x + payload.base_x)
                            < payload.width_in_elems
                    : 1;
#pragma unroll
            for (int sub_block_y = 0; sub_block_y < remained_size_y;
                    sub_block_y += payload_t::num_channel_y) {
                xetla_mask<payload_t::num_channel> pred_y = oob_check
                        ? (payload.step_y + offset_y + payload.base_y
                                  + sub_block_y)
                                < payload.height_in_elems
                        : 1;
                uint64_t address_offset = offset_x * sizeof(dtype)
                        + (sub_block_y + offset_y) * payload.pitch_in_bytes;

                xetla_tatomic_store_global<dtype, payload_t::num_channel, L1,
                        L2, op_kind, payload_t::arch_tag,
                        typename payload_t::Toffset>(
                        (uint64_t)payload.base_pointer + address_offset,
                        payload.channel_offset,
                        reg_sub.xetla_select<payload_t::store_elems, 1>(
                                sub_block_y * block_size_x),
                        pred_x & pred_y);
            }
        }
    }
}

/// @brief Is the func storing data from register file to shared local memory,
/// which supports the memory surface 2d scenario. And the dst memory layout is
/// is always row major.
/// @tparam tile_t Is the tile_t struct contains registers
/// These registers will be the source of store operation.
/// @tparam payload_t Is the mem_payload_t struct describing the memory info
/// payload indicates the destination of store operation
/// @tparam L1 Is the cache hint for L1 cache.
/// @tparam L2 Is the cache hint for L2 cache.
/// @param tile Is the tile object with type tile_t, contains the data to be stored.
/// @param payload Is the payload object with type payload_t. Contains all the information for stores.
template <cache_hint L1 = cache_hint::write_back,
        cache_hint L2 = cache_hint::write_back, typename tile_t,
        typename payload_t>
__XETLA_API typename std::enable_if_t<
        detail::check_store_type<tile_t, payload_t>::is_local_scatter_xe>
tile_store(tile_t &tile, payload_t &payload) {
    using dtype = typename tile_t::dtype;
    using tile_desc = typename tile_t::tile_desc;
    using store_dtype = typename payload_t::mem_dtype;
    using check_store = typename subgroup::check_store<gpu_arch::Xe, dtype,
            store_dtype>::template local_scatter<payload_t::tile_bytes,
            payload_t::min_bytes, payload_t::block_bytes,
            payload_t::num_channel_x, payload_t::num_channel>;

    constexpr uint32_t num_channel_y = payload_t::num_channel_y;
    constexpr uint32_t store_elems = num_channel_y * tile_desc::block_size_x;
#pragma unroll
    for (int i = 0; i < tile_desc::tile_size_y / tile_desc::block_size_y; i++) {
        uint32_t offset_y = i * tile_desc::block_size_y;
#pragma unroll
        for (int j = 0; j < tile_desc::num_block_x; j++) {
            uint32_t offset_x = j * tile_desc::block_size_x;
            auto reg_sub = tile.reg.xetla_select<tile_desc::block_elems, 1>(
                    (i * tile_desc::num_block_x + j) * tile_desc::block_elems);
#pragma unroll
            for (int sub_block_y = 0; sub_block_y < tile_desc::block_size_y;
                    sub_block_y += num_channel_y) {
                uint32_t address_offset = offset_x * sizeof(dtype)
                        + (sub_block_y + offset_y) * payload.pitch_in_bytes;
                xetla_store_local<store_dtype>(payload.address + address_offset,
                        reg_sub.xetla_select<store_elems, 1>(
                                       sub_block_y * tile_desc::block_size_x)
                                .xetla_format<store_dtype>());
            }
        }
    }
    // process the tail
    if constexpr ((tile_desc::tile_size_y % tile_desc::block_size_y) != 0) {
        constexpr uint32_t remained_size_y = tile_desc::remained_size_y;
        constexpr uint32_t offset_y = tile_desc::block_size_y - remained_size_y;
        constexpr uint32_t processed_elems = offset_y * tile_desc::tile_size_x;
        constexpr uint32_t remain_block_elems
                = remained_size_y * tile_desc::block_size_x;
#pragma unroll
        for (int j = 0; j < tile_desc::num_block_x; j++) {
            uint32_t offset_x = j * tile_desc::block_size_x;
            auto reg_sub = tile.reg.xetla_select<remain_block_elems, 1>(
                    processed_elems + j * remain_block_elems);
#pragma unroll
            for (int sub_block_y = 0; sub_block_y < remained_size_y;
                    sub_block_y += num_channel_y) {
                uint32_t address_offset = offset_x * sizeof(dtype)
                        + (sub_block_y + offset_y) * payload.pitch_in_bytes;
                xetla_store_local<store_dtype>(payload.address + address_offset,
                        reg_sub.xetla_select<store_elems, 1>(
                                       sub_block_y * tile_desc::block_size_x)
                                .xetla_format<store_dtype>());
            }
        }
    }
}

/// @brief Is the data store func from register file to local shared memory,
/// where the data in register is vnni packed and col major. And we always assume
/// the dst memory layout is row major.
/// @tparam tile_t Is the tile_t struct contains registers
/// These registers will be the source of store operation.
/// @tparam payload_t Is the mem_payload_t struct describing the memory info
/// payload indicates the destination of store operation
/// @tparam L1 Is the cache hint for L1 cache.
/// @tparam L2 Is the cache hint for L2 cache.
/// @param tile Is the tile object with type tile_t, contains the data to be stored.
/// @param payload Is the payload object with type payload_t. Contains all the information for stores.
template <cache_hint L1 = cache_hint::write_back,
        cache_hint L2 = cache_hint::write_back, typename tile_t,
        typename payload_t>
__XETLA_API typename std::enable_if_t<detail::check_store_type<tile_t,
        payload_t>::is_local_scatter_vnni_col_xe>
tile_store(tile_t &tile, payload_t &payload) {
    using dtype = typename tile_t::dtype;
    using tile_desc = typename tile_t::tile_desc;
    using store_dtype = typename payload_t::store_dtype;

    constexpr uint32_t vnni_scale_factor = payload_t::vnni_scale_factor;
    constexpr uint32_t num_vector_size = payload_t::num_vector_size;
    constexpr uint32_t store_elems = payload_t::store_elems;
#pragma unroll
    for (int i = 0; i < tile_desc::tile_size_y / tile_desc::block_size_y; i++) {
        uint32_t offset_y = i * tile_desc::block_size_y;
#pragma unroll
        for (int j = 0; j < tile_desc::num_block_x; j++) {
            uint32_t offset_x = j * tile_desc::block_size_x;
            auto reg_sub = tile.reg.xetla_select<tile_desc::block_elems, 1>(
                    (i * tile_desc::num_block_x + j) * tile_desc::block_elems);
#pragma unroll
            for (int sub_block_y = 0; sub_block_y < tile_desc::block_size_y;
                    sub_block_y += num_vector_size * vnni_scale_factor) {
                uint32_t address_offset = payload.base_address
                        + offset_x * payload.pitch_in_bytes
                        + (sub_block_y + offset_y) * sizeof(dtype);
                xetla_store_local<store_dtype, num_vector_size>(
                        payload.channel_address + address_offset,
                        reg_sub.xetla_select<store_elems, 1>(
                                       sub_block_y * tile_desc::block_size_x)
                                .xetla_format<store_dtype>());
            }
        }
    }
    // process the tail
    if constexpr ((tile_desc::tile_size_y % tile_desc::block_size_y) != 0) {
        constexpr uint32_t remained_size_y = tile_desc::remained_size_y;
        constexpr uint32_t offset_y = tile_desc::tile_size_y - remained_size_y;
        constexpr uint32_t processed_elems = offset_y * tile_desc::tile_size_x;
        constexpr uint32_t remain_block_elems
                = remained_size_y * tile_desc::block_size_x;
#pragma unroll
        for (int j = 0; j < tile_desc::num_block_x; j++) {
            uint32_t offset_x = j * tile_desc::block_size_x;
            auto reg_sub = tile.reg.xetla_select<remain_block_elems, 1>(
                    processed_elems + j * remain_block_elems);
#pragma unroll
            for (int sub_block_y = 0; sub_block_y < remained_size_y;
                    sub_block_y += num_vector_size * vnni_scale_factor) {
                uint32_t address_offset = payload.base_address
                        + offset_x * payload.pitch_in_bytes
                        + (sub_block_y + offset_y) * sizeof(dtype);
                xetla_store_local<store_dtype, num_vector_size>(
                        payload.channel_address + address_offset,
                        reg_sub.xetla_select<store_elems, 1>(
                                       sub_block_y * tile_desc::block_size_x)
                                .xetla_format<store_dtype>());
            }
        }
    }
}

/// @brief Is the data store func from register file to shared local memory,
/// where supports memory surface 1d or 2d scenario, and we always assume dst memory
/// layout is row major.
/// @tparam tile_t Is the tile_t struct contains registers
/// These registers will be the source of store operation.
/// @tparam payload_t Is the mem_payload_t struct describing the memory info
/// payload indicates the destination of store operation
/// @tparam L1 Is the cache hint for L1 cache.
/// @tparam L2 Is the cache hint for L2 cache.
/// @param tile Is the tile object with type tile_t, contains the data to be stored.
/// @param payload Is the payload object with type payload_t. Contains all the information for stores.
template <cache_hint L1 = cache_hint::write_back,
        cache_hint L2 = cache_hint::write_back, typename tile_t,
        typename payload_t>
__XETLA_API typename std::enable_if_t<
        detail::check_store_type<tile_t, payload_t>::is_local_block_1d_xe
        && tile_t::block_size_y != 1>
tile_store(tile_t &tile, payload_t &payload) {
    using dtype = typename tile_t::dtype;
    using tile_desc = typename tile_t::tile_desc;
    using store_dtype = typename payload_t::mem_dtype;
    using check_store = typename subgroup::check_store<gpu_arch::Xe, dtype,
            store_dtype>::local_1d;

    constexpr uint32_t vector_size
            = payload_t::bytes_per_row / sizeof(store_dtype);

#pragma unroll
    for (int i = 0; i < tile_desc::tile_size_y / tile_desc::block_size_y; i++) {
        uint32_t offset_y = i * tile_desc::block_size_y;
#pragma unroll
        for (int j = 0; j < tile_desc::num_block_x; j++) {
            uint32_t offset_x = j * tile_desc::block_size_x;
            auto reg_sub = tile.reg.xetla_select<tile_desc::block_elems, 1>(
                    (i * tile_desc::num_block_x + j) * tile_desc::block_elems);
            auto reg_sub_2d = reg_sub.xetla_format<dtype,
                    tile_desc::block_size_y, tile_desc::block_size_x>();
            uint32_t address_offset = offset_x * sizeof(dtype)
                    + offset_y * payload.pitch_in_bytes;
#pragma unroll
            for (int row_i = 0; row_i < tile_desc::block_size_y; row_i++) {
                xetla_store_local<store_dtype, vector_size>(payload.base_address
                                + payload.address + address_offset
                                + row_i * payload.pitch_in_bytes,
                        reg_sub_2d.row(row_i).xetla_format<store_dtype>());
            }
        }
    }
    // process the tail
    if constexpr ((tile_desc::tile_size_y % tile_desc::block_size_y) != 0) {
        constexpr uint32_t remained_size_y = tile_desc::remained_size_y;
        constexpr uint32_t offset_y = tile_desc::tile_size_y - remained_size_y;
        constexpr uint32_t processed_elems = offset_y * tile_desc::tile_size_x;
        constexpr uint32_t remain_block_elems
                = remained_size_y * tile_desc::block_size_x;
#pragma unroll
        for (int j = 0; j < tile_desc::num_block_x; j++) {
            uint32_t offset_x = j * tile_desc::block_size_x;
            auto reg_sub = tile.reg.xetla_select<remain_block_elems, 1>(
                    processed_elems + j * remain_block_elems);
            auto reg_sub_2d = reg_sub.xetla_format<dtype, remained_size_y,
                    tile_desc::block_size_x>();
            uint32_t address_offset = offset_x * sizeof(dtype)
                    + offset_y * payload.pitch_in_bytes;
#pragma unroll
            for (int row_i = 0; row_i < remained_size_y; row_i++) {
                xetla_store_local<store_dtype, vector_size>(payload.base_address
                                + payload.address + address_offset
                                + row_i * payload.pitch_in_bytes,
                        reg_sub_2d.row(row_i).xetla_format<store_dtype>());
            }
        }
    }
}

/// @brief Is the func storing data from register file to shared local memory,
/// the data in registers will be stored to SLM in 1d mode, and we always assume dst memory
/// layout is row major.
/// @tparam tile_t Is the tile_t struct contains registers
/// These registers will be the source of store operation.
/// @tparam payload_t Is the mem_payload_t struct describing the memory info
/// payload indicates the destination of store operation
/// @tparam L1 Is the cache hint for L1 cache.
/// @tparam L2 Is the cache hint for L2 cache.
/// @param tile Is the tile object with type tile_t, contains the data to be stored.
/// @param payload Is the payload object with type payload_t. Contains all the information for stores.
template <cache_hint L1 = cache_hint::write_back,
        cache_hint L2 = cache_hint::write_back, typename tile_t,
        typename payload_t>
__XETLA_API typename std::enable_if_t<
        detail::check_store_type<tile_t, payload_t>::is_local_block_1d_xe
        && tile_t::tile_size_y == 1 && tile_t::block_size_y == 1>
tile_store(tile_t &tile, payload_t &payload) {
    using dtype = typename tile_t::dtype;
    using tile_desc = typename payload_t::tile_desc;
    using store_dtype = typename payload_t::mem_dtype;
    using check_store = typename subgroup::check_store<gpu_arch::Xe, dtype,
            store_dtype>::local_1d;

    constexpr uint32_t scale_factor = payload_t::scale_factor;
    constexpr uint32_t store_len = tile_desc::tile_size_x / scale_factor;
    if constexpr (store_len >= 64) {
#pragma unroll
        for (int j = 0; j < store_len / 64; j++) {
            uint32_t offset_x = j * 64 * scale_factor;
            auto reg_sub
                    = tile.reg.xetla_select<64 * scale_factor, 1>(offset_x);
            uint32_t address_offset = offset_x * sizeof(dtype);
            xetla_store_local<store_dtype, 64>(
                    payload.base_address + payload.address + address_offset,
                    reg_sub.xetla_format<store_dtype>());
        }
    }
    uint32_t tail_offset = store_len / 64 * 64 * scale_factor;
    detail::process_1d_tail<store_len % 64, 32, detail::process_flag::store, L1,
            L2>(tile, payload, tail_offset, tail_offset * sizeof(dtype));
}

} // namespace gpu::xetla::subgroup
