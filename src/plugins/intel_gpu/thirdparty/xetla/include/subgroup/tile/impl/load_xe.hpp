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
struct check_load_type {
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
            = ((payload_t::memory_space == mem_space::global)
                    && (payload_t::message_type == msg_type::unaligned_2d)
                    && (payload_t::arch_tag == gpu_arch::Xe));

    static constexpr bool is_local_scatter_xe
            = ((payload_t::memory_space == mem_space::local)
                    && (payload_t::message_type == msg_type::scatter)
                    && (payload_t::arch_tag == gpu_arch::Xe));

    static constexpr bool is_local_block_1d_xe
            = ((payload_t::memory_space == mem_space::local)
                    && (tile_t::block_size_y == 1)
                    && (payload_t::message_type == msg_type::block_1d)
                    && (payload_t::arch_tag == gpu_arch::Xe));
};

} // namespace detail

/// @brief This function loads data from 2D memory surface.
/// Loads an array of rectangular regions (X,Y)..(X+W,Y+H) from memory into registers.
/// Each block will be loaded serially by its corresponding payload.
/// @tparam tile_t Is the tile_t struct contains registers.
/// These registers will be the destination of load operation.
/// @tparam payload_t Is the mem_payload_t struct describing the memory information
/// Payload indicates the source of load operation.
/// @tparam L1 Is the cache hint for L1 cache.
/// @tparam L2 Is the cache hint for L2 cache.
/// @param tile Is the tile object with type tile_t, holds the return data of the loads.
/// @param payload Is the payload object with type payload_t. Contains all the information for loads.
/// @return No return, update in place.
template <cache_hint L1 = cache_hint::cached,
        cache_hint L2 = cache_hint::cached, typename tile_t, typename payload_t>
__XETLA_API typename std::enable_if_t<tile_t::dst
        && detail::check_load_type<tile_t, payload_t>::is_global_2d_xe>
tile_load(tile_t &tile, payload_t &payload) {
    using dtype = typename tile_t::dtype;
    using load_dtype = typename payload_t::mem_dtype;
    using tile_desc = typename tile_t::tile_desc;
    using check_load = typename subgroup::check_load<gpu_arch::Xe, dtype,
            load_dtype>::template global_2d<payload_t::mem_transform,
            tile_desc::block_size_x>;

    static constexpr uint32_t tile_size_x = tile_desc::tile_size_x;
    static constexpr uint32_t tile_size_y = tile_desc::tile_size_y;
    static constexpr uint32_t block_size_x = tile_desc::block_size_x;
    static constexpr uint32_t block_size_y = tile_desc::block_size_y;
    static constexpr uint32_t remained_size_y = tile_desc::remained_size_y;

    static constexpr uint32_t block_elems = tile_desc::block_elems;

    static constexpr uint32_t num_block_x = tile_desc::num_block_x;
    static constexpr uint32_t num_block_y = tile_desc::num_block_y;
    static constexpr uint32_t num_block = tile_desc::num_block;

    static constexpr gpu_arch arch_tag = payload_t::arch_tag;

    static constexpr reg_layout reg_layout_ = tile_desc::register_layout;
    static constexpr bool is_vnni_reverse = payload_t::mem_dword_transpose
            && ((reg_layout_ == reg_layout::tiled)
                    || (reg_layout_ == reg_layout::transpose_tiled));
    static constexpr bool reg_transpose = tile_desc::reg_transpose;

    static constexpr mem_layout mem_layout_ = payload_t::memory_layout;
    static constexpr bool mem_transpose = payload_t::mem_transpose;
    static constexpr bool trans = reg_transpose ^ mem_transpose;
    static constexpr uint32_t scale_factor = payload_t::scale_factor;

    static constexpr bool mem_transform = payload_t::mem_transform;

    using load_store_attr = typename arch_attr_t<
            arch_tag>::template load_store_attr<msg_type::block_2d>;
    static constexpr uint32_t elems_per_CL
            = load_store_attr::cache_line_size_in_bytes / sizeof(dtype);
    static constexpr uint32_t elems_per_reg
            = arch_attr_t<arch_tag>::template register_attr<>::reg_in_bytes
            / sizeof(dtype);
    static constexpr int32_t max_load_block_height
            = load_store_attr::max_load_height_in_elem;
    static constexpr int32_t max_block_width
            = load_store_attr::max_load_width_in_bytes / sizeof(dtype);
    static constexpr int32_t max_trans_block_width
            = load_store_attr::max_trans_load_width_in_bytes / sizeof(dtype);
    static constexpr int32_t special_trans_block_width
            = load_store_attr::special_trans_load_width_in_bytes
            / sizeof(dtype);
    static constexpr int32_t trans_block_width_limit
            = (block_size_y % special_trans_block_width) == 0
            ? special_trans_block_width
            : max_trans_block_width;
    static constexpr int32_t max_vnni_block_width
            = load_store_attr::max_vnni_load_width_in_elems;

    static constexpr uint32_t ld_blk_size_y_limit
            = mem_transpose ? trans_block_width_limit : max_load_block_height;
    static constexpr uint32_t ld_blk_size_y = reg_transpose
            ? block_size_y
            : (block_size_y > ld_blk_size_y_limit ? ld_blk_size_y_limit
                                                  : block_size_y);

    // array len is used to make sure memory load is cache line aligned
    // disabled while register or memory transpose
    static constexpr uint8_t arr_len_candidate
            = (reg_transpose
                      || mem_transpose
                      // block elements should be integer
                      // times of register bytes
                      || ((block_size_y * block_size_x) % elems_per_reg != 0)
                      // tail blocks also need to meet above condition
                      || (((tile_size_y % block_size_y) * block_size_x)
                                      % elems_per_reg
                              != 0))
                    || (block_size_y > ld_blk_size_y_limit)
            ? 1
            : (((tile_size_x % elems_per_CL) == 0)
                            ? (((elems_per_CL % block_size_x) == 0)
                                            ? elems_per_CL / block_size_x
                                            : 1)
                            : ((tile_size_x < elems_per_CL)
                                            ? (tile_size_x / block_size_x)
                                            : 1));
    static constexpr bool is_valid_arr_len_candidate = (arr_len_candidate == 1)
            || (arr_len_candidate == 2) || (arr_len_candidate == 4);

    static constexpr uint8_t arr_len
            = is_valid_arr_len_candidate ? arr_len_candidate : 1;

    static_assert(reg_transpose || mem_transpose
                    || (!mem_transpose
                            && (block_size_x * arr_len) <= max_block_width),
            "When reg_transpose was disabled, check 2d block width "
            "restriction");
    static_assert(!reg_transpose
                    || (!mem_transpose
                            && (block_size_x * arr_len)
                                    <= trans_block_width_limit)
                    || (mem_transpose
                            && (block_size_y * arr_len) <= max_block_width),
            "When reg_transpose was enabled, check 2d block width "
            "restriction");
    static_assert(!reg_transpose
                    || (!mem_transpose
                            && (block_size_y <= max_load_block_height))
                    || (mem_transpose
                            && (block_size_x) <= max_load_block_height),
            "When reg_transpose was enabled, check 2d block height "
            "restriction");
    static_assert(tile_size_x % (block_size_x * arr_len) == 0,
            "tile_size_x should be a multiple of (block_size_x * arr_len)");
    static_assert(
            (reg_transpose
                    && ((block_size_x * sizeof(dtype)) % sizeof(load_dtype)
                            == 0))
                    || ((block_size_y * sizeof(dtype)) % sizeof(load_dtype)
                            == 0),
            "check vnni limitation for DW transpose");

    auto payload_2d = payload.payloads.xetla_format<uint32_t, num_block, 16>();
    uint32_t base_offset_y = 0;
#pragma unroll
    for (int i = 0; i < num_block_y; ++i) {
        constexpr uint32_t load_block_elems = block_elems * arr_len;
        auto payload_row = payload_2d.xetla_select<num_block_x, 1, 16, 1>(
                i * num_block_x, 0);
        detail::reset_tile_desc_core<num_block_x, block_size_x, ld_blk_size_y,
                scale_factor, arr_len, mem_transpose>(payload_row);
#pragma unroll
        for (int j = 0; j < num_block_x; j += arr_len) {
            xetla_tdescriptor tdesc = payload_row.row(j);
            auto reg_blk = tile.reg.xetla_select<load_block_elems, 1>(
                    (i * num_block_x + j) * block_elems);
            constexpr uint32_t ld_blk_height = (reg_transpose && trans)
                    ? detail::getNextPowerOf2<ld_blk_size_y>()
                    : ld_blk_size_y;
            constexpr uint32_t tmp_size
                    = ld_blk_height * block_size_x * arr_len;
            xetla_vector<dtype, tmp_size> reg_tmp;
#pragma unroll
            for (int ii = 0; ii < block_size_y / ld_blk_size_y; ++ii) {
                constexpr uint32_t load_elems
                        = ld_blk_size_y * block_size_x * arr_len;

                reg_tmp.xetla_format<native_type_t<load_dtype>>()
                        = xetla_tload_global<load_dtype,
                                ld_blk_height * block_size_x * arr_len
                                        / scale_factor,
                                L1, L2, trans, mem_transform, arch_tag>(tdesc);

                if constexpr (reg_transpose && trans) {
                    reg_blk.xetla_select<load_elems, 1>(ii * load_elems)
                            .xetla_format<native_type_t<load_dtype>>()
                            = reg_tmp.xetla_format<load_dtype,
                                             block_size_x / scale_factor,
                                             ld_blk_height>()
                                      .xetla_select<block_size_x / scale_factor,
                                              1, ld_blk_size_y, 1>(0, 0);
                } else {
                    reg_blk.xetla_select<tmp_size, 1>(ii * tmp_size) = reg_tmp;
                }

                if constexpr (mem_transpose) {
                    xetla_update_tdesc_offsetx(tdesc.xetla_format<uint32_t>(),
                            ld_blk_size_y / scale_factor);
                } else {
                    xetla_update_tdesc_offsety(
                            tdesc.xetla_format<uint32_t>(), ld_blk_size_y);
                }
            }
            // exceed HW limitation
            if constexpr (block_size_y % ld_blk_size_y != 0) {
                constexpr uint32_t remained_start_y
                        = block_size_y / ld_blk_size_y * ld_blk_size_y;
                constexpr uint32_t remained_start
                        = remained_start_y * block_size_x * arr_len;
                constexpr uint32_t remained_blk_size_y
                        = block_size_y % ld_blk_size_y;
                constexpr uint32_t load_elems
                        = remained_blk_size_y * block_size_x * arr_len;

                constexpr uint8_t block_width = mem_transpose
                        ? (remained_blk_size_y / scale_factor)
                        : block_size_x;
                constexpr uint8_t block_height
                        = trans ? block_size_x : remained_blk_size_y;
                constexpr uint32_t block_widthx_widthy_arrlen
                        = (block_width - 1) | ((block_height - 1) << 8);
                gpu::xetla::detail::xetla_set_block_widthx_widthy_arrlen(
                        tdesc.xetla_format<uint32_t>(),
                        block_widthx_widthy_arrlen);

                reg_blk.xetla_select<load_elems, 1>(remained_start)
                        .xetla_format<native_type_t<load_dtype>>()
                        = xetla_tload_global<load_dtype,
                                (load_elems / scale_factor), L1, L2, trans,
                                mem_transform, arch_tag>(tdesc);
            }
        }
    }
    // process tail
    if constexpr (remained_size_y > 0) {
        constexpr uint32_t remained_block_elems
                = block_size_x * remained_size_y;
        constexpr uint32_t processed_elems
                = num_block_y * num_block_x * block_elems;

        static constexpr int32_t tail_trans_block_width_limit
                = (remained_size_y % special_trans_block_width) == 0
                ? special_trans_block_width
                : max_trans_block_width;

        static constexpr uint32_t tail_ld_blk_size_y_limit = mem_transpose
                ? tail_trans_block_width_limit
                : max_load_block_height;

        constexpr uint32_t remained_ld_blk_size_y
                = (!reg_transpose
                          && (remained_size_y > tail_ld_blk_size_y_limit))
                ? tail_ld_blk_size_y_limit
                : remained_size_y;
        auto payload_row = payload_2d.xetla_select<num_block_x, 1, 16, 1>(
                num_block_y * num_block_x, 0);
        detail::reset_tile_desc_core<num_block_x, block_size_x,
                remained_ld_blk_size_y, scale_factor, arr_len, mem_transpose>(
                payload_row);
#pragma unroll
        for (int j = 0; j < num_block_x; j += arr_len) {
            xetla_tdescriptor tdesc = payload_row.row(j);
            auto reg_blk
                    = tile.reg.xetla_select<remained_block_elems * arr_len, 1>(
                            processed_elems + j * remained_block_elems);
            constexpr uint32_t ld_blk_height = (reg_transpose && trans)
                    ? detail::getNextPowerOf2<remained_ld_blk_size_y>()
                    : remained_ld_blk_size_y;
            constexpr uint32_t tmp_size
                    = ld_blk_height * block_size_x * arr_len;
            xetla_vector<dtype, tmp_size> reg_tmp;
#pragma unroll
            for (int ii = 0; ii < remained_size_y / remained_ld_blk_size_y;
                    ++ii) {
                constexpr uint32_t load_elems
                        = remained_ld_blk_size_y * block_size_x * arr_len;

                reg_tmp.xetla_format<native_type_t<load_dtype>>()
                        = xetla_tload_global<load_dtype,
                                (ld_blk_height * block_size_x * arr_len
                                        / scale_factor),
                                L1, L2, trans, mem_transform, arch_tag>(tdesc);

                if constexpr (reg_transpose && trans) {
                    reg_blk.xetla_select<load_elems, 1>(ii * load_elems)
                            .xetla_format<native_type_t<load_dtype>>()
                            = reg_tmp.xetla_format<load_dtype,
                                             block_size_x / scale_factor,
                                             ld_blk_height>()
                                      .xetla_select<block_size_x / scale_factor,
                                              1, remained_ld_blk_size_y, 1>(
                                              0, 0);
                } else {
                    reg_blk.xetla_select<tmp_size, 1>(ii * tmp_size) = reg_tmp;
                }
                if constexpr (mem_transpose) {
                    xetla_update_tdesc_offsetx(tdesc.xetla_format<uint32_t>(),
                            remained_ld_blk_size_y / scale_factor);
                } else {
                    xetla_update_tdesc_offsety(tdesc.xetla_format<uint32_t>(),
                            remained_ld_blk_size_y);
                }
            }
            constexpr uint32_t final_ld_blk_size_y
                    = remained_size_y % remained_ld_blk_size_y;
            if constexpr (final_ld_blk_size_y != 0) {
                constexpr uint32_t final_start = remained_size_y
                        / remained_ld_blk_size_y * remained_ld_blk_size_y
                        * block_size_x * arr_len;
                constexpr uint32_t final_load_elems
                        = final_ld_blk_size_y * block_size_x * arr_len;
                constexpr uint8_t block_width = mem_transpose
                        ? (final_ld_blk_size_y / scale_factor)
                        : block_size_x;
                constexpr uint8_t block_height
                        = trans ? block_size_x : final_ld_blk_size_y;
                constexpr uint32_t block_widthx_widthy_arrlen
                        = (block_width - 1) | ((block_height - 1) << 8);
                gpu::xetla::detail::xetla_set_block_widthx_widthy_arrlen(
                        tdesc.xetla_format<uint32_t>(),
                        block_widthx_widthy_arrlen);
                reg_blk.xetla_select<final_load_elems, 1>(final_start)
                        .xetla_format<native_type_t<load_dtype>>()
                        = xetla_tload_global<load_dtype,
                                final_load_elems / scale_factor, L1, L2, trans,
                                mem_transform, arch_tag>(tdesc);
            }
        }
    }

    if constexpr (is_vnni_reverse) {
        SW_BARRIER();
        vnni_reverse(tile);
    }
}

template <cache_hint L1 = cache_hint::cached,
        cache_hint L2 = cache_hint::cached, typename tile_t, typename payload_t>
__XETLA_API typename std::enable_if_t<!tile_t::dst> tile_load(
        tile_t &tile, payload_t &payload) {
    using dtype = typename tile_t::dtype;
    using load_dtype = typename payload_t::mem_dtype;
    using tile_desc = typename tile_t::tile_desc;
    using check_load = typename subgroup::check_load<gpu_arch::Xe, dtype,
            load_dtype>::template global_2d<payload_t::mem_transform,
            tile_desc::block_size_x>;

    static constexpr uint32_t tile_size_x = tile_desc::tile_size_x;
    static constexpr uint32_t tile_size_y = tile_desc::tile_size_y;
    static constexpr uint32_t block_size_x = tile_desc::block_size_x;
    static constexpr uint32_t block_size_y = tile_desc::block_size_y;
    static constexpr uint32_t remained_size_y = tile_desc::remained_size_y;

    static constexpr uint32_t block_elems = tile_desc::block_elems;

    static constexpr uint32_t num_block_x = tile_desc::num_block_x;
    static constexpr uint32_t num_block_y = tile_desc::num_block_y;
    static constexpr uint32_t num_block = tile_desc::num_block;

    static constexpr gpu_arch arch_tag = payload_t::arch_tag;

    static constexpr reg_layout reg_layout_ = tile_desc::register_layout;
    static constexpr bool is_vnni_reverse = payload_t::mem_dword_transpose
            && ((reg_layout_ == reg_layout::tiled)
                    || (reg_layout_ == reg_layout::transpose_tiled));
    static constexpr bool reg_transpose = tile_desc::reg_transpose;

    static constexpr mem_layout mem_layout_ = payload_t::memory_layout;
    static constexpr bool mem_transpose = payload_t::mem_transpose;
    static constexpr bool trans = reg_transpose ^ mem_transpose;
    static constexpr uint32_t scale_factor = payload_t::scale_factor;

    static constexpr bool mem_transform = payload_t::mem_transform;

    using load_store_attr = typename arch_attr_t<
            arch_tag>::template load_store_attr<msg_type::block_2d>;
    static constexpr uint32_t elems_per_CL
            = load_store_attr::cache_line_size_in_bytes / sizeof(dtype);
    static constexpr uint32_t elems_per_reg
            = arch_attr_t<arch_tag>::template register_attr<>::reg_in_bytes
            / sizeof(dtype);
    static constexpr int32_t max_load_block_height
            = load_store_attr::max_load_height_in_elem;
    static constexpr int32_t max_block_width
            = load_store_attr::max_load_width_in_bytes / sizeof(dtype);
    static constexpr int32_t max_trans_block_width
            = load_store_attr::max_trans_load_width_in_bytes / sizeof(dtype);
    static constexpr int32_t special_trans_block_width
            = load_store_attr::special_trans_load_width_in_bytes
            / sizeof(dtype);
    static constexpr int32_t trans_block_width_limit
            = (block_size_y % special_trans_block_width) == 0
            ? special_trans_block_width
            : max_trans_block_width;
    static constexpr int32_t max_vnni_block_width
            = load_store_attr::max_vnni_load_width_in_elems;

    static constexpr uint32_t ld_blk_size_y_limit
            = mem_transpose ? trans_block_width_limit : max_load_block_height;
    static constexpr uint32_t ld_blk_size_y = reg_transpose
            ? block_size_y
            : (block_size_y > ld_blk_size_y_limit ? ld_blk_size_y_limit
                                                  : block_size_y);

    // array len is used to make sure memory load is cache line aligned
    // disabled while register or memory transpose
    static constexpr uint8_t arr_len_candidate
            = (reg_transpose
                      || mem_transpose
                      // block elements should be integer
                      // times of register bytes
                      || ((block_size_y * block_size_x) % elems_per_reg != 0)
                      // tail blocks also need to meet above condition
                      || (((tile_size_y % block_size_y) * block_size_x)
                                      % elems_per_reg
                              != 0))
                    || (block_size_y > ld_blk_size_y_limit)
            ? 1
            : (((tile_size_x % elems_per_CL) == 0)
                            ? (((elems_per_CL % block_size_x) == 0)
                                            ? elems_per_CL / block_size_x
                                            : 1)
                            : ((tile_size_x < elems_per_CL)
                                            ? (tile_size_x / block_size_x)
                                            : 1));
    static constexpr bool is_valid_arr_len_candidate = (arr_len_candidate == 1)
            || (arr_len_candidate == 2) || (arr_len_candidate == 4);

    static constexpr uint8_t arr_len
            = is_valid_arr_len_candidate ? arr_len_candidate : 1;

    static_assert(reg_transpose || mem_transpose
                    || (!mem_transpose
                            && (block_size_x * arr_len) <= max_block_width),
            "When reg_transpose was disabled, check 2d block width "
            "restriction");
    static_assert(!reg_transpose
                    || (!mem_transpose
                            && (block_size_x * arr_len)
                                    <= trans_block_width_limit)
                    || (mem_transpose
                            && (block_size_y * arr_len) <= max_block_width),
            "When reg_transpose was enabled, check 2d block width "
            "restriction");
    static_assert(!reg_transpose
                    || (!mem_transpose
                            && (block_size_y <= max_load_block_height))
                    || (mem_transpose
                            && (block_size_x) <= max_load_block_height),
            "When reg_transpose was enabled, check 2d block height "
            "restriction");
    static_assert(tile_size_x % (block_size_x * arr_len) == 0,
            "tile_size_x should be a multiple of (block_size_x * arr_len)");
    static_assert(
            (reg_transpose
                    && ((block_size_x * sizeof(dtype)) % sizeof(load_dtype)
                            == 0))
                    || ((block_size_y * sizeof(dtype)) % sizeof(load_dtype)
                            == 0),
            "check vnni limitation for DW transpose");

    auto payload_2d = payload.payloads.xetla_format<uint32_t, num_block, 16>();
    uint32_t base_offset_y = 0;
#pragma unroll
    for (int i = 0; i < num_block_y; ++i) {
        constexpr uint32_t load_block_elems = block_elems * arr_len;
        auto payload_row = payload_2d.xetla_select<num_block_x, 1, 16, 1>(
                i * num_block_x, 0);
        detail::reset_tile_desc_core<num_block_x, block_size_x, ld_blk_size_y,
                scale_factor, arr_len, mem_transpose>(payload_row);
#pragma unroll
        for (int j = 0; j < num_block_x; j += arr_len) {
            xetla_tdescriptor tdesc = payload_row.row(j);
            constexpr uint32_t ld_blk_height = (reg_transpose && trans)
                    ? detail::getNextPowerOf2<ld_blk_size_y>()
                    : ld_blk_size_y;
#pragma unroll
            for (int ii = 0; ii < block_size_y / ld_blk_size_y; ++ii) {
                xetla_tprefetch_global<load_dtype, L1, L2, arch_tag>(tdesc);
                if constexpr (mem_transpose) {
                    xetla_update_tdesc_offsetx(tdesc.xetla_format<uint32_t>(),
                            ld_blk_size_y / scale_factor);
                } else {
                    xetla_update_tdesc_offsety(
                            tdesc.xetla_format<uint32_t>(), ld_blk_size_y);
                }
            }
            // exceed HW limitation
            if constexpr (block_size_y % ld_blk_size_y != 0) {
                constexpr uint32_t remained_start_y
                        = block_size_y / ld_blk_size_y * ld_blk_size_y;
                constexpr uint32_t remained_start
                        = remained_start_y * block_size_x * arr_len;
                constexpr uint32_t remained_blk_size_y
                        = block_size_y % ld_blk_size_y;
                constexpr uint32_t load_elems
                        = remained_blk_size_y * block_size_x * arr_len;

                constexpr uint8_t block_width = mem_transpose
                        ? (remained_blk_size_y / scale_factor)
                        : block_size_x;
                constexpr uint8_t block_height
                        = trans ? block_size_x : remained_blk_size_y;
                constexpr uint32_t block_widthx_widthy_arrlen
                        = (block_width - 1) | ((block_height - 1) << 8);
                gpu::xetla::detail::xetla_set_block_widthx_widthy_arrlen(
                        tdesc.xetla_format<uint32_t>(),
                        block_widthx_widthy_arrlen);
                xetla_tprefetch_global<load_dtype, L1, L2, arch_tag>(tdesc);
            }
        }
    }
    // process tail
    if constexpr (remained_size_y > 0) {
        constexpr uint32_t remained_block_elems
                = block_size_x * remained_size_y;
        constexpr uint32_t processed_elems
                = num_block_y * num_block_x * block_elems;

        static constexpr int32_t tail_trans_block_width_limit
                = (remained_size_y % special_trans_block_width) == 0
                ? special_trans_block_width
                : max_trans_block_width;

        static constexpr uint32_t tail_ld_blk_size_y_limit = mem_transpose
                ? tail_trans_block_width_limit
                : max_load_block_height;

        constexpr uint32_t remained_ld_blk_size_y
                = (!reg_transpose
                          && (remained_size_y > tail_ld_blk_size_y_limit))
                ? tail_ld_blk_size_y_limit
                : remained_size_y;
        auto payload_row = payload_2d.xetla_select<num_block_x, 1, 16, 1>(
                num_block_y * num_block_x, 0);
        detail::reset_tile_desc_core<num_block_x, block_size_x,
                remained_ld_blk_size_y, scale_factor, arr_len, mem_transpose>(
                payload_row);
#pragma unroll
        for (int j = 0; j < num_block_x; j += arr_len) {
            xetla_tdescriptor tdesc = payload_row.row(j);
            constexpr uint32_t ld_blk_height = (reg_transpose && trans)
                    ? detail::getNextPowerOf2<remained_ld_blk_size_y>()
                    : remained_ld_blk_size_y;
#pragma unroll
            for (int ii = 0; ii < remained_size_y / remained_ld_blk_size_y;
                    ++ii) {
                xetla_tprefetch_global<load_dtype, L1, L2, arch_tag>(tdesc);
            }
            constexpr uint32_t final_ld_blk_size_y
                    = remained_size_y % remained_ld_blk_size_y;
            if constexpr (final_ld_blk_size_y != 0) {
                constexpr uint32_t final_start = remained_size_y
                        / remained_ld_blk_size_y * remained_ld_blk_size_y
                        * block_size_x * arr_len;
                constexpr uint32_t final_load_elems
                        = final_ld_blk_size_y * block_size_x * arr_len;
                constexpr uint8_t block_width = mem_transpose
                        ? (final_ld_blk_size_y / scale_factor)
                        : block_size_x;
                constexpr uint8_t block_height
                        = trans ? block_size_x : final_ld_blk_size_y;
                constexpr uint32_t block_widthx_widthy_arrlen
                        = (block_width - 1) | ((block_height - 1) << 8);
                gpu::xetla::detail::xetla_set_block_widthx_widthy_arrlen(
                        tdesc.xetla_format<uint32_t>(),
                        block_widthx_widthy_arrlen);
                xetla_tprefetch_global<load_dtype, L1, L2, arch_tag>(tdesc);
            }
        }
    }
}

/// @brief This function loads data from memory.
/// For each enabled SIMT lane, a vector is read from memory into registers.
/// @tparam tile_t Is the tile_t struct contains registers.
/// These registers will be the destination of load operation.
/// @tparam payload_t Is the mem_payload_t struct describing the memory information.
/// Payload indicates the source of load operation.
/// @tparam L1 Is the cache hint for L1 cache.
/// @tparam L2 Is the cache hint for L2 cache.
/// @param tile Is the tile object with type tile_t, holds the return data of the loads.
/// @param payload Is the payload object with type payload_t. Contains all the information for loads.
/// @return No return, update in place.
template <cache_hint L1 = cache_hint::cached,
        cache_hint L2 = cache_hint::cached, typename tile_t, typename payload_t>
__XETLA_API typename std::enable_if_t<
        detail::check_load_type<tile_t, payload_t>::is_global_block_1d_xe>
tile_load(tile_t &tile, payload_t &payload) {
    using dtype = typename tile_t::dtype;
    using load_dtype = typename payload_t::mem_dtype;
    using check_load = typename subgroup::check_load<gpu_arch::Xe, dtype,
            load_dtype>::global_1d;

    static constexpr uint32_t tile_size_x = tile_t::tile_size_x;
    static constexpr uint32_t scale_factor = payload_t::scale_factor;
    constexpr uint32_t load_len = tile_size_x / scale_factor;

    if constexpr (load_len >= 64) {
#pragma unroll
        for (int i = 0; i < load_len / 64; i++) {
            uint32_t offset_x = i * 64 * scale_factor;
            auto reg_sub
                    = tile.reg.xetla_select<64 * scale_factor, 1>(offset_x);
            uint32_t address_offset = offset_x * sizeof(dtype);
            reg_sub.xetla_format<load_dtype>() = xetla_load_global<load_dtype,
                    64, data_size::default_size, L1, L2>(
                    payload.base_ptr, payload.base_offset + address_offset);
        }
    }
    constexpr uint32_t tail_len = load_len % 64;
    uint32_t tail_offset = load_len / 64 * 64 * scale_factor;
    detail::process_1d_tail<tail_len, 32, detail::process_flag::load, L1, L2>(
            tile, payload, tail_offset);
}

/// @brief This function loads data from unaligned-2D memory surface.
/// Loads an array of rectangular regions (X,Y)..(X+W,Y+H) from memory into registers.
/// Each block will be loaded serially by its corresponding payload.
/// @tparam tile_t Is the tile_t struct contains registers.
/// These registers will be the destination of load operation.
/// @tparam payload_t Is the mem_payload_t struct describing the memory information.
/// Payload indicates the source of load operation.
/// @tparam L1 Is the cache hint for L1 cache.
/// @tparam L3 Is the cache hint for L3 cache.
/// @param tile Is the tile object with type tile_t, holds the return data of the loads.
/// @param payload Is the payload object with type payload_t. Contains all the information for loads.
/// @return No return, update in place.
template <cache_hint L1 = cache_hint::cached,
        cache_hint L3 = cache_hint::cached, typename tile_t, typename payload_t,
        typename oob_check_tag = global_atomic_oob_check_on_tag>
__XETLA_API typename std::enable_if_t<
        detail::check_load_type<tile_t, payload_t>::is_global_unaligned_2d_xe>
tile_load(tile_t &tile, payload_t &payload, oob_check_tag tag = {}) {
    constexpr bool oob_check = std::is_same<oob_check_tag,
            global_atomic_oob_check_on_tag>::value;
    using dtype = typename payload_t::dtype;
    using tile_desc = typename payload_t::tile_desc;
    using load_dtype = typename payload_t::mem_dtype;
    using check_load = typename subgroup::check_load<gpu_arch::Xe, dtype,
            load_dtype>::template unaligned_2d<payload_t::mem_transform,
            tile_desc::block_size_x>;
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
                xetla_vector<load_dtype, load_elems> reg_tmp;
                xetla_mask<load_elems> pred_y = oob_check
                        ? payload.step_y + payload.base_y + offset_y
                                        + sub_block_y
                                < payload.height_in_elems
                        : 1;

                uint32_t address_offset = payload_t::trans
                        ? offset_x * payload.pitch_in_bytes
                                + (offset_y + sub_block_y) * sizeof(dtype)
                        : offset_x * sizeof(dtype)
                                + (offset_y + sub_block_y)
                                        * payload.pitch_in_bytes;

                reg_tmp = xetla_load_global<load_dtype, 1,
                        data_size::default_size, L1, L3, load_elems>(
                        payload.base_ptr,
                        payload.channel_offset + payload.base_offset
                                + address_offset,
                        pred_x && pred_y);
                reg_tmp.xetla_merge(reg_tmp, 0, pred_x && pred_y);

                reg_sub.xetla_select<load_elems * scale_factor, 1>(
                               sub_block_y * tile_desc::block_size_x)
                        .xetla_format<load_dtype>()
                        = reg_tmp;
            }
        }
    }
    //process the tail
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
                xetla_vector<load_dtype, load_elems> reg_tmp;
                xetla_mask<load_elems> pred_y = oob_check
                        ? payload.step_y + payload.base_y + offset_y
                                        + sub_block_y
                                < payload.height_in_elems
                        : 1;

                uint32_t address_offset = payload_t::trans
                        ? offset_x * payload.pitch_in_bytes
                                + (offset_y + sub_block_y) * sizeof(dtype)
                        : offset_x * sizeof(dtype)
                                + (offset_y + sub_block_y)
                                        * payload.pitch_in_bytes;

                reg_tmp = xetla_load_global<load_dtype, 1,
                        data_size::default_size, L1, L3, load_elems>(
                        payload.base_ptr,
                        payload.channel_offset + payload.base_offset
                                + address_offset,
                        pred_x && pred_y);

                reg_tmp.xetla_merge(reg_tmp, 0, pred_x && pred_y);

                reg_sub.xetla_select<load_elems * scale_factor, 1>(
                               sub_block_y * tile_desc::block_size_x)
                        .xetla_format<load_dtype>()
                        = reg_tmp;
            }
        }
    }

    if constexpr (payload_t::mem_transform) {
        SW_BARRIER();
        vnni_convert(tile);
    }
}

/// @brief Is the data load func from local shared memory to register file, which
/// supports the memory surface is 1d or 2d scenario. And we always assume data in SLM
/// is row major.
/// @tparam tile_t Is the tile_t struct contains registers
/// These registers will be the destination of load operation.
/// @tparam payload_t Is the mem_payload_t struct describing the memory information.
/// Payload indicates the source of load operation.
/// @tparam L1 Is the cache hint for L1 cache.
/// @tparam L2 Is the cache hint for L2 cache.
/// @param tile Is the tile object with type tile_t, holds the return data of the loads.
/// @param payload Is the payload object with type payload_t. Contains all the information for loads.
/// @return No return, update in place.
template <cache_hint L1 = cache_hint::cached,
        cache_hint L2 = cache_hint::cached, typename tile_t, typename payload_t>
__XETLA_API typename std::enable_if_t<
        detail::check_load_type<tile_t, payload_t>::is_local_scatter_xe>
tile_load(tile_t &tile, payload_t &payload) {
    using dtype = typename payload_t::dtype;
    using tile_desc = typename payload_t::tile_desc;
    using load_dtype = typename payload_t::mem_dtype;
    using check_load = typename subgroup::check_load<gpu_arch::Xe, dtype,
            load_dtype>::template local_scatter<payload_t::memory_layout,
            payload_t::tile_desc::block_size_x, payload_t::tile_bytes,
            payload_t::min_bytes, payload_t::block_bytes,
            payload_t::num_channel_x, payload_t::num_channel>;

    constexpr uint32_t num_channel_y = payload_t::num_channel_y;
    constexpr uint32_t load_elems = num_channel_y * tile_desc::block_size_x;
    static constexpr bool mem_transform = payload_t::mem_transform;

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
                reg_sub.xetla_select<load_elems, 1>(
                               sub_block_y * tile_desc::block_size_x)
                        .xetla_format<load_dtype>()
                        = xetla_load_local<load_dtype>(
                                payload.address + address_offset);
            }
        }
    }
    //process the tail
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
                    sub_block_y += num_channel_y) {
                uint32_t address_offset = offset_x * sizeof(dtype)
                        + (sub_block_y + offset_y) * payload.pitch_in_bytes;
                reg_sub.xetla_select<load_elems, 1>(
                               sub_block_y * tile_desc::block_size_x)
                        .xetla_format<load_dtype>()
                        = xetla_load_local<load_dtype>(
                                payload.address + address_offset);
            }
        }
    }
    if constexpr (mem_transform) {
        SW_BARRIER();
        vnni_convert(tile);
    }
}

/// @brief Is the data load func from shared local memory to register file, which
/// supports the memory surface is 1d scenario. And the src memory layout is always
/// row major.
/// @tparam tile_t Is the tile_t struct contains registers.
/// These registers will be the destination of load operation.
/// @tparam payload_t Is the mem_payload_t struct describing the memory information.
/// Payload indicates the source of load operation.
/// @tparam L1 Is the cache hint for L1 cache.
/// @tparam L2 Is the cache hint for L2 cache.
/// @param tile Is the tile object with type tile_t, holds the return data of the loads.
/// @param payload Is the payload object with type payload_t. Contains all the information for loads.
/// @return No return, update in place.
template <cache_hint L1 = cache_hint::cached,
        cache_hint L2 = cache_hint::cached, typename tile_t, typename payload_t>
__XETLA_API typename std::enable_if_t<
        detail::check_load_type<tile_t, payload_t>::is_local_block_1d_xe>
tile_load(tile_t &tile, payload_t &payload) {
    using dtype = typename tile_t::dtype;
    using tile_desc = typename tile_t::tile_desc;
    using load_dtype = typename payload_t::mem_dtype;
    using check_load = typename subgroup::check_load<gpu_arch::Xe, dtype,
            load_dtype>::local_1d;

    constexpr uint32_t scale_factor = payload_t::scale_factor;
    constexpr uint32_t load_len = tile_desc::tile_size_x / scale_factor;
#pragma unroll
    for (int i = 0; i < tile_desc::tile_size_y; i++) {
        uint32_t offset_y = i * tile_desc::tile_size_x;
        uint32_t address_offset_y = i * payload.pitch_in_bytes;
        if constexpr (load_len >= 64) {
#pragma unroll
            for (int j = 0; j < load_len / 64; j++) {
                uint32_t offset_x = j * 64 * scale_factor;
                auto reg_sub = tile.reg.xetla_select<64 * scale_factor, 1>(
                        offset_x + offset_y);
                uint32_t address_offset
                        = address_offset_y + offset_x * sizeof(dtype);
                reg_sub.xetla_format<load_dtype>()
                        = xetla_load_local<load_dtype, 64,
                                data_size::default_size>(payload.base_address
                                + payload.address + address_offset);
            }
        }
        uint32_t tail_offset = offset_y + load_len / 64 * 64 * scale_factor;
        uint32_t tail_address_offset = address_offset_y
                + load_len / 64 * 64 * scale_factor * sizeof(dtype);
        detail::process_1d_tail<load_len % 64, 32, detail::process_flag::load,
                L1, L2>(tile, payload, tail_offset, tail_address_offset);
    }
}

} // namespace gpu::xetla::subgroup
