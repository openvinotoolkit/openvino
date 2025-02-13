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
#include "subgroup/tile/common.hpp"

namespace gpu::xetla::subgroup {
/// @brief Is the element wise data conversion, the src and dst tile should have the same layout.
///
/// @tparam T_dst Is the destination tile data type.
/// @tparam T_src Is the source tile data type.
/// @param dst Is the reference of the destination tile object.
/// @param src Is the reference of the destination tile object.
/// @return No return, in-place update in the destination tile.
template <typename T_dst, typename T_src>
__XETLA_API
        typename std::enable_if_t<(T_src::register_layout != reg_layout::linear)
                && (T_dst::register_layout != reg_layout::linear)
                && (is_same_elements<T_dst, T_src>::value)
                && (!is_floating_to_integer<T_dst, T_src>::value)>
        elemwise_cvt(T_dst &dst, T_src &src) {
    constexpr uint32_t block_size_x = T_dst::block_size_x;
    constexpr uint32_t tile_elems = T_dst::tile_elems;
    using dtype_src = typename T_src::dtype;
    using dtype_dst = typename T_dst::dtype;
    if constexpr (std::is_same<dtype_src, dtype_dst>::value) {
        dst.reg = xetla_cvt<dtype_dst, dtype_src>(src.reg);
    } else {
#pragma unroll
        for (int i = 0; i < tile_elems; i += block_size_x) {
            dst.reg.xetla_select<block_size_x, 1>(i)
                    = xetla_cvt<dtype_dst, dtype_src, block_size_x>(
                            src.reg.xetla_select<block_size_x, 1>(i));
        }
    }
}

/// @brief Is the element wise data conversion from floating point to integral,
/// the src and dst tile should have the same layout.
/// @tparam T_dst Is the destination tile data type.
/// @tparam T_src Is the source tile data type.
/// @param dst Is the reference of the destination tile object.
/// @param src Is the reference of the destination tile object.
/// @return No return, in-place update in the destination tile.
template <typename T_dst, typename T_src>
__XETLA_API
        typename std::enable_if_t<(T_src::register_layout != reg_layout::linear)
                && (T_dst::register_layout != reg_layout::linear)
                && is_same_layout<T_dst, T_src>::value
                && is_floating_to_integer<T_dst, T_src>::value>
        elemwise_cvt(T_dst &dst, T_src &src) {
    constexpr uint32_t block_size_x = T_dst::block_size_x;
    constexpr uint32_t tile_elems = T_dst::tile_elems;
    using dtype_src = typename T_src::dtype;
    using dtype_dst = typename T_dst::dtype;

    xetla_vector<dtype_src, tile_elems> rnde_reg;
    //rnde
#pragma unroll
    for (int i = 0; i < tile_elems; i += block_size_x) {
        rnde_reg.xetla_select<block_size_x, 1>(i)
                = xetla_rnde<dtype_src, block_size_x>(
                        src.reg.xetla_select<block_size_x, 1>(i));
    }
    //sat
#pragma unroll
    for (int i = 0; i < tile_elems; i += block_size_x) {
        dst.reg.xetla_select<block_size_x, 1>(i)
                = xetla_sat<dtype_dst, dtype_src, block_size_x>(
                        rnde_reg.xetla_select<block_size_x, 1>(i));
    }
}

/// @brief element wise data conversion with scaling, the src and dst tile should have the same layout.
/// @tparam T_dst is the destination tile data type.
/// @tparam T_src is the source tile data type.
/// @param dst is the reference of the destination tile object.
/// @param src is the reference of the destination tile object.
/// @param scale is the scaling value to be applied before the assignment.
/// @return no return, in-place update in the destination tile.
template <typename T_dst, typename T_src>
__XETLA_API
        typename std::enable_if_t<(T_src::register_layout != reg_layout::linear)
                && (T_dst::register_layout != reg_layout::linear)
                && is_same_layout<T_dst, T_src>::value>
        elemwise_cvt(T_dst &dst, T_src &src, float scale) {
    dst.reg = xetla_cvt<typename T_dst::dtype, typename T_src::dtype>(
            src.reg, scale);
}

/// @brief Converts tiled layout to vnni_tiled layout format.
///
/// @tparam T Is the tile data type.
/// @param mat_Acc Is the reference of the tile object.
/// @return No return, update the data in-place.
template <typename T>
__XETLA_API
        typename std::enable_if_t<T::register_layout == reg_layout::vnni_tiled>
        vnni_convert(T &mat_Acc) {
    constexpr uint32_t tile_size_y = T::tile_size_y;
    constexpr uint32_t tile_size_x = T::tile_size_x;
    constexpr uint32_t tile_elems = tile_size_y * tile_size_x;
    constexpr uint32_t block_size_y = T::block_size_y;
    constexpr uint32_t block_size_x = T::block_size_x;
    constexpr uint32_t block_elems = block_size_y * block_size_x;
    constexpr int32_t num_block_x = tile_size_x / block_size_x;
    using dtype = typename T::dtype;
    constexpr int32_t vnni_stride = sizeof(uint32_t) / sizeof(dtype);
    constexpr int32_t move_cols = block_size_x * vnni_stride;
    constexpr int32_t move_rows = block_size_y / vnni_stride;
    xetla_vector<dtype, tile_elems> rdst;
    static_assert(block_size_y % vnni_stride == 0, "vnni alignement check");
    if constexpr (tile_size_x == 1) { return; }
#pragma unroll
    for (int i = 0; i < tile_size_y / block_size_y; i++) {
#pragma unroll
        for (int j = 0; j < num_block_x; j++) {
            auto reg = (mat_Acc.reg)
                               .xetla_select<block_elems, 1>(
                                       (i * num_block_x + j) * block_elems);
            auto reg_2d = reg.xetla_format<native_type_t<dtype>, block_size_y,
                    block_size_x>();
            auto reg_dst = rdst.xetla_select<block_elems, 1>(
                    (i * num_block_x + j) * block_elems);
            auto reg_dst_2d = reg_dst.xetla_format<native_type_t<dtype>,
                    move_rows, move_cols>();
#pragma unroll
            for (int vnni_i = 0; vnni_i < vnni_stride; vnni_i++) {
                reg_dst_2d
                        .xetla_select<move_rows, 1, block_size_x, vnni_stride>(
                                0, vnni_i)
                        = reg_2d.xetla_select<move_rows, vnni_stride,
                                block_size_x, 1>(vnni_i, 0);
            }
        }
    }
    // process the tail
    if constexpr ((tile_size_y % block_size_y) != 0) {
        constexpr int i = tile_size_y / block_size_y;
        constexpr uint32_t remain_elems_start = i * block_size_y * tile_size_x;
        constexpr uint32_t remain_size_y = tile_size_y % block_size_y;
        constexpr uint32_t remain_block_elems = remain_size_y * block_size_x;
        static_assert(
                remain_size_y % vnni_stride == 0, "vnni alignement check");
        constexpr int32_t remain_move_cols = block_size_x * vnni_stride;
        constexpr int32_t remain_move_rows = remain_size_y / vnni_stride;
#pragma unroll
        for (int j = 0; j < num_block_x; j++) {
            auto reg = (mat_Acc.reg)
                               .xetla_select<remain_block_elems, 1>(
                                       remain_elems_start
                                       + j * remain_block_elems);
            auto reg_2d = reg.xetla_format<native_type_t<dtype>, remain_size_y,
                    block_size_x>();
            auto reg_dst = rdst.xetla_select<remain_block_elems, 1>(
                    remain_elems_start + j * remain_block_elems);
            auto reg_dst_2d = reg_dst.xetla_format<native_type_t<dtype>,
                    remain_move_rows, remain_move_cols>();
            for (int vnni_i = 0; vnni_i < vnni_stride; vnni_i++) {
                reg_dst_2d.xetla_select<remain_move_rows, 1, block_size_x,
                        vnni_stride>(0, vnni_i)
                        = reg_2d.xetla_select<remain_move_rows, vnni_stride,
                                block_size_x, 1>(vnni_i, 0);
            }
        }
    }
    mat_Acc.reg = rdst;
}

/// @brief Converts vnni_tiled layout format to tiled layout.
///
/// @tparam T Is the tile data type.
/// @param mat_Acc Is the reference of the tile object.
/// @return No return, update the data in-place.
template <typename T>
__XETLA_API typename std::enable_if_t<T::register_layout == reg_layout::tiled
        || T::register_layout == reg_layout::vnni_tiled>
vnni_reverse(T &mat_Acc) {
    constexpr uint32_t tile_size_y = T::tile_size_y;
    constexpr uint32_t tile_size_x = T::tile_size_x;
    constexpr uint32_t tile_elems = tile_size_y * tile_size_x;
    constexpr uint32_t block_size_y = T::block_size_y;
    constexpr uint32_t block_size_x = T::block_size_x;
    constexpr uint32_t block_elems = block_size_y * block_size_x;
    constexpr int32_t num_block_x = tile_size_x / block_size_x;
    using dtype = typename T::dtype;
    constexpr int32_t vnni_stride = sizeof(uint32_t) / sizeof(dtype);
    constexpr int32_t move_cols = block_size_x * vnni_stride;
    constexpr int32_t move_rows = block_size_y / vnni_stride;
    xetla_vector<dtype, tile_elems> rdst;
    static_assert(block_size_y % vnni_stride == 0, "vnni alignement check");
    if constexpr (tile_size_x == 1) { return; }
#pragma unroll
    for (int i = 0; i < tile_size_y / block_size_y; i++) {
#pragma unroll
        for (int j = 0; j < num_block_x; j++) {
            auto reg = (mat_Acc.reg)
                               .xetla_select<block_elems, 1>(
                                       (i * num_block_x + j) * block_elems);
            auto reg_2d = reg.xetla_format<native_type_t<dtype>, move_rows,
                    move_cols>();
            auto reg_dst = rdst.xetla_select<block_elems, 1>(
                    (i * num_block_x + j) * block_elems);
            auto reg_dst_2d = reg_dst.xetla_format<native_type_t<dtype>,
                    block_size_y, block_size_x>();
#pragma unroll
            for (int vnni_i = 0; vnni_i < vnni_stride; vnni_i++) {
                reg_dst_2d
                        .xetla_select<move_rows, vnni_stride, block_size_x, 1>(
                                vnni_i, 0)
                        = reg_2d.xetla_select<move_rows, 1, block_size_x,
                                vnni_stride>(0, vnni_i);
            }
        }
    }
    // process the tail
    if constexpr ((tile_size_y % block_size_y) != 0) {
        constexpr int i = tile_size_y / block_size_y;
        constexpr uint32_t remain_elems_start = i * block_size_y * tile_size_x;
        constexpr uint32_t remain_size_y = tile_size_y % block_size_y;
        constexpr uint32_t remain_block_elems = remain_size_y * block_size_x;
        static_assert(
                remain_size_y % vnni_stride == 0, "vnni alignement check");
        constexpr int32_t remain_move_cols = block_size_x * vnni_stride;
        constexpr int32_t remain_move_rows = remain_size_y / vnni_stride;
#pragma unroll
        for (int j = 0; j < num_block_x; j++) {
            auto reg = (mat_Acc.reg)
                               .xetla_select<remain_block_elems, 1>(
                                       remain_elems_start
                                       + j * remain_block_elems);
            auto reg_2d = reg.xetla_format<native_type_t<dtype>,
                    remain_move_rows, remain_move_cols>();
            auto reg_dst = rdst.xetla_select<remain_block_elems, 1>(
                    remain_elems_start + j * remain_block_elems);
            auto reg_dst_2d = reg_dst.xetla_format<native_type_t<dtype>,
                    remain_size_y, block_size_x>();
            for (int vnni_i = 0; vnni_i < vnni_stride; vnni_i++) {
                reg_dst_2d.xetla_select<remain_move_rows, vnni_stride,
                        block_size_x, 1>(vnni_i, 0)
                        = reg_2d.xetla_select<remain_move_rows, 1, block_size_x,
                                vnni_stride>(0, vnni_i);
            }
        }
    }
    mat_Acc.reg = rdst;
}

/// @brief Converts vnni_tiled layout format to transpose_tiled layout.
///
/// @tparam T Is the tile data type.
/// @param mat_Acc Is the reference of the tile object.
/// @return No return, update the data in-place.
template <typename T>
__XETLA_API typename std::enable_if_t<T::register_layout
        == reg_layout::transpose_tiled>
vnni_reverse(T &mat_Acc) {
    constexpr uint32_t tile_size_y = T::tile_size_y;
    constexpr uint32_t tile_size_x = T::tile_size_x;
    constexpr uint32_t tile_elems = tile_size_y * tile_size_x;
    constexpr uint32_t block_size_y = T::block_size_y;
    constexpr uint32_t block_size_x = T::block_size_x;
    constexpr uint32_t block_elems = block_size_y * block_size_x;
    constexpr int32_t num_block_x = tile_size_x / block_size_x;
    using dtype = typename T::dtype;
    constexpr int32_t vnni_stride = sizeof(uint32_t) / sizeof(dtype);
    constexpr int32_t move_cols = block_size_y * vnni_stride;
    constexpr int32_t move_rows = block_size_x / vnni_stride;
    xetla_vector<dtype, tile_elems> rdst;
    static_assert(block_size_x % vnni_stride == 0, "vnni alignement check");
    if constexpr (tile_size_y == 1) { return; }
#pragma unroll
    for (int i = 0; i < tile_size_y / block_size_y; i++) {
#pragma unroll
        for (int j = 0; j < num_block_x; j++) {
            auto reg = (mat_Acc.reg)
                               .xetla_select<block_elems, 1>(
                                       (i * num_block_x + j) * block_elems);
            auto reg_2d = reg.xetla_format<native_type_t<dtype>, move_rows,
                    move_cols>();
            auto reg_dst = rdst.xetla_select<block_elems, 1>(
                    (i * num_block_x + j) * block_elems);
            //transpose
            auto reg_dst_2d = reg_dst.xetla_format<native_type_t<dtype>,
                    block_size_x, block_size_y>();
            for (int vnni_i = 0; vnni_i < vnni_stride; vnni_i++) {
                reg_dst_2d
                        .xetla_select<move_rows, vnni_stride, block_size_y, 1>(
                                vnni_i, 0)
                        = reg_2d.xetla_select<move_rows, 1, block_size_y,
                                vnni_stride>(0, vnni_i);
            }
        }
    }
    // process the tail
    if constexpr ((tile_size_y % block_size_y) != 0) {
        constexpr int i = tile_size_y / block_size_y;
        constexpr uint32_t remain_elems_start = i * block_size_y * tile_size_x;
        constexpr uint32_t remain_size_y = tile_size_y % block_size_y;
        constexpr uint32_t remain_block_elems = remain_size_y * block_size_x;
        constexpr int32_t remain_move_cols = remain_size_y * vnni_stride;
        constexpr int32_t remain_move_rows = block_size_x / vnni_stride;
#pragma unroll
        for (int j = 0; j < num_block_x; j++) {
            auto reg = (mat_Acc.reg)
                               .xetla_select<remain_block_elems, 1>(
                                       remain_elems_start
                                       + j * remain_block_elems);
            auto reg_2d = reg.xetla_format<native_type_t<dtype>,
                    remain_move_rows, remain_move_cols>();
            auto reg_dst = rdst.xetla_select<remain_block_elems, 1>(
                    remain_elems_start + j * remain_block_elems);
            //transpose
            auto reg_dst_2d = reg_dst.xetla_format<native_type_t<dtype>,
                    block_size_x, remain_size_y>();

            for (int vnni_i = 0; vnni_i < vnni_stride; vnni_i++) {
                reg_dst_2d.xetla_select<remain_move_rows, vnni_stride,
                        remain_size_y, 1>(vnni_i, 0)
                        = reg_2d.xetla_select<remain_move_rows, 1,
                                remain_size_y, vnni_stride>(0, vnni_i);
            }
        }
    }
    mat_Acc.reg = rdst;
}

/// @brief Changes vnni layout.
///
/// @tparam T_dst Is the destination tile data type.
/// @tparam T_src Is the source tile data type.
/// @param dst Is the reference of the destination tile object.
/// @param src Is the reference of the destination tile object.
/// @return No return, in-place update in the destination tile.
template <typename T_dst, typename T_src>
__XETLA_API typename std::enable_if_t<is_same_layout<T_dst, T_src>::value>
vnni_transform(T_dst &dst, T_src &src) {
    constexpr uint32_t tile_size_y = T_dst::tile_size_y;
    constexpr uint32_t tile_size_x = T_dst::tile_size_x;
    constexpr uint32_t tile_elems = tile_size_y * tile_size_x;
    constexpr uint32_t block_size_y = T_dst::block_size_y;
    constexpr uint32_t block_size_x = T_dst::block_size_x;
    constexpr uint32_t block_elems = block_size_y * block_size_x;
    constexpr int32_t num_block_x = tile_size_x / block_size_x;
    using dtype_dst = typename T_dst::dtype;
    using dtype_src = typename T_src::dtype;
    constexpr uint32_t vnni_row_src = sizeof(uint32_t) / sizeof(dtype_src);
    constexpr uint32_t vnni_row_dst = sizeof(uint32_t) / sizeof(dtype_dst);
    constexpr int32_t vnni_row
            = vnni_row_src > vnni_row_dst ? vnni_row_src : vnni_row_dst;
    static_assert(block_size_y % vnni_row == 0);
    static_assert(tile_size_y % vnni_row == 0);
    constexpr int32_t move_elems = vnni_row * block_size_x;
    xetla_vector<dtype_dst, tile_elems> reg_src
            = xetla_cvt<dtype_dst, dtype_src, tile_elems>(src.reg);
    if constexpr (sizeof(dtype_src) == sizeof(dtype_dst)) {
        dst.reg = reg_src;
        return;
    }
    xetla_vector<dtype_dst, tile_elems> reg_dst;
    constexpr uint32_t scale_factor
            = detail::gcd<vnni_row_src, vnni_row_dst>::value;
    using move_dtype = get_uint_type_t<sizeof(dtype_dst) * scale_factor>;
    constexpr uint32_t select_stride = vnni_row / scale_factor;
#pragma unroll
    for (int i = 0; i < tile_size_y / block_size_y; i++) {
#pragma unroll
        for (int j = 0; j < num_block_x; j++) {
            auto reg_src_blk = reg_src.xetla_select<block_elems, 1>(
                    (i * num_block_x + j) * block_elems);
            auto reg_dst_blk = reg_dst.xetla_select<block_elems, 1>(
                    (i * num_block_x + j) * block_elems);
            for (int row_i = 0; row_i < block_size_y; row_i += vnni_row) {
                auto reg_src_move
                        = reg_src_blk
                                  .xetla_select<move_elems, 1>(
                                          row_i * block_size_x)
                                  .xetla_format<native_type_t<move_dtype>>();
                auto reg_dst_move
                        = reg_dst_blk
                                  .xetla_select<move_elems, 1>(
                                          row_i * block_size_x)
                                  .xetla_format<native_type_t<move_dtype>>();
#pragma unroll
                for (int move_i = 0; move_i < select_stride; move_i++) {
                    if constexpr (sizeof(dtype_dst) > sizeof(dtype_src)) {
                        reg_dst_move.xetla_select<block_size_x, 1>(
                                move_i * block_size_x)
                                = reg_src_move.xetla_select<block_size_x,
                                        select_stride>(move_i);
                    } else {
                        reg_dst_move.xetla_select<block_size_x, select_stride>(
                                move_i)
                                = reg_src_move.xetla_select<block_size_x, 1>(
                                        move_i * block_size_x);
                    }
                }
            }
        }
    }
    // process the tail
    if constexpr ((tile_size_y % block_size_y) != 0) {
        constexpr int i = tile_size_y / block_size_y;
        constexpr uint32_t remain_elems_start = i * block_size_y * tile_size_x;
        constexpr uint32_t remain_size_y = tile_size_y % block_size_y;
        constexpr uint32_t remain_block_elems = remain_size_y * block_size_x;
#pragma unroll
        for (int j = 0; j < num_block_x; j++) {
            auto reg_src_blk = reg_src.xetla_select<remain_block_elems, 1>(
                    remain_elems_start + j * remain_block_elems);
            auto reg_dst_blk = reg_dst.xetla_select<remain_block_elems, 1>(
                    remain_elems_start + j * remain_block_elems);
            // for mma, here we can guarantee that the remaining is a multiple of
            // vnni_row
            for (int row_i = 0; row_i < remain_size_y; row_i += vnni_row) {
                auto reg_src_move
                        = reg_src_blk
                                  .xetla_select<move_elems, 1>(
                                          row_i * block_size_x)
                                  .xetla_format<native_type_t<move_dtype>>();
                auto reg_dst_move
                        = reg_dst_blk
                                  .xetla_select<move_elems, 1>(
                                          row_i * block_size_x)
                                  .xetla_format<native_type_t<move_dtype>>();
#pragma unroll
                for (int move_i = 0; move_i < select_stride; move_i++) {
                    if constexpr (sizeof(dtype_dst) > sizeof(dtype_src)) {
                        reg_dst_move.xetla_select<block_size_x, 1>(
                                move_i * block_size_x)
                                = reg_src_move.xetla_select<block_size_x,
                                        select_stride>(move_i);
                    } else {
                        reg_dst_move.xetla_select<block_size_x, select_stride>(
                                move_i)
                                = reg_src_move.xetla_select<block_size_x, 1>(
                                        move_i * block_size_x);
                    }
                }
            }
        }
    }
    dst.reg = reg_dst;
}

/// @brief Broadcasts 1d src tile to the entire 2d tile, as well as do the data conversion.
///
/// @tparam T_dst Is the destination tile data type.
/// @tparam T_src Is the source tile data type, interpreted as 1D data.
/// @param dst Is the reference of the destination tile object.
/// @param src Is the reference of the destination tile object.
/// @return No return, in-place update in the destination tile.
template <typename T_dst, typename T_src>
__XETLA_API
        typename std::enable_if_t<(T_dst::register_layout == reg_layout::tiled)
                && (T_src::register_layout == reg_layout::tiled)
                && (T_src::tile_size_x == T_dst::tile_size_x)
                && (T_src::tile_size_y == 1)>
        row_broadcast(T_dst &dst, T_src &src) {
    static constexpr uint32_t dst_tile_size_y = T_dst::tile_size_y;
    static constexpr uint32_t dst_tile_size_x = T_dst::tile_size_x;
    static constexpr uint32_t dst_tile_elems = T_dst::tile_elems;
    static constexpr uint32_t dst_block_size_y = T_dst::block_size_y;
    static constexpr uint32_t dst_block_size_x = T_dst::block_size_x;
    static constexpr uint32_t dst_block_elems = T_dst::block_elems;
    static constexpr int32_t dst_num_block_y = T_dst::num_block_y;
    static constexpr int32_t dst_num_block_x = T_dst::num_block_x;
    using dst_dtype = typename T_dst::dtype;
    using src_dtype = typename T_src::dtype;

#pragma unroll
    for (int i = 0; i < dst_tile_size_y / dst_block_size_y; i++) {
#pragma unroll
        for (int j = 0; j < dst_num_block_x; j++) {
            auto dst_reg
                    = (dst.reg)
                              .xetla_select<dst_block_elems, 1>(
                                      (i * dst_num_block_x + j)
                                      * dst_block_elems)
                              .xetla_format<native_type_t<dst_dtype>,
                                      dst_block_size_y, dst_block_size_x>();
#pragma unroll
            for (int row_i = 0; row_i < dst_block_size_y; row_i++) {
                auto src_reg = src.reg.xetla_select<dst_block_size_x, 1>(
                        j * dst_block_size_x);
                dst_reg.row(row_i)
                        = xetla_cvt<dst_dtype, src_dtype, dst_block_size_x>(
                                src_reg);
            }
        }
    }

    // process the tail
    if constexpr ((dst_tile_size_y % dst_block_size_y) != 0) {
        constexpr uint32_t tail_start_y
                = dst_tile_size_y / dst_block_size_y * dst_block_size_y;
        constexpr int32_t dst_tail_size_y = dst_tile_size_y % dst_block_size_y;
        constexpr int32_t dst_tail_block_elems
                = dst_tail_size_y * dst_block_size_x;
#pragma unroll
        for (int j = 0; j < dst_num_block_x; j++) {
            auto dst_reg = (dst.reg)
                                   .xetla_select<dst_tail_block_elems, 1>(
                                           tail_start_y * dst_tile_size_x
                                           + j * dst_tail_block_elems)
                                   .xetla_format<native_type_t<dst_dtype>,
                                           dst_tail_size_y, dst_block_size_x>();
#pragma unroll
            for (int row_i = 0; row_i < dst_tail_size_y; row_i++) {
                auto src_reg = src.reg.xetla_select<dst_block_size_x, 1>(
                        j * dst_block_size_x);
                dst_reg.row(row_i)
                        = xetla_cvt<dst_dtype, src_dtype, dst_block_size_x>(
                                src_reg);
            }
        }
    }
}

/// @brief convert 2d tile in a tiled register layout to a 2d tile in a linear register layout
///
/// @tparam T_dst Is the destination tile data type.
/// @tparam T_src Is the source tile data type.
/// @param dst Is the reference of the destination tile object.
/// @param src Is the reference of the destination tile object.
/// @return No return, in-place update in the destination tile.
template <typename T_dst, typename T_src>
__XETLA_API typename std::enable_if_t<(T_dst::register_layout
                                              == reg_layout::linear)
        && (T_src::register_layout == reg_layout::tiled)
        && (T_src::tile_size_x == T_dst::tile_size_x)
        && (T_src::tile_size_y == T_dst::tile_size_y)
        && (T_dst::tile_size_x == T_dst::block_size_x)
        && (T_dst::tile_size_y == T_dst::block_size_y)
        && (std::is_same<typename T_dst::dtype, typename T_src::dtype>::value)>
layout_convert(T_dst &dst, T_src &src) {
    using tile_desc = typename T_src::tile_desc;
    using dtype = typename T_dst::dtype;
    static constexpr uint32_t num_block_x = tile_desc::num_block_x;
    static constexpr uint32_t num_block_y = tile_desc::num_block_y;
    static constexpr uint32_t block_elems = tile_desc::block_elems;
    static constexpr uint32_t block_size_x = tile_desc::block_size_x;
    static constexpr uint32_t block_size_y = tile_desc::block_size_y;
    static constexpr uint32_t tile_size_x = tile_desc::tile_size_x;
    static constexpr uint32_t tile_size_y = tile_desc::tile_size_y;

    auto dst_reg = dst.reg.xetla_format<native_type_t<dtype>, tile_size_y,
            tile_size_x>();
#pragma unroll
    for (int i = 0; i < num_block_y; ++i) {
        uint32_t offset_y = i * block_size_y;
#pragma unroll
        for (int j = 0; j < num_block_x; ++j) {
            uint32_t offset_x = j * block_size_x;
            auto src_reg = src.reg.xetla_select<block_elems, 1>(
                    (i * num_block_x + j) * block_elems);
            dst_reg.xetla_select<block_size_y, 1, block_size_x, 1>(
                    offset_y, offset_x)
                    = src_reg;
        }
    }
    // process the tail
    if constexpr (tile_desc::remained_size_y > 0) {
        constexpr uint32_t remained_size_y = tile_desc::remained_size_y;
        constexpr uint32_t offset_y = tile_size_y - remained_size_y;
        constexpr uint32_t processed_elems = offset_y * tile_size_x;
        constexpr uint32_t remained_block_elems
                = remained_size_y * block_size_x;
#pragma unroll
        for (int j = 0; j < num_block_x; ++j) {
            uint32_t offset_x = j * block_size_x;
            auto src_reg = src.reg.xetla_select<remained_block_elems, 1>(
                    processed_elems + j * remained_block_elems);
            dst_reg.xetla_select<remained_size_y, 1, block_size_x, 1>(
                    offset_y, offset_x)
                    = src_reg;
        }
    }
}

/// @brief convert 2d tile in a linear register layout to a 2d tile in a tiled register layout
///
/// @tparam T_dst Is the destination tile data type.
/// @tparam T_src Is the source tile data type.
/// @param dst Is the reference of the destination tile object.
/// @param src Is the reference of the destination tile object.
/// @return No return, in-place update in the destination tile.
template <typename T_dst, typename T_src>
__XETLA_API typename std::enable_if_t<(T_dst::register_layout
                                              == reg_layout::tiled)
        && (T_src::register_layout == reg_layout::linear)
        && (T_dst::tile_size_x == T_src::tile_size_x)
        && (T_dst::tile_size_y == T_src::tile_size_y)
        && (T_src::tile_size_x == T_src::block_size_x)
        && (T_src::tile_size_y == T_src::block_size_y)
        && (std::is_same<typename T_dst::dtype, typename T_src::dtype>::value)>
layout_convert(T_dst &dst, T_src &src) {
    using tile_desc = typename T_dst::tile_desc;
    using dtype = typename T_dst::dtype;
    static constexpr uint32_t num_block_x = tile_desc::num_block_x;
    static constexpr uint32_t num_block_y = tile_desc::num_block_y;
    static constexpr uint32_t block_elems = tile_desc::block_elems;
    static constexpr uint32_t block_size_x = tile_desc::block_size_x;
    static constexpr uint32_t block_size_y = tile_desc::block_size_y;
    static constexpr uint32_t tile_size_x = tile_desc::tile_size_x;
    static constexpr uint32_t tile_size_y = tile_desc::tile_size_y;

    auto src_reg = src.reg.xetla_format<native_type_t<dtype>, tile_size_y,
            tile_size_x>();
#pragma unroll
    for (int i = 0; i < num_block_y; ++i) {
        uint32_t offset_y = i * block_size_y;
#pragma unroll
        for (int j = 0; j < num_block_x; ++j) {
            uint32_t offset_x = j * block_size_x;
            auto dst_reg = dst.reg.xetla_select<block_elems, 1>(
                    (i * num_block_x + j) * block_elems);
            dst_reg = src_reg.xetla_select<block_size_y, 1, block_size_x, 1>(
                    offset_y, offset_x);
        }
    }
    // process the tail
    if constexpr (tile_desc::remained_size_y > 0) {
        constexpr uint32_t remained_size_y = tile_desc::remained_size_y;
        constexpr uint32_t offset_y = tile_size_y - remained_size_y;
        constexpr uint32_t processed_elems = offset_y * tile_size_x;
        constexpr uint32_t remained_block_elems
                = remained_size_y * block_size_x;
#pragma unroll
        for (int j = 0; j < num_block_x; ++j) {
            uint32_t offset_x = j * block_size_x;
            auto dst_reg = dst.reg.xetla_select<remained_block_elems, 1>(
                    processed_elems + j * remained_block_elems);
            dst_reg = src_reg.xetla_select<remained_size_y, 1, block_size_x, 1>(
                    offset_y, offset_x);
        }
    }
}
} // namespace gpu::xetla::subgroup
