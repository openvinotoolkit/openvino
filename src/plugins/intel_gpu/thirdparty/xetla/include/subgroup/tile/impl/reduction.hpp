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

// dim=0 : reduce along y dir;
// dim=1 : reduce along x dir;
template <reduce_op reduce_kind, typename dtype_out, typename dtype_acc,
        int dim, typename mat_t>
__XETLA_API typename std::enable_if_t<(dim == 1),
        xetla_vector<dtype_out, mat_t::tile_size_y>>
tile_reduce(mat_t &src) {
    static constexpr uint32_t tile_size_y = mat_t::tile_size_y;
    static constexpr uint32_t tile_size_x = mat_t::tile_size_x;
    static constexpr uint32_t tile_elems = mat_t::tile_elems;
    static constexpr uint32_t block_size_y = mat_t::block_size_y;
    static constexpr uint32_t block_size_x = mat_t::block_size_x;
    static constexpr uint32_t block_elems = mat_t::block_elems;
    static constexpr int32_t num_block_y = mat_t::num_block_y;
    static constexpr int32_t num_block_x = mat_t::num_block_x;
    using dtype = typename mat_t::dtype;
    /// The idea is
    /// 1) allocate a temp buffer;
    /// 2) reduce the entire tile into temp buffer;
    /// 3) reduce within temp buffer
    xetla_vector<dtype_acc, tile_size_y * block_size_x> acc;
#pragma unroll
    for (int i = 0; i < tile_size_y / block_size_y; i++) {
        //j=0, initial the buffer
        {
            auto src_reg = (src.reg).xetla_select<block_elems, 1>(
                    (i * num_block_x) * block_elems);
            auto src_reg_acc
                    = xetla_cvt<dtype_acc, dtype, block_elems>(src_reg);
            auto dst_reg_acc
                    = acc.xetla_select<block_elems, 1>(i * block_elems);
            dst_reg_acc = src_reg_acc;
        }
#pragma unroll
        for (int j = 1; j < num_block_x; j++) {
            auto src_reg = (src.reg).xetla_select<block_elems, 1>(
                    (i * num_block_x + j) * block_elems);
            auto src_reg_acc
                    = xetla_cvt<dtype_acc, dtype, block_elems>(src_reg);
            auto dst_reg_acc
                    = acc.xetla_select<block_elems, 1>(i * block_elems);
            dst_reg_acc = reduce_helper<reduce_kind, dtype_acc, block_elems>(
                    dst_reg_acc, src_reg_acc);
        }
    }

    // process the tail
    if constexpr ((tile_size_y % block_size_y) != 0) {
        constexpr uint32_t tail_start_y
                = tile_size_y / block_size_y * block_size_y;
        constexpr uint32_t tail_size_y = tile_size_y % block_size_y;
        constexpr uint32_t tail_block_elems = tail_size_y * block_size_x;
        //j=0, initial the buffer
        {
            auto src_reg = (src.reg).xetla_select<tail_block_elems, 1>(
                    tail_start_y * tile_size_x);
            auto src_reg_acc
                    = xetla_cvt<dtype_acc, dtype, tail_block_elems>(src_reg);
            auto dst_reg_acc = acc.xetla_select<tail_block_elems, 1>(
                    tail_start_y * block_size_x);
            dst_reg_acc = src_reg_acc;
        }
#pragma unroll
        for (int j = 1; j < num_block_x; j++) {
            auto src_reg = (src.reg).xetla_select<tail_block_elems, 1>(
                    tail_start_y * tile_size_x + j * tail_block_elems);
            auto src_reg_acc
                    = xetla_cvt<dtype_acc, dtype, tail_block_elems>(src_reg);
            auto dst_reg_acc = acc.xetla_select<tail_block_elems, 1>(
                    tail_start_y * block_size_x);
            dst_reg_acc
                    = reduce_helper<reduce_kind, dtype_acc, tail_block_elems>(
                            dst_reg_acc, src_reg_acc);
        }
    }

    xetla_vector<dtype_acc, tile_size_y> out = recur_col_reduce<reduce_kind,
            dtype_acc, block_size_x, tile_size_y>(acc);

    return xetla_cvt<dtype_out, dtype_acc, tile_size_y>(out);
}

template <reduce_op reduce_kind, typename dtype_out, typename dtype_acc,
        int dim, typename mat_t>
__XETLA_API typename std::enable_if_t<(dim == 0),
        xetla_vector<dtype_out, mat_t::tile_size_x>>
tile_reduce(mat_t &src) {
    static constexpr uint32_t tile_size_y = mat_t::tile_size_y;
    static constexpr uint32_t tile_size_x = mat_t::tile_size_x;
    static constexpr uint32_t tile_elems = mat_t::tile_elems;
    static constexpr uint32_t block_size_y = mat_t::block_size_y;
    static constexpr uint32_t block_size_x = mat_t::block_size_x;
    static constexpr uint32_t block_elems = mat_t::block_elems;
    static constexpr int32_t num_block_y = mat_t::num_block_y;
    static constexpr int32_t num_block_x = mat_t::num_block_x;
    using dtype = typename mat_t::dtype;
    static constexpr uint32_t num_acc = 8;
    static constexpr uint32_t first_block_size_y
            = (tile_size_y / block_size_y == 0) ? (tile_size_y % block_size_y)
                                                : block_size_y;
    static constexpr uint32_t acc_size_y
            = (num_acc > first_block_size_y) ? first_block_size_y : num_acc;
    /// The idea is
    /// 1) allocate a temp buffer;
    /// 2) reduce the entire tile into temp buffer;
    /// 3) reduce within temp buffer
    /// This will introduce additional instructions to initialize the temp
    /// buffer, but will have more parallelism
    static constexpr uint32_t first_block_elems
            = first_block_size_y * block_size_x;
    static constexpr uint32_t acc_block_elems = acc_size_y * block_size_x;
    xetla_vector<dtype_acc, acc_size_y * tile_size_x> acc;
#pragma unroll
    for (int j = 0; j < num_block_x; j++) {
        auto src_reg = (src.reg).xetla_select<first_block_elems, 1>(
                j * first_block_elems);
        auto src_reg_acc
                = xetla_cvt<dtype_acc, dtype, first_block_elems>(src_reg);
        acc.xetla_select<acc_block_elems, 1>(j * acc_block_elems)
                = src_reg_acc.xetla_select<acc_block_elems, 1>(0);
    }

#pragma unroll
    for (int i = 0; i < tile_size_y / block_size_y; i++) {
#pragma unroll
        for (int j = 0; j < num_block_x; j++) {
            auto src_reg = (src.reg).xetla_select<block_elems, 1>(
                    (i * num_block_x + j) * block_elems);
            auto src_reg_acc
                    = xetla_cvt<dtype_acc, dtype, block_elems>(src_reg);
            auto dst_reg_acc
                    = acc.xetla_select<acc_block_elems, 1>(j * acc_block_elems);
#pragma unroll
            for (int row_i = 0; row_i < block_size_y / acc_size_y; row_i++) {
                if (i == 0 && row_i == 0) continue;
                dst_reg_acc = reduce_helper<reduce_kind, dtype_acc,
                        acc_block_elems>(dst_reg_acc,
                        src_reg_acc.xetla_select<acc_block_elems, 1>(
                                row_i * acc_block_elems));
            }
            // process the tail
            if constexpr ((block_size_y % acc_size_y) != 0) {
                constexpr uint32_t acc_tail_start_y
                        = block_size_y / acc_size_y * acc_size_y;
                constexpr uint32_t acc_tail_size_y = block_size_y % acc_size_y;
                constexpr uint32_t acc_tail_block_elems
                        = acc_tail_size_y * block_size_x;
                auto dst_reg_acc_tail
                        = dst_reg_acc.xetla_select<acc_tail_block_elems>(0);
                dst_reg_acc_tail = reduce_helper<reduce_kind, dtype_acc,
                        acc_tail_block_elems>(dst_reg_acc_tail,
                        src_reg_acc.xetla_select<acc_tail_block_elems, 1>(
                                acc_tail_start_y * block_size_x));
            }
        }
    }

    // process the tail
    if constexpr ((tile_size_y % block_size_y) != 0) {
        constexpr uint32_t tail_start_y
                = tile_size_y / block_size_y * block_size_y;
        constexpr uint32_t tail_size_y = tile_size_y % block_size_y;
        constexpr uint32_t tail_block_elems = tail_size_y * block_size_x;
#pragma unroll
        for (int j = 0; j < num_block_x; j++) {
            auto src_reg = (src.reg).xetla_select<tail_block_elems, 1>(
                    tail_start_y * tile_size_x + j * tail_block_elems);
            auto src_reg_acc
                    = xetla_cvt<dtype_acc, dtype, tail_block_elems>(src_reg);
            auto dst_reg_acc
                    = acc.xetla_select<acc_block_elems, 1>(j * acc_block_elems);
#pragma unroll
            for (int row_i = 0; row_i < tail_size_y / acc_size_y; row_i++) {
                if ((tile_size_y / block_size_y == 0) && row_i == 0) continue;
                dst_reg_acc = reduce_helper<reduce_kind, dtype_acc,
                        acc_block_elems>(dst_reg_acc,
                        src_reg_acc.xetla_select<acc_block_elems, 1>(
                                row_i * acc_block_elems));
            }
            // process the tail
            if constexpr ((tail_size_y % acc_size_y) != 0) {
                constexpr uint32_t acc_tail_start_y
                        = tail_size_y / acc_size_y * acc_size_y;
                constexpr uint32_t acc_tail_size_y = tail_size_y % acc_size_y;
                constexpr uint32_t acc_tail_block_elems
                        = acc_tail_size_y * block_size_x;
                auto dst_reg_acc_tail
                        = dst_reg_acc.xetla_select<acc_tail_block_elems>(0);
                dst_reg_acc_tail = reduce_helper<reduce_kind, dtype_acc,
                        acc_tail_block_elems>(dst_reg_acc_tail,
                        src_reg_acc.xetla_select<acc_tail_block_elems, 1>(
                                acc_tail_start_y * block_size_x));
            }
        }
    }

    xetla_vector<dtype_acc, tile_size_x> out;
#pragma unroll
    for (int i = 0; i < acc_size_y; i++) {
#pragma unroll
        for (int j = 0; j < num_block_x; j++) {
            auto reg_acc
                    = acc.xetla_select<acc_block_elems, 1>(j * acc_block_elems);
            auto reg_acc_2d = reg_acc.xetla_format<dtype_acc, acc_size_y,
                    block_size_x>();
            if (i == 0) {
                out.xetla_select<block_size_x, 1>(j * block_size_x)
                        = reg_acc_2d.row(i);
            } else {
                out.xetla_select<block_size_x, 1>(j * block_size_x)
                        = reduce_helper<reduce_kind, dtype_acc, block_size_x>(
                                out.xetla_select<block_size_x, 1>(
                                        j * block_size_x),
                                reg_acc_2d.row(i));
            }
        }
    }

    return xetla_cvt<dtype_out, dtype_acc, tile_size_x>(out);
}

/// @brief Reduce 2d src tile to the 1d tile, and output to 1d dst.
///
/// @tparam T_dst Is the destination tile data type, interpreted as 1d data.
/// @tparam T_src Is the source tile data type, interpreted as 2d data.
/// @tparam accumulate is to accumulate the old value or not.
/// @tparam dtype_acc Is the accumulation data type, src ==> convert to
/// dtype_acc ==> reduction + accumulation ==>  convert to dtype_dst.
/// @param dst Is the reference of the destination tile object.
/// @param src Is the reference of the destination tile object.
/// @return No return, in-place update in the destination tile.
/// @note This is only for reduce add, and will be deprecated in future. Please use tile_reduce instead.
template <typename T_dst, typename T_src, bool accumulate = true,
        typename dtype_acc = float, uint32_t num_acc = 4>
XETLA_MARKER(
        "This is only for reduce add, and will be deprecated in future. "
        "Please use tile_reduce instead.")
__XETLA_API
        typename std::enable_if_t<(T_dst::register_layout == reg_layout::tiled)
                && (T_src::register_layout == reg_layout::tiled)
                && (T_dst::tile_size_x == T_src::tile_size_x)
                && (T_dst::tile_size_y == 1)> tile_row_reduce(T_dst &dst,
                T_src &src) {
    static constexpr uint32_t tile_size_y = T_src::tile_size_y;
    static constexpr uint32_t tile_size_x = T_src::tile_size_x;
    static constexpr uint32_t tile_elems = T_src::tile_elems;
    static constexpr uint32_t block_size_y = T_src::block_size_y;
    static constexpr uint32_t block_size_x = T_src::block_size_x;
    static constexpr uint32_t block_elems = T_src::block_elems;
    static constexpr int32_t num_block_y = T_src::num_block_y;
    static constexpr int32_t num_block_x = T_src::num_block_x;
    using dtype_dst = typename T_dst::dtype;
    using dtype_src = typename T_src::dtype;
    /// Here we rely on compiler to generate mixed mode for bf16
    static constexpr uint32_t SIMD = 64 / sizeof(dtype_acc);
    static constexpr uint32_t accum_len
            = ((block_size_x % SIMD) && (sizeof(dtype_src) < 4)) == 0
            ? SIMD
            : block_size_x;
    static constexpr uint32_t accum_elems = accum_len * block_size_y;
    /// The idea is
    /// 1) allocate a temp buffer;
    /// 2) accumulate the entire tile into temp buffer;
    /// 3) reduce within temp buffer
    /// This will introduce additional instructions to initialize the temp
    /// buffer, but will have more parallelism
    xetla_vector<dtype_acc, tile_size_x * num_acc> acc(0);
    auto acc_2d = acc.xetla_format<dtype_acc, num_acc, tile_size_x>();
    if constexpr (accumulate) {
        acc_2d.row(0) = xetla_cvt<dtype_acc, dtype_dst, tile_size_x>(dst.reg);
    }
#pragma unroll
    for (int i = 0; i < tile_size_y / block_size_y; i++) {
#pragma unroll
        for (int j = 0; j < num_block_x; j++) {
            auto src_reg = (src.reg).xetla_select<block_elems, 1>(
                    (i * num_block_x + j) * block_elems);
            auto src_reg_dtype_acc
                    = xetla_cvt<dtype_acc, dtype_src, block_elems>(src_reg);
            auto src_reg_2d = src_reg_dtype_acc.xetla_format<dtype_acc,
                    block_size_y, block_size_x>();
#pragma unroll
            for (int row_i = 0; row_i < block_size_y; row_i += num_acc) {
#pragma unroll
                for (int acc_i = 0;
                        (acc_i < num_acc) && (row_i + acc_i < block_size_y);
                        acc_i++) {
                    auto acc_sub = acc.xetla_select<block_size_x, 1>(
                            acc_i * tile_size_x + j * block_size_x);
#pragma unroll
                    for (int k = 0; k < block_size_x / accum_len; k++) {
                        acc_sub.xetla_select<accum_len, 1>(k * accum_len)
                                = acc_sub.xetla_select<accum_len, 1>(
                                          k * accum_len)
                                + src_reg_2d.row(row_i + acc_i)
                                          .xetla_select<accum_len, 1>(
                                                  k * accum_len);
                    }
                }
            }
        }
    }

    // process the tail
    if constexpr ((tile_size_y % block_size_y) != 0) {
        constexpr uint32_t tail_start_y
                = tile_size_y / block_size_y * block_size_y;
        constexpr uint32_t tail_size_y = tile_size_y % block_size_y;
        constexpr uint32_t tail_block_elems = tail_size_y * block_size_x;
#pragma unroll
        for (int j = 0; j < num_block_x; j++) {
            auto src_reg = (src.reg).xetla_select<tail_block_elems, 1>(
                    tail_start_y * tile_size_x + j * tail_block_elems);
            auto src_reg_dtype_acc
                    = xetla_cvt<dtype_acc, dtype_src, tail_block_elems>(
                            src_reg);
            auto src_reg_2d = src_reg_dtype_acc.xetla_format<dtype_acc,
                    tail_size_y, block_size_x>();
#pragma unroll
            for (int row_i = 0; row_i < tail_size_y; row_i += num_acc) {
#pragma unroll
                for (int acc_i = 0;
                        (acc_i < num_acc) && (row_i + acc_i < tail_size_y);
                        acc_i++) {
                    auto acc_sub = acc.xetla_select<block_size_x, 1>(
                            acc_i * tile_size_x + j * block_size_x);
#pragma unroll
                    for (int k = 0; k < block_size_x / accum_len; k++) {
                        acc_sub.xetla_select<accum_len, 1>(k * accum_len)
                                = acc_sub.xetla_select<accum_len, 1>(
                                          k * accum_len)
                                + src_reg_2d.row(row_i + acc_i)
                                          .xetla_select<accum_len, 1>(
                                                  k * accum_len);
                    }
                }
            }
        }
    }

#pragma unroll
    for (int i = 1; i < num_acc; i++) {
        acc_2d.row(0) += acc_2d.row(i);
    }

    dst.reg = xetla_cvt<dtype_dst, dtype_acc, tile_size_x>(acc_2d.row(0));
}

} // namespace gpu::xetla::subgroup