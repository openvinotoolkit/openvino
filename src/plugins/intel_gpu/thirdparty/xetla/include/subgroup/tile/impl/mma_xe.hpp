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

namespace gpu::xetla::subgroup {

/// @brief Is the tile mma operation functor, specialized for Xe and matrix engine.
template <typename matAcc_dst_t_, typename matAcc_src_t_, typename matB_t_,
        typename matA_t_, gpu_arch arch_tag_>
struct tile_mma_t<matAcc_dst_t_, matAcc_src_t_, matB_t_, matA_t_,
        mma_engine::xmx, arch_tag_,
        std::enable_if_t<(arch_tag_ == gpu_arch::Xe)>> {
    using matA_t = matA_t_;
    using matB_t = matB_t_;
    using matSrc_t = matAcc_src_t_;
    using matDst_t = matAcc_dst_t_;
    using dtype_a = typename matA_t::dtype;
    using dtype_b = typename matB_t::dtype;
    using dtype_src = typename matSrc_t::dtype;
    using dtype_dst = typename matDst_t::dtype;

    using mma_attr = typename arch_attr_t<arch_tag_>::mma_attr;

    static constexpr uint32_t a_tile_size_y = matA_t::tile_size_y;
    static constexpr uint32_t a_tile_size_x = matA_t::tile_size_x;
    static constexpr uint32_t a_tile_elems = matA_t::tile_elems;
    static constexpr uint32_t a_block_size_y = matA_t::block_size_y;
    static constexpr uint32_t a_block_size_x = matA_t::block_size_x;
    static constexpr uint32_t a_block_elems = matA_t::block_elems;

    static constexpr uint32_t b_tile_size_x = matB_t::tile_size_x;
    static constexpr uint32_t b_tile_size_y = matB_t::tile_size_y;
    static constexpr uint32_t b_tile_elems = matB_t::tile_elems;
    static constexpr uint32_t b_block_size_x = matB_t::block_size_x;
    static constexpr uint32_t b_block_size_y = matB_t::block_size_y;
    static constexpr uint32_t b_block_elems = matB_t::block_elems;

    static constexpr uint32_t tile_size_m = matDst_t::tile_size_y;
    static constexpr uint32_t tile_size_k = a_tile_size_x;
    static constexpr uint32_t tile_size_n = matDst_t::tile_size_x;
    static constexpr uint32_t tile_elems = tile_size_m * tile_size_n;
    static constexpr uint32_t block_size_n = matDst_t::block_size_x;
    static constexpr uint32_t block_size_k
            = a_block_size_x; //cannot use b_block_size_y
    static constexpr uint32_t block_size_m = matDst_t::block_size_y;
    static constexpr uint32_t block_elems = block_size_m * block_size_n;

    static_assert(tile_size_m == matA_t::tile_size_y,
            "matAcc tile m should match with matA tile m");
    static_assert(a_tile_size_x == b_tile_size_y,
            "matA tile k should match with matB tile k");
    static_assert(tile_size_n == matB_t::tile_size_x,
            "matAcc tile n should match with matB tile n");
    static_assert(block_size_m == a_block_size_y,
            "matAcc block m should match with matA block m");
    static_assert(block_size_n == b_block_size_x,
            "matAcc block n should match with matB block n");
    static_assert(b_block_size_y % a_block_size_x == 0,
            "matA block k should match with matB block k");
    static_assert((tile_size_k % block_size_k) == 0,
            "matAcc tile_size_k should be a multiple of block_size_k");
    static_assert((block_size_k == 32 / sizeof(dtype_a)),
            "DPAS depth only support the value of 32 / sizeof(dtype_a). "
            "Currently we don't support the "
            "splitting of block when call the DPAS");

    static constexpr int32_t num_block_n = matDst_t::num_block_x;
    static constexpr int32_t num_block_m = matDst_t::num_block_y;
    static constexpr int32_t num_block_k = tile_size_k / block_size_k;
    static constexpr int32_t num_block_mma_b = b_block_size_y / block_size_k;
    static constexpr uint32_t b_block_mma_elems
            = b_block_elems / num_block_mma_b;

    static constexpr int32_t mma_m = mma_attr::mma_m_in_elem;
    static constexpr int32_t mma_k
            = mma_attr::mma_k_in_bytes / sizeof(uint32_t);
    static_assert(tile_size_m % mma_m == 0,
            "tile_size_m shoud be a multiple of mma_m");

    __XETLA_API static void mma(
            matDst_t &dst, matSrc_t &src, matB_t &b, matA_t &a) {
        constexpr int32_t a_mma_elems = mma_m * a_block_size_x;
        constexpr int32_t c_mma_elems = mma_m * block_size_n;
#pragma unroll
        for (int j = 0; j < num_block_n; j++) {
#pragma unroll
            for (int i = 0; i < tile_size_m / block_size_m; i++) {
                auto src_block = src.reg.xetla_select<block_elems, 1>(
                        (i * num_block_n + j) * block_elems);
                auto dst_block = dst.reg.xetla_select<block_elems, 1>(
                        (i * num_block_n + j) * block_elems);
#pragma unroll
                for (int mma_i = 0; mma_i < block_size_m / mma_m; mma_i++) {
                    auto src_sub_blk = src_block.xetla_select<c_mma_elems, 1>(
                            mma_i * c_mma_elems);
                    auto dst_sub_blk = dst_block.xetla_select<c_mma_elems, 1>(
                            mma_i * c_mma_elems);
                    { //k=0
                        auto a_block = a.reg.xetla_select<a_block_elems, 1>(
                                (i * num_block_k) * a_block_elems);
                        auto a_sub_blk = a_block.xetla_select<a_mma_elems, 1>(
                                mma_i * a_mma_elems);
                        auto b_blk = b.reg.xetla_select<b_block_elems, 1>(
                                j * b_block_elems);
                        auto b_sub_blk
                                = b_blk.xetla_select<b_block_mma_elems, 1>(0);
                        dst_sub_blk = xetla_mma<
                                gpu::xetla::detail::mma_argument_type<
                                        dtype_b>(),
                                gpu::xetla::detail::mma_argument_type<
                                        dtype_a>(),
                                mma_k, mma_m, dtype_src, uint32_t, uint32_t,
                                c_mma_elems,
                                b_block_mma_elems
                                        / (sizeof(uint32_t) / sizeof(dtype_b)),
                                a_mma_elems
                                        / (sizeof(uint32_t) / sizeof(dtype_a))>(
                                src_sub_blk, b_sub_blk.xetla_format<uint32_t>(),
                                a_sub_blk.xetla_format<uint32_t>());
                    }

#pragma unroll
                    for (int k = 1; k < num_block_k; k++) {
                        auto a_block = a.reg.xetla_select<a_block_elems, 1>(
                                (i * num_block_k + k) * a_block_elems);
                        auto a_sub_blk = a_block.xetla_select<a_mma_elems, 1>(
                                mma_i * a_mma_elems);
                        int inter_k_b = k / num_block_mma_b;
                        int inner_k_b = k % num_block_mma_b;
                        auto b_blk = b.reg.xetla_select<b_block_elems, 1>(
                                (j + inter_k_b * num_block_n) * b_block_elems);
                        auto b_sub_blk
                                = b_blk.xetla_select<b_block_mma_elems, 1>(
                                        inner_k_b * b_block_mma_elems);
                        dst_sub_blk = xetla_mma<
                                gpu::xetla::detail::mma_argument_type<
                                        dtype_b>(),
                                gpu::xetla::detail::mma_argument_type<
                                        dtype_a>(),
                                mma_k, mma_m, dtype_src, uint32_t, uint32_t,
                                c_mma_elems,
                                b_block_mma_elems
                                        / (sizeof(uint32_t) / sizeof(dtype_b)),
                                a_mma_elems
                                        / (sizeof(uint32_t) / sizeof(dtype_a))>(
                                dst_sub_blk, b_sub_blk.xetla_format<uint32_t>(),
                                a_sub_blk.xetla_format<uint32_t>());
                    }
                }
            }
            if constexpr ((tile_size_m % block_size_m) != 0) {
                constexpr uint32_t tail_block_size_m
                        = tile_size_m % block_size_m;
                constexpr uint32_t tail_block_elems
                        = block_size_n * tail_block_size_m;
                constexpr uint32_t a_tail_block_elems
                        = tail_block_size_m * a_block_size_x;
                constexpr uint32_t tail_m_start
                        = tile_size_m / block_size_m * block_size_m;
                constexpr uint32_t tail_elems_start
                        = tail_m_start * tile_size_n;
                constexpr uint32_t a_tail_elems_start
                        = tail_m_start * a_tile_size_x;
                auto src_block = src.reg.xetla_select<tail_block_elems, 1>(
                        tail_elems_start + j * tail_block_elems);
                auto dst_block = dst.reg.xetla_select<tail_block_elems, 1>(
                        tail_elems_start + j * tail_block_elems);
#pragma unroll
                for (int mma_i = 0; mma_i < tail_block_size_m / mma_m;
                        mma_i++) {
                    auto src_sub_blk = src_block.xetla_select<c_mma_elems, 1>(
                            mma_i * c_mma_elems);
                    auto dst_sub_blk = dst_block.xetla_select<c_mma_elems, 1>(
                            mma_i * c_mma_elems);
                    { //k=0
                        auto a_block
                                = a.reg.xetla_select<a_tail_block_elems, 1>(
                                        a_tail_elems_start);
                        auto a_sub_blk = a_block.xetla_select<a_mma_elems, 1>(
                                mma_i * a_mma_elems);
                        auto b_blk = b.reg.xetla_select<b_block_elems, 1>(
                                j * b_block_elems);
                        auto b_sub_blk
                                = b_blk.xetla_select<b_block_mma_elems, 1>(0);

                        dst_sub_blk = xetla_mma<
                                gpu::xetla::detail::mma_argument_type<
                                        dtype_b>(),
                                gpu::xetla::detail::mma_argument_type<
                                        dtype_a>(),
                                mma_k, mma_m, dtype_src, uint32_t, uint32_t,
                                c_mma_elems,
                                b_block_mma_elems
                                        / (sizeof(uint32_t) / sizeof(dtype_b)),
                                a_mma_elems
                                        / (sizeof(uint32_t) / sizeof(dtype_a))>(
                                src_sub_blk, b_sub_blk.xetla_format<uint32_t>(),
                                a_sub_blk.xetla_format<uint32_t>());
                    }
#pragma unroll
                    for (int k = 1; k < num_block_k; k++) {
                        auto a_block
                                = a.reg.xetla_select<a_tail_block_elems, 1>(
                                        a_tail_elems_start
                                        + k * a_tail_block_elems);
                        auto a_sub_blk = a_block.xetla_select<a_mma_elems, 1>(
                                mma_i * a_mma_elems);
                        int inter_k_b = k / num_block_mma_b;
                        int inner_k_b = k % num_block_mma_b;
                        auto b_blk = b.reg.xetla_select<b_block_elems, 1>(
                                (j + inter_k_b * num_block_n) * b_block_elems);
                        auto b_sub_blk
                                = b_blk.xetla_select<b_block_mma_elems, 1>(
                                        inner_k_b * b_block_mma_elems);

                        dst_sub_blk = xetla_mma<
                                gpu::xetla::detail::mma_argument_type<
                                        dtype_b>(),
                                gpu::xetla::detail::mma_argument_type<
                                        dtype_a>(),
                                mma_k, mma_m, dtype_src, uint32_t, uint32_t,
                                c_mma_elems,
                                b_block_mma_elems
                                        / (sizeof(uint32_t) / sizeof(dtype_b)),
                                a_mma_elems
                                        / (sizeof(uint32_t) / sizeof(dtype_a))>(
                                dst_sub_blk, b_sub_blk.xetla_format<uint32_t>(),
                                a_sub_blk.xetla_format<uint32_t>());
                    }
                }
            }
        }
        if constexpr (num_block_k > 1) {
            constexpr uint32_t last_uint16_idx
                    = tile_elems * sizeof(dtype_dst) / sizeof(uint16_t) - 1;
            xetla_wait(dst.reg.xetla_format<uint16_t>()[0]);
        }
    }
};

} // namespace gpu::xetla::subgroup
