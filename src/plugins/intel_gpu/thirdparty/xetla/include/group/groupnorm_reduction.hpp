/*******************************************************************************
* Copyright (c) 2022-2024 Intel Corporation
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

#include "group/tile_shape.hpp"
#include "subgroup/subgroup.hpp"

namespace gpu::xetla::group {
/// @brief This is the groupnorm reduction, calculate sumx and sumxsq statistics. Use slm to exchange the data and atomics to calculate results.
/// @tparam tile_shape Is the group-level tile shape.
/// @tparam matAcc_t Is the input mat type.
/// @tparam mem_desc_data_t Is the memory descriptor of input data buffer. Not used for read/write.
/// @tparam mem_desc_stat_t Is the data type of sumx and sumxsq buffers. Used for calculating results with atomics.
/// @tparam arch_tag Is the HW architecture.
template <typename tile_shape, typename matAcc_t, typename mem_desc_data_t,
        typename mem_desc_stat_t, gpu_arch arch_tag, class enable = void>
struct groupnorm_reduce_t {};

/// @brief Groupnorm reduction. Specialized for 4D NHWC tile shape and Xe architecture.
template <typename tile_shape_, typename matAcc_t_, typename mem_desc_data_t_,
        typename mem_desc_stat_t_, gpu_arch arch_tag_>
struct groupnorm_reduce_t<tile_shape_, matAcc_t_, mem_desc_data_t_,
        mem_desc_stat_t_, arch_tag_,
        std::enable_if_t<(tile_shape_::dim == 4)
                && (arch_tag_ == gpu_arch::Xe)>> {
    using mem_desc_data_t = mem_desc_data_t_;
    using mem_desc_stat_t = mem_desc_stat_t_;
    using dtype_accum = typename mem_desc_stat_t::dtype;
    using tile_shape = tile_shape_;
    using matAcc_t = matAcc_t_;
    static constexpr gpu_arch arch_tag = arch_tag_;

    static_assert((mem_desc_data_t::dim == 4),
            "mem_desc_data_t needs to describe 4D buffer");
    static_assert((mem_desc_stat_t::dim == 2),
            "mem_desc_stat_t needs to describe 2D buffer");
    static_assert((mem_desc_stat_t::space == mem_space::global),
            "mem_desc_stat_t needs to describe global memory for atomic add");

    using work_group_t = typename tile_shape::work_group_t;
    static constexpr uint32_t wg_tile_n = tile_shape::wg_tile_size_n;
    static constexpr uint32_t wg_tile_p = tile_shape::wg_tile_size_p;
    static constexpr uint32_t wg_tile_q = tile_shape::wg_tile_size_q;
    static constexpr uint32_t wg_tile_k = tile_shape::wg_tile_size_k;

    static constexpr uint32_t sg_tile_n = tile_shape::sg_tile_size_n;
    static constexpr uint32_t sg_tile_p = tile_shape::sg_tile_size_p;
    static constexpr uint32_t sg_tile_q = tile_shape::sg_tile_size_q;
    static constexpr uint32_t sg_tile_k = tile_shape::sg_tile_size_k;

    static constexpr uint32_t wg_size_n = tile_shape::wg_size_n;
    static constexpr uint32_t wg_size_p = tile_shape::wg_size_p;
    static constexpr uint32_t wg_size_q = tile_shape::wg_size_q;
    static constexpr uint32_t wg_size_k = tile_shape::wg_size_k;

    using dtype = typename matAcc_t::dtype;
    static constexpr uint32_t tile_size_x = matAcc_t::tile_size_x;
    static constexpr uint32_t tile_elems = matAcc_t::tile_elems;

    static constexpr uint32_t num_cooperative_wg = work_group_t::size;
    static constexpr uint32_t atomic_threads = wg_size_n * wg_size_k;
    static constexpr uint32_t slm_size = atomic_threads != num_cooperative_wg
            ? 2 * tile_size_x * num_cooperative_wg * sizeof(dtype_accum)
            : 0;
    static constexpr uint32_t barrier_count
            = atomic_threads != num_cooperative_wg ? 1 : 0;

    using tile_desc_t = typename matAcc_t::tile_desc;
    using accum_tile_t = subgroup::tile_t<dtype_accum, tile_desc_t>;

    using reduced_tile_desc_t = subgroup::tile_desc_t<matAcc_t::tile_size_x, 1,
            matAcc_t::block_size_x, 1>;
    using reduced_tile_t = subgroup::tile_t<dtype_accum, reduced_tile_desc_t>;
    using local_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_accum, mem_layout::row_major, mem_space::local>,
            reduced_tile_desc_t, msg_type::block_1d, arch_tag>;

    /// @brief Groupnorm reduction arguments.
    /// @note input leading dimension (channel count) should be divisible by group_size and
    /// correspond to sumx, sumxsq leading dimension (group count)
    struct arguments_t {
        mem_desc_data_t mem_desc_in;
        mem_desc_stat_t mem_desc_sumx;
        mem_desc_stat_t mem_desc_sumxsq;
        uint32_t group_size;
        inline arguments_t()
            : mem_desc_in(nullptr, {0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0})
            , mem_desc_sumx(nullptr, {0, 0, 0}, {0, 0})
            , mem_desc_sumxsq(nullptr, {0, 0, 0}, {0, 0}) {}
        inline arguments_t(const mem_desc_data_t &mem_desc_in_,
                const mem_desc_stat_t &mem_desc_sumx_,
                const mem_desc_stat_t &mem_desc_sumxsq_, uint32_t group_size_)
            : mem_desc_in(mem_desc_in_)
            , mem_desc_sumx(mem_desc_sumx_)
            , mem_desc_sumxsq(mem_desc_sumxsq_)
            , group_size(group_size_) {}
    };

    /// @brief Update memory descriptor coordinate based on the tid.
    /// @param sg_idk [in] id of subgroup tile in dimension k.
    /// @param sg_idq [in] id of subgroup tile in dimension q.
    /// @param sg_idp [in] id of subgroup tile in dimension p.
    /// @param sg_idn [in] id of subgroup tile in dimension n.
    /// @param args [in|out] includes memory descriptors.
    __XETLA_API static void update_mem_desc(int32_t sg_idk, int32_t sg_idq,
            int32_t sg_idp, int32_t sg_idn, arguments_t &args) {
        int32_t tile_offset_n = sg_idn * sg_tile_n;
        int32_t tile_offset_p = sg_idp * sg_tile_p;
        int32_t tile_offset_q = sg_idq * sg_tile_q;
        int32_t tile_offset_k = sg_idk * sg_tile_k;
        args.mem_desc_in.update_coord(
                tile_offset_k, tile_offset_q, tile_offset_p, tile_offset_n);

        int32_t group_offset = args.mem_desc_in.coord.x / args.group_size;
        int32_t batch_offset = args.mem_desc_in.coord.w;
        args.mem_desc_sumx.set_coord(group_offset, batch_offset);
        args.mem_desc_sumxsq.set_coord(group_offset, batch_offset);
    }

    /// @brief Masks all tile elements which are out of bound in q dimension.
    /// @param mem_desc [in] memory descriptor to determine oob elements.
    /// @param sumx [in|out] sumx tile to mask.
    /// @param sumxsq [in|out] sumxsq tile to mask.
    __XETLA_API void mask_oob_qdim(mem_desc_data_t &mem_desc,
            accum_tile_t &sumx, accum_tile_t &sumxsq) {
        static constexpr uint32_t tile_size_x = accum_tile_t::tile_size_x;
        static constexpr uint32_t tile_size_y = accum_tile_t::tile_size_y;
        static constexpr uint32_t tile_elems = accum_tile_t::tile_elems;
        static constexpr uint32_t block_size_x = accum_tile_t::block_size_x;
        static constexpr uint32_t block_size_y = accum_tile_t::block_size_y;
        static constexpr uint32_t num_block_x = accum_tile_t::num_block_x;
        static constexpr uint32_t num_block_y = accum_tile_t::num_block_y;
        static constexpr uint32_t block_elems = accum_tile_t::block_elems;
        static constexpr uint32_t remained_size_y = tile_size_y % block_size_y;
#pragma unroll
        for (uint32_t i = 0; i < num_block_x; ++i) {
#pragma unroll
            for (uint32_t j = 0; j < num_block_y; ++j) {
                auto sumx_block = sumx.reg.xetla_select<block_elems, 1>(
                        (j * num_block_x + i) * block_elems);
                auto sumxsq_block = sumxsq.reg.xetla_select<block_elems, 1>(
                        (j * num_block_x + i) * block_elems);
#pragma unroll
                for (uint32_t k = 0; k < block_size_y; ++k) {
                    auto sumx_row = sumx_block.xetla_select<block_size_x, 1>(
                            k * block_size_x);
                    auto sumxsq_row
                            = sumxsq_block.xetla_select<block_size_x, 1>(
                                    k * block_size_x);
                    int y_off = mem_desc.get_coord_y() + j * block_size_y + k;
                    int mask = static_cast<int>(
                            (y_off >= 0) && (y_off < mem_desc.get_shape_y()));
                    sumx_row *= mask;
                    sumxsq_row *= mask;
                }
            }
        }
    }

    /// @brief Groupnorm reduction.
    /// @param g Is the workgroup of the current tile.
    /// @param matAcc Is the input of the reduction.
    /// @param args Is the additional arguments for groupnorm reduction.
    /// @param slm_base Is the slm base address.
    /// @param nbarrier_id Is the named barrier id.
    __XETLA_API KERNEL_FUNC void operator()(work_group_t &g,
            matAcc_t matAcc[sg_tile_n][sg_tile_p], arguments_t args,
            uint32_t slm_base = 0, uint32_t nbarrier_id = 0) {
        int32_t sg_idk = g.get_id() % wg_size_k;
        int32_t sg_idq = (g.get_id() / wg_size_k) % wg_size_q;
        int32_t sg_idp = (g.get_id() / (wg_size_k * wg_size_q)) % wg_size_p;
        int32_t sg_idn = (g.get_id() / (wg_size_k * wg_size_q * wg_size_p))
                % wg_size_n;

        update_mem_desc(sg_idk, sg_idq, sg_idp, sg_idn, args);
        const auto &coord_in = args.mem_desc_in.coord;
        const auto &coord_stat = args.mem_desc_sumx.coord;
        // Only first threads in p,q dim do atomics
        auto is_atomic_thread = (sg_idp == 0) && (sg_idq == 0);
#pragma unroll
        for (uint32_t n = 0; n < sg_tile_n; ++n) {
            accum_tile_t sumx(0);
            accum_tile_t sumxsq(0);

            // Reduce image spatial dimensions
#pragma unroll
            for (uint32_t p = 0; p < sg_tile_p; ++p) {
                int oob_flag = args.mem_desc_in.get_mask_from_z(p);
                xetla_vector<dtype_accum, tile_elems> tmp = oob_flag
                        * xetla_cvt<dtype_accum, dtype, tile_elems>(
                                matAcc[n][p].reg);
                sumx.reg += tmp;
                sumxsq.reg += tmp * tmp;
            }
            mask_oob_qdim(args.mem_desc_in, sumx, sumxsq);
            auto sumx_channels = subgroup::tile_reduce<reduce_op::sum,
                    dtype_accum, dtype_accum, 0>(sumx);
            auto sumxsq_channels = subgroup::tile_reduce<reduce_op::sum,
                    dtype_accum, dtype_accum, 0>(sumxsq);

            // Reduce channels into groups
            reduced_tile_t sumx_grouped(0);
            reduced_tile_t sumxsq_grouped(0);
            auto current_group = 0;
            auto next_group_start
                    = (coord_stat.x + current_group + 1) * args.group_size;
#pragma unroll
            for (uint32_t i = 0; i < tile_size_x; ++i) {
                if (coord_in.x + i >= next_group_start) {
                    ++current_group;
                    next_group_start = (coord_stat.x + current_group + 1)
                            * args.group_size;
                }
                sumx_grouped.reg.xetla_select<1, 1>(current_group)
                        += sumx_channels.xetla_select<1, 1>(i);
                sumxsq_grouped.reg.xetla_select<1, 1>(current_group)
                        += sumxsq_channels.xetla_select<1, 1>(i);
            }
            if constexpr (wg_size_p * wg_size_q > 1) {
                // Reduce image spatial dimensions across threads
                local_payload_t local_st_payload(slm_base, tile_size_x,
                        2 * num_cooperative_wg, tile_size_x, 0, g.get_id());
                subgroup::tile_store(sumx_grouped, local_st_payload);
                local_st_payload.template update_tdesc<tdesc_update_dir::y_dir>(
                        num_cooperative_wg);
                subgroup::tile_store(sumxsq_grouped, local_st_payload);

                xetla_nbarrier_t<num_cooperative_wg, num_cooperative_wg,
                        arch_tag>
                        nbarrier;
                nbarrier.init_nbarrier(
                        nbarrier_id, nbarrier_role::producer_consumer);
                xetla_fence<memory_kind::shared_local>();
                nbarrier.arrive();
                nbarrier.wait();
                if (is_atomic_thread) {
                    local_payload_t local_ld_sumx_payload(slm_base, tile_size_x,
                            2 * num_cooperative_wg, tile_size_x, 0, g.get_id());
                    local_payload_t local_ld_sumxsq_payload(slm_base,
                            tile_size_x, 2 * num_cooperative_wg, tile_size_x, 0,
                            g.get_id() + num_cooperative_wg);
#pragma unroll
                    for (uint32_t j = 1; j < wg_size_p * wg_size_q; ++j) {
                        local_ld_sumx_payload
                                .template update_tdesc<tdesc_update_dir::y_dir>(
                                        wg_size_k);
                        local_ld_sumxsq_payload
                                .template update_tdesc<tdesc_update_dir::y_dir>(
                                        wg_size_k);
                        reduced_tile_t tmp_sumx;
                        reduced_tile_t tmp_sumxsq;
                        subgroup::tile_load(tmp_sumx, local_ld_sumx_payload);
                        subgroup::tile_load(
                                tmp_sumxsq, local_ld_sumxsq_payload);
                        sumx_grouped.reg += tmp_sumx.reg;
                        sumxsq_grouped.reg += tmp_sumxsq.reg;
                    }
                }
                nbarrier.arrive();
                nbarrier.wait();
            }

            if (is_atomic_thread) {
                // Global atomic reduce
                xetla_mask<tile_size_x> atomic_pred(1);
                auto atomic_offset
                        = xetla_vector_gen<uint32_t, tile_size_x>(0, 1);
                atomic_offset += coord_stat.x;
                atomic_offset
                        += (coord_stat.y + n) * args.mem_desc_sumx.shape.x;
                atomic_offset *= sizeof(dtype_accum);
                xetla_atomic_global<atomic_op::fadd, dtype_accum, tile_size_x,
                        data_size::default_size, cache_hint::uncached,
                        cache_hint::write_back>(args.mem_desc_sumx.base.base,
                        atomic_offset, sumx_grouped.reg, atomic_pred);
                xetla_atomic_global<atomic_op::fadd, dtype_accum, tile_size_x,
                        data_size::default_size, cache_hint::uncached,
                        cache_hint::write_back>(args.mem_desc_sumxsq.base.base,
                        atomic_offset, sumxsq_grouped.reg, atomic_pred);
            }
        }
    }
};
} // namespace gpu::xetla::group
