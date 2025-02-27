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

#include "group/tile_shape.hpp"
#include "subgroup/subgroup.hpp"

namespace gpu::xetla::group {

/// @brief Workgroups to do the cooperative reduction.
/// @tparam reduce_kind Is the reduction type.
/// @tparam tile_shape Is the group-level tile shape.
/// @tparam matAcc_t Is the input mat type.
/// @tparam num_cooperative_wg Is the number of workgroups to do the cooperation.
/// @tparam arch_tag Is the HW architecture.
template <reduce_op reduce_kind, typename tile_shape, typename matAcc_t,
        uint32_t num_cooperative_wg, gpu_arch arch_tag,
        mem_space space_ = mem_space::local, class enable = void>
class cooperative_reduce_t {};

/// @brief Workgroups to do the cooperative reduction. Specialized for Xe architecture.
template <reduce_op reduce_kind, typename tile_shape_, typename matAcc_t,
        uint32_t num_cooperative_wg, gpu_arch arch_tag_>
class cooperative_reduce_t<reduce_kind, tile_shape_, matAcc_t,
        num_cooperative_wg, arch_tag_, mem_space::local,
        std::enable_if_t<((arch_tag_ == gpu_arch::Xe))
                && (tile_shape_::dim == 2)>> {
public:
    static constexpr gpu_arch arch_tag = arch_tag_;
    using tile_shape = tile_shape_;
    using dtype = typename matAcc_t::dtype;

private:
    static constexpr uint32_t sg_tile_m = tile_shape::sg_tile_size_y;
    static constexpr uint32_t sg_tile_n = tile_shape::sg_tile_size_x;
    static constexpr uint32_t wg_size_x = tile_shape::wg_size_x;
    static constexpr uint32_t wg_size_y = tile_shape::wg_size_y;
    static constexpr uint32_t real_wg_tile_m = sg_tile_m * wg_size_y;
    static constexpr uint32_t real_wg_tile_n = sg_tile_n * wg_size_x;

    static constexpr uint32_t wg_tile_size
            = real_wg_tile_m * real_wg_tile_n * sizeof(dtype);
    using work_group_t = typename tile_shape::work_group_t;
    static constexpr uint32_t work_group_size = work_group_t::size;
    // cooperative split, y dir first
    static_assert((num_cooperative_wg & (num_cooperative_wg - 1)) == 0,
            "num_cooperative_wg should be power of 2");

public:
    static constexpr uint32_t coop_num_y
            = gpu::xetla::subgroup::detail::gcd<num_cooperative_wg,
                    sg_tile_m>::value;
    static constexpr uint32_t coop_remain_num_x
            = num_cooperative_wg / coop_num_y;
    static constexpr bool has_redundant_wg
            = (coop_remain_num_x * 16) > sg_tile_n;
    static constexpr uint32_t tile_size_y = sg_tile_m / coop_num_y;
    static constexpr uint32_t tile_size_x
            = has_redundant_wg ? 16 : sg_tile_n / coop_remain_num_x;
    static constexpr uint32_t coop_num_x = sg_tile_n / tile_size_x;
    static constexpr uint32_t num_reduce_wg = coop_num_x * coop_num_y;

private:
    static constexpr uint32_t src_block_size_x = matAcc_t::block_size_x;
    static constexpr uint32_t src_block_size_y = matAcc_t::block_size_y;

    static constexpr uint32_t block_size_x
            = gpu::xetla::subgroup::detail::gcd<tile_size_x,
                    src_block_size_x>::value;
    static constexpr uint32_t block_size_y
            = (tile_size_y > src_block_size_y) ? src_block_size_y : tile_size_y;

    using local_st_tile_desc_t = subgroup::tile_desc_t<sg_tile_n, sg_tile_m,
            src_block_size_x, src_block_size_y, reg_layout::tiled>;
    using local_st_tile_t = subgroup::tile_t<dtype, local_st_tile_desc_t>;
    using local_st_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype, mem_layout::row_major, mem_space::local>,
            local_st_tile_desc_t,
            subgroup::msg_type_v<local_st_tile_desc_t, mem_space::local>,
            arch_tag>;
    using local_ld_tile_desc_t = subgroup::tile_desc_t<tile_size_x, tile_size_y,
            block_size_x, block_size_y, reg_layout::tiled>;
    using local_ld_tile_t = subgroup::tile_t<dtype, local_ld_tile_desc_t>;
    using local_ld_payload_t = subgroup::mem_payload_t<
            mem_desc_t<dtype, mem_layout::row_major, mem_space::local>,
            local_ld_tile_desc_t,
            subgroup::msg_type_v<local_ld_tile_desc_t, mem_space::local>,
            arch_tag>;

public:
    using mat_slice_t = subgroup::tile_t<dtype, local_ld_tile_desc_t>;

    static constexpr uint32_t barrier_count = work_group_size;
    static constexpr uint32_t slm_size = wg_tile_size * num_cooperative_wg;

    uint32_t coop_id;
    uint32_t coop_id_x;
    uint32_t coop_id_y;
    inline cooperative_reduce_t(uint32_t coop_id_) : coop_id(coop_id_) {
        coop_id_x = coop_id % coop_remain_num_x;
        coop_id_y = coop_id / coop_remain_num_x;
    }
    inline bool is_valid_post_process_wg() { return coop_id_x < coop_num_x; }

    /// @brief Cooperative workgroup reduction.
    /// 1) each workgroup stores tile data to local memory ->
    /// 2) cross workgroup (but still within a group) sync ->
    /// 3) workgroups loads slice of tile data, do the reduction.
    /// @note only workgroups with coop_id_x < coop_num_x have valid data.
    /// @param g Is the workgroup of the current tile.
    /// @param mat_slice Is the output of the reduction. Each workgroup only keeps part of the tile data.
    /// @param matAcc Is the input of the reduction.
    /// @param slm_base Is the slm base address.
    /// @param nbarrier_base Is the named barrier base.
    inline KERNEL_FUNC void operator()(work_group_t &g, mat_slice_t &mat_slice,
            matAcc_t &matAcc, uint32_t slm_base = 0,
            uint32_t nbarrier_base = 0) {
        uint32_t sg_idx = g.get_id() % wg_size_x;
        uint32_t sg_idy = g.get_id() / wg_size_x;

        int32_t slm_store_offset_x = sg_idx * sg_tile_n;
        int32_t slm_store_offset_y
                = coop_id * real_wg_tile_m + sg_idy * sg_tile_m;
        local_st_tile_t local_st;
        local_st_payload_t local_st_payload(slm_base, real_wg_tile_n,
                real_wg_tile_m * num_cooperative_wg, real_wg_tile_n,
                slm_store_offset_x, slm_store_offset_y);
        local_st.reg = matAcc.reg;
        tile_store(local_st, local_st_payload);

        xetla_nbarrier_t<num_cooperative_wg, num_cooperative_wg, arch_tag>
                nbarrier;
        uint32_t nbar_id = nbarrier_base + g.get_id();
        nbarrier.init_nbarrier(nbar_id, nbarrier_role::producer_consumer);
        xetla_fence<memory_kind::shared_local>();
        nbarrier.arrive();
        nbarrier.wait();

        if (is_valid_post_process_wg()) {
            // nbarrier.init_nbarrier(nbar_id, nbarrier_role::consumer);
            // nbarrier.arrive();
            int32_t slm_load_offset_x
                    = sg_idx * sg_tile_n + coop_id_x * tile_size_x;
            int32_t slm_load_offset_y
                    = sg_idy * sg_tile_m + coop_id_y * tile_size_y;

            local_ld_tile_t local_ld;
            local_ld_payload_t local_ld_payload(slm_base, real_wg_tile_n,
                    real_wg_tile_m * num_cooperative_wg, real_wg_tile_n,
                    slm_load_offset_x, slm_load_offset_y);

            tile_load(local_ld, local_ld_payload);
            mat_slice.reg = local_ld.reg;
#pragma unroll
            for (int i = 1; i < num_cooperative_wg; i++) {
                local_ld_payload.template update_tdesc<tdesc_update_dir::y_dir>(
                        real_wg_tile_m);
                tile_load(local_ld, local_ld_payload);
                mat_slice.reg = reduce_helper<reduce_kind, dtype>(
                        mat_slice.reg, local_ld.reg);
            }
        }
    }
};

/// @brief Workgroups to do the cooperative reduction.
/// Specialized for Xe architecture with 1 workgroups.
template <reduce_op reduce_kind, typename tile_shape_, typename matAcc_t,
        gpu_arch arch_tag_>
class cooperative_reduce_t<reduce_kind, tile_shape_, matAcc_t, 1, arch_tag_,
        mem_space::local,
        std::enable_if_t<((arch_tag_ == gpu_arch::Xe))
                && (tile_shape_::dim == 2)>> {
public:
    static constexpr gpu_arch arch_tag = arch_tag_;
    using tile_shape = tile_shape_;
    using dtype = typename matAcc_t::dtype;

private:
    using work_group_t = typename tile_shape::work_group_t;

public:
    using mat_slice_t = matAcc_t;
    static constexpr uint32_t barrier_count = 0;
    static constexpr uint32_t slm_size = 0;
    static constexpr uint32_t coop_num_x = 1;
    static constexpr uint32_t coop_num_y = 1;
    uint32_t coop_id;
    uint32_t coop_id_x;
    uint32_t coop_id_y;
    inline cooperative_reduce_t(uint32_t coop_id_) {
        coop_id = 0;
        coop_id_x = 0;
        coop_id_y = 0;
    }
    inline bool is_valid_post_process_wg() { return true; }

    inline KERNEL_FUNC void operator()(work_group_t &g, mat_slice_t &mat_slice,
            matAcc_t &matAcc, uint32_t slm_base = 0,
            uint32_t nbarrier_base = 0) {
        mat_slice.reg = matAcc.reg;
    }
};
} // namespace gpu::xetla::group
