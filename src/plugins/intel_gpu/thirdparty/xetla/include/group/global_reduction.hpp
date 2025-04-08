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

/// @brief Cross group global reduction.
/// @tparam reduce_kind Is the reduction type.
/// @tparam tile_shape_acc Is the group-level tile shape for accumulation tile.
/// @tparam tile_shape_cnt Is the group-level tile shape for counter tile.
/// @tparam mem_desc_acc_t Is the memory descriptor of accumulation buffer.
/// @tparam mem_desc_cnt_t Is the memory descriptor of counter buffer.
/// @tparam num_group_reduction Is the number of workgroups to do the reduction.
/// @tparam counter_size Is the stride to the next counter.
/// @tparam arch_tag Is the HW architecture.
template <reduce_op reduce_kind, typename tile_shape_acc,
        typename tile_shape_cnt, typename mem_desc_acc_t,
        typename mem_desc_cnt_t, uint32_t num_group_reduction,
        uint32_t counter_size, gpu_arch arch_tag, class enable = void>
class global_reduce_t {};

/// @brief Cross group global reduction. Specialized for reduce_op::sum and Xe architecture.
template <typename tile_shape_acc_, typename tile_shape_cnt_,
        typename mem_desc_acc_t_, typename mem_desc_cnt_t_,
        uint32_t num_group_reduction, uint32_t counter_size, gpu_arch arch_tag_>
class global_reduce_t<reduce_op::sum, tile_shape_acc_, tile_shape_cnt_,
        mem_desc_acc_t_, mem_desc_cnt_t_, num_group_reduction, counter_size,
        arch_tag_,
        std::enable_if_t<((arch_tag_ == gpu_arch::Xe))
                && (tile_shape_acc_::dim == 2)>> {
public:
    static constexpr gpu_arch arch_tag = arch_tag_;
    using tile_shape_acc = tile_shape_acc_;
    using tile_shape_cnt = tile_shape_cnt_;
    using mem_desc_acc_t = mem_desc_acc_t_;
    using mem_desc_cnt_t = mem_desc_cnt_t_;
    using dtype_acc = typename mem_desc_acc_t::dtype;
    using dtype_cnt = typename mem_desc_cnt_t::dtype;

private:
    static constexpr uint32_t acc_sg_tile_y = tile_shape_acc::sg_tile_size_y;
    static constexpr uint32_t acc_sg_tile_x = tile_shape_acc::sg_tile_size_x;
    static constexpr uint32_t cnt_sg_tile_y = tile_shape_cnt::sg_tile_size_y;
    static constexpr uint32_t cnt_sg_tile_x = tile_shape_cnt::sg_tile_size_x;
    static constexpr uint32_t wg_size_x = tile_shape_acc::wg_size_x;
    static constexpr uint32_t wg_size_y = tile_shape_acc::wg_size_y;
    static_assert((tile_shape_acc::wg_size_x == tile_shape_cnt::wg_size_x)
                    && (tile_shape_acc::wg_size_y == tile_shape_cnt::wg_size_y),
            "acc and cnt wg shape need to be matched");
    using work_group_t = typename tile_shape_acc::work_group_t;

    /// @brief Updates tile base descriptor based on the tid.
    inline void update_sg_tile_tdesc(work_group_t &g,
            mem_desc_acc_t &mem_desc_acc, mem_desc_cnt_t &mem_desc_cnt) {
        int32_t sg_idx = g.get_id() % wg_size_x;
        int32_t sg_idy = g.get_id() / wg_size_x;
        int32_t acc_tile_offset_x = sg_idx * acc_sg_tile_x;
        int32_t acc_tile_offset_y = sg_idy * acc_sg_tile_y;
        mem_desc_acc.update_coord(acc_tile_offset_x, acc_tile_offset_y);
        int32_t cnt_tile_offset_x = sg_idx * cnt_sg_tile_x;
        int32_t cnt_tile_offset_y = sg_idy * cnt_sg_tile_y;
        mem_desc_cnt.update_coord(cnt_tile_offset_x, cnt_tile_offset_y);
    }

    inline uint32_t update_reduce_counter(mem_desc_cnt_t &mem_desc_cnt) {
        constexpr uint32_t SIMD = 16;
        uint32_t pitch_in_bytes = mem_desc_cnt.shape.stride_x
                * sizeof(dtype_cnt) * counter_size;
        uint32_t offset_x = mem_desc_cnt.coord.x;
        uint32_t offset_y = mem_desc_cnt.coord.y;
        uint64_t address = (uint64_t)mem_desc_cnt.base.base
                + offset_y * pitch_in_bytes
                + offset_x * sizeof(dtype_cnt) * counter_size;
        xetla_vector<uint32_t, SIMD> offsets
                = xetla_vector_gen<uint32_t, SIMD>(0, 1);
        offsets *= sizeof(dtype_cnt);
        xetla_mask<SIMD> pred(0);
        pred[0] = 1;
        xetla_vector<dtype_cnt, SIMD> ret = xetla_atomic_global<atomic_op::iinc,
                dtype_cnt, SIMD, data_size::default_size, cache_hint::uncached,
                cache_hint::write_back>((dtype_cnt *)address, offsets, pred);
        return ret[0];
    }

    inline void clean_reduce_counter(mem_desc_cnt_t &mem_desc_cnt) {
        uint32_t pitch_in_bytes = mem_desc_cnt.shape.stride_x
                * sizeof(dtype_cnt) * counter_size;
        uint32_t offset_x = mem_desc_cnt.coord.x;
        uint32_t offset_y = mem_desc_cnt.coord.y;
        uint64_t address = (uint64_t)mem_desc_cnt.base.base
                + offset_y * pitch_in_bytes
                + offset_x * sizeof(dtype_cnt) * counter_size;
        xetla_vector<dtype_cnt, 1> zeros(0);

        xetla_store_global<dtype_cnt, 1, data_size::default_size,
                cache_hint::uncached, cache_hint::write_back>(
                (dtype_cnt *)address, 0, zeros);
    }

public:
    static constexpr uint32_t barrier_count = 0;
    static constexpr uint32_t slm_size = 0;
    uint32_t reduce_id = 0;

    inline bool is_last_group() {
        return reduce_id == (num_group_reduction - 1);
    }

    /// @brief Global reduction.
    /// 1) each group stores tile data to global memory by using global atomic add ->
    /// 2) after reduction complete, update the counter by using atomic inc ->
    /// 3) the last group load the data back to EU, and clean the accumulation buffer and counter buffer.
    /// @note only the last group has the valid data.
    /// @param g Is the workgroup of the current tile.
    /// @param matAcc Is the input of the reduction.
    /// @param mem_desc_acc Is the memory descriptor of accumulation buffer.
    /// @param mem_desc_cnt Is the memory descriptor of counter buffer.
    /// @param slm_base Is the slm base address.
    /// @param nbarrier_base Is the named barrier base.
    template <typename matAcc_t>
    __XETLA_API KERNEL_FUNC void operator()(work_group_t &g, matAcc_t &matAcc,
            mem_desc_acc_t mem_desc_acc, mem_desc_cnt_t mem_desc_cnt,
            uint32_t slm_base = 0, uint32_t nbarrier_base = 0) {
        static_assert(std::is_same<typename matAcc_t::dtype, dtype_acc>::value,
                "matAcc_t::dtype should match with dtype_acc");
        update_sg_tile_tdesc(g, mem_desc_acc, mem_desc_cnt);
        using matAcc_tile_desc_t = typename matAcc_t::tile_desc;
        using matAcc_store_payload_t = subgroup::mem_payload_t<mem_desc_acc_t,
                matAcc_tile_desc_t, msg_type::atomic_add, arch_tag>;
        matAcc_store_payload_t matAcc_store_payload(mem_desc_acc);
        subgroup::tile_store<cache_hint::uncached, cache_hint::write_back>(
                matAcc, matAcc_store_payload);
        xetla_fence<memory_kind::untyped_global, fence_op::none,
                fence_scope::tile>();
        reduce_id = update_reduce_counter(mem_desc_cnt);
        if (reduce_id == (num_group_reduction - 1)) {
            using matAcc_payload_t = subgroup::mem_payload_t<mem_desc_acc_t,
                    matAcc_tile_desc_t, msg_type::block_2d, arch_tag>;
            matAcc_payload_t matAcc_payload(mem_desc_acc);
            subgroup::tile_load(matAcc, matAcc_payload);
            clean_reduce_counter(mem_desc_cnt);
            using mat_zero_t = subgroup::tile_t<dtype_acc, matAcc_tile_desc_t>;
            mat_zero_t mat_zero;
            mat_zero.reg = 0;
            subgroup::tile_store<cache_hint::uncached, cache_hint::write_back>(
                    mat_zero, matAcc_payload);
            SW_BARRIER();
        }
    }
};

/// @brief Cross group global reduction. Specialized for num_group_reduction=1 and Xe architecture.
template <typename tile_shape_acc_, typename tile_shape_cnt_,
        typename mem_desc_acc_t_, typename mem_desc_cnt_t_,
        uint32_t counter_size_, gpu_arch arch_tag_>
class global_reduce_t<reduce_op::sum, tile_shape_acc_, tile_shape_cnt_,
        mem_desc_acc_t_, mem_desc_cnt_t_, 1, counter_size_, arch_tag_,
        std::enable_if_t<((arch_tag_ == gpu_arch::Xe))
                && (tile_shape_acc_::dim == 2)>> {
public:
    static constexpr gpu_arch arch_tag = arch_tag_;
    using tile_shape_acc = tile_shape_acc_;
    using tile_shape_cnt = tile_shape_cnt_;
    using mem_desc_acc_t = mem_desc_acc_t_;
    using mem_desc_cnt_t = mem_desc_cnt_t_;
    using dtype_acc = typename mem_desc_acc_t::dtype;

private:
    using work_group_t = typename tile_shape_acc::work_group_t;

public:
    static constexpr uint32_t barrier_count = 0;
    static constexpr uint32_t slm_size = 0;
    inline bool is_last_group() { return true; }

    template <typename matAcc_t>
    inline KERNEL_FUNC void operator()(work_group_t &g, matAcc_t &matAcc,
            mem_desc_acc_t mem_desc_acc, mem_desc_cnt_t mem_desc_cnt,
            uint32_t slm_base = 0, uint32_t nbarrier_base = 0) {}
};

/// @brief Cross group global reduction. Specialized for reduce_op::sum and 4 dimensional tiles
template <typename tile_shape_acc_, typename tile_shape_cnt_,
        typename mem_desc_acc_t_, typename mem_desc_cnt_t_,
        uint32_t num_group_reduction, uint32_t counter_size, gpu_arch arch_tag_>
class global_reduce_t<reduce_op::sum, tile_shape_acc_, tile_shape_cnt_,
        mem_desc_acc_t_, mem_desc_cnt_t_, num_group_reduction, counter_size,
        arch_tag_,
        std::enable_if_t<((arch_tag_ == gpu_arch::Xe))
                && (tile_shape_acc_::dim == 4)>> {
public:
    static constexpr gpu_arch arch_tag = arch_tag_;
    using tile_shape_acc = tile_shape_acc_;
    using tile_shape_cnt = tile_shape_cnt_;
    using mem_desc_acc_t = mem_desc_acc_t_;
    using mem_desc_cnt_t = mem_desc_cnt_t_;
    using dtype_acc = typename mem_desc_acc_t::dtype;
    using dtype_cnt = typename mem_desc_cnt_t::dtype;

private:
    static constexpr uint32_t acc_sg_tile_n = tile_shape_acc::sg_tile_size_n;
    static constexpr uint32_t acc_sg_tile_p = tile_shape_acc::sg_tile_size_p;
    static constexpr uint32_t acc_sg_tile_q = tile_shape_acc::sg_tile_size_q;
    static constexpr uint32_t acc_sg_tile_k = tile_shape_acc::sg_tile_size_k;

    static constexpr uint32_t wg_size_n = tile_shape_acc::wg_size_n;
    static constexpr uint32_t wg_size_p = tile_shape_acc::wg_size_p;
    static constexpr uint32_t wg_size_q = tile_shape_acc::wg_size_q;
    static constexpr uint32_t wg_size_k = tile_shape_acc::wg_size_k;

    static constexpr uint32_t cnt_sg_tile_n = tile_shape_cnt::sg_tile_size_n;
    static constexpr uint32_t cnt_sg_tile_p = tile_shape_cnt::sg_tile_size_p;
    static constexpr uint32_t cnt_sg_tile_q = tile_shape_cnt::sg_tile_size_q;
    static constexpr uint32_t cnt_sg_tile_k = tile_shape_cnt::sg_tile_size_k;

    static_assert((tile_shape_acc::wg_size_n == tile_shape_cnt::wg_size_n),
            "acc and cnt wg_szie_n shape need to be matched");
    static_assert((tile_shape_acc::wg_size_p == tile_shape_cnt::wg_size_p),
            "acc and cnt wg_szie_n shape need to be matched");
    static_assert((tile_shape_acc::wg_size_q == tile_shape_cnt::wg_size_q),
            "acc and cnt wg_szie_n shape need to be matched");
    static_assert((tile_shape_acc::wg_size_k == tile_shape_cnt::wg_size_k),
            "acc and cnt wg_szie_n shape need to be matched");

    static_assert((tile_shape_acc::wg_size_n == tile_shape_cnt::wg_size_n)
                    && (tile_shape_acc::wg_size_p == tile_shape_cnt::wg_size_p)
                    && (tile_shape_acc::wg_size_q == tile_shape_cnt::wg_size_q)
                    && (tile_shape_acc::wg_size_k == tile_shape_cnt::wg_size_k),
            "acc and cnt wg shape need to be matched");
    using work_group_t = typename tile_shape_acc::work_group_t;

    /// @brief Updates tile base descriptor based on the tid.
    inline void update_sg_tile_tdesc(work_group_t &g,
            mem_desc_acc_t &mem_desc_acc, mem_desc_cnt_t &mem_desc_cnt) {
        int32_t sg_idk = g.get_id() % wg_size_k;
        int32_t sg_idq = (g.get_id() / wg_size_k) % wg_size_q;
        int32_t sg_idp = (g.get_id() / (wg_size_k * wg_size_q)) % wg_size_p;
        int32_t sg_idn = (g.get_id() / (wg_size_k * wg_size_q * wg_size_p))
                % wg_size_n;

        int32_t acc_tile_offset_n = sg_idn * acc_sg_tile_n;
        int32_t acc_tile_offset_p = sg_idp * acc_sg_tile_p;
        int32_t acc_tile_offset_q = sg_idq * acc_sg_tile_q;
        int32_t acc_tile_offset_k = sg_idk * acc_sg_tile_k;
        mem_desc_acc.update_coord(acc_tile_offset_k, acc_tile_offset_q,
                acc_tile_offset_p, acc_tile_offset_n);

        int32_t cnt_tile_offset_n = sg_idn * cnt_sg_tile_n;
        int32_t cnt_tile_offset_p = sg_idp * cnt_sg_tile_p;
        int32_t cnt_tile_offset_q = sg_idq * cnt_sg_tile_q;
        int32_t cnt_tile_offset_k = sg_idk * cnt_sg_tile_k;
        mem_desc_cnt.update_coord(cnt_tile_offset_k, cnt_tile_offset_q,
                cnt_tile_offset_p, cnt_tile_offset_n);
    }

    inline uint32_t update_reduce_counter(mem_desc_cnt_t &mem_desc_cnt) {
        constexpr uint32_t SIMD = 16;
        mem_shape_t<4> shape = mem_desc_cnt.shape;
        mem_coord_t<4> coord = mem_desc_cnt.coord;

        uint32_t pitch_in_bytes
                = shape.stride_x * sizeof(dtype_cnt) * counter_size;
        uint32_t offset_x = coord.x;

        uint32_t offset_y = coord.w * shape.stride_z * shape.stride_y
                + coord.z * shape.stride_y + coord.y;
        uint64_t address = (uint64_t)mem_desc_cnt.base.base
                + offset_y * pitch_in_bytes
                + offset_x * sizeof(dtype_cnt) * counter_size;
        xetla_vector<uint32_t, SIMD> offsets
                = xetla_vector_gen<uint32_t, SIMD>(0, 1);
        offsets *= sizeof(dtype_cnt);
        xetla_mask<SIMD> pred(0);
        pred[0] = 1;
        xetla_vector<dtype_cnt, SIMD> ret = xetla_atomic_global<atomic_op::iinc,
                dtype_cnt, SIMD, data_size::default_size, cache_hint::uncached,
                cache_hint::write_back>((dtype_cnt *)address, offsets, pred);

        return ret[0];
    }

    inline void clean_reduce_counter(mem_desc_cnt_t &mem_desc_cnt) {
        mem_shape_t<4> shape = mem_desc_cnt.shape;
        mem_coord_t<4> coord = mem_desc_cnt.coord;

        uint32_t pitch_in_bytes
                = shape.stride_x * sizeof(dtype_cnt) * counter_size;
        uint32_t offset_x = coord.x;

        uint32_t offset_y = coord.w * shape.stride_z * shape.stride_y
                + coord.z * shape.stride_y + coord.y;
        uint64_t address = (uint64_t)mem_desc_cnt.base.base
                + offset_y * pitch_in_bytes
                + offset_x * sizeof(dtype_cnt) * counter_size;
        xetla_vector<dtype_cnt, 1> zeros(0);

        xetla_store_global<dtype_cnt, 1, data_size::default_size,
                cache_hint::uncached, cache_hint::write_back>(
                (dtype_cnt *)address, 0, zeros);
    }

public:
    static constexpr uint32_t barrier_count = 0;
    static constexpr uint32_t slm_size = 0;
    uint32_t reduce_id = 0;

    inline bool is_last_group() {
        return reduce_id == (num_group_reduction - 1);
    }

    /// @brief Global reduction.
    /// 1) each group stores tile data to global memory by using global atomic add ->
    /// 2) after reduction complete, update the counter by using atomic inc ->
    /// 3) the last group load the data back to EU, and clean the accumulation buffer and counter buffer.
    /// @note only the last group has the valid data.
    /// @param g Is the workgroup of the current tile.
    /// @param matAcc Is the input of the reduction.
    /// @param mem_desc_acc Is the memory descriptor of accumulation buffer.
    /// @param mem_desc_cnt Is the memory descriptor of counter buffer.
    /// @param slm_base Is the slm base address.
    /// @param nbarrier_base Is the named barrier base.
    template <typename matAcc_t, int sg_tile_n, int sg_tile_p>
    __XETLA_API KERNEL_FUNC void operator()(work_group_t &g,
            matAcc_t matAcc[sg_tile_n][sg_tile_p], mem_desc_acc_t mem_desc_acc,
            mem_desc_cnt_t mem_desc_cnt, uint32_t slm_base = 0,
            uint32_t nbarrier_base = 0) {
        static_assert(std::is_same<typename matAcc_t::dtype, dtype_acc>::value,
                "matAcc_t::dtype should match with dtype_acc");

        update_sg_tile_tdesc(g, mem_desc_acc, mem_desc_cnt);

        using matAcc_tile_desc_t = typename matAcc_t::tile_desc;
        using matAcc_store_payload_t = subgroup::mem_payload_t<
                mem_desc_t<dtype_acc, mem_layout::nhwc, mem_space::global, 8,
                        4>,
                matAcc_tile_desc_t, msg_type::atomic_add, arch_tag>;

        matAcc_store_payload_t matAcc_store_payload(mem_desc_acc);

#pragma unroll
        for (uint32_t n = 0; n < sg_tile_n; n++) {
#pragma unroll
            for (uint32_t p = 0; p < sg_tile_p; p++) {

                int32_t offset_n = mem_desc_acc.get_base_offset_from_w(n);
                int32_t offset_p = mem_desc_acc.get_base_offset_from_z(p);
                int mask_n = mem_desc_acc.get_mask_from_w(n);
                int mask_p = mem_desc_acc.get_mask_from_z(p);
                int32_t base_offset = (offset_n + offset_p) * sizeof(dtype_acc);

                matAcc_store_payload.set_tdesc_base_address_masked(
                        (uint64_t)mem_desc_acc.base.base + base_offset,
                        mask_n & mask_p);

                subgroup::tile_store<cache_hint::uncached,
                        cache_hint::write_back>(
                        matAcc[n][p], matAcc_store_payload);
            }
        }

        xetla_fence<memory_kind::untyped_global, fence_op::none,
                fence_scope::group>();

        reduce_id = update_reduce_counter(mem_desc_cnt);
        if (reduce_id == (num_group_reduction - 1)) {
            using matAcc_payload_t = subgroup::mem_payload_t<
                    mem_desc_t<dtype_acc, mem_layout::nhwc, mem_space::global,
                            8, 4>,
                    matAcc_tile_desc_t, msg_type::block_2d, arch_tag>;
            matAcc_payload_t matAcc_payload(mem_desc_acc);

            using mat_zero_t = subgroup::tile_t<dtype_acc, matAcc_tile_desc_t>;
            mat_zero_t mat_zero;
            mat_zero.reg = 0;

#pragma unroll
            for (uint32_t n = 0; n < sg_tile_n; n++) {
#pragma unroll
                for (uint32_t p = 0; p < sg_tile_p; p++) {
                    int32_t offset_n = mem_desc_acc.get_base_offset_from_w(n);
                    int32_t offset_p = mem_desc_acc.get_base_offset_from_z(p);
                    int mask_n = mem_desc_acc.get_mask_from_w(n);
                    int mask_p = mem_desc_acc.get_mask_from_z(p);
                    int32_t base_offset
                            = (offset_n + offset_p) * sizeof(dtype_acc);

                    matAcc_payload.set_tdesc_base_address_masked(
                            (uint64_t)mem_desc_acc.base.base + base_offset,
                            mask_n & mask_p);

                    subgroup::tile_load(matAcc[n][p], matAcc_payload);
                    SW_BARRIER();

                    subgroup::tile_store<cache_hint::uncached,
                            cache_hint::write_back>(mat_zero, matAcc_payload);
                }
            }

            clean_reduce_counter(mem_desc_cnt);
            SW_BARRIER();
        }
    }
};

/// @brief Cross group global reduction. Specialized for num_group_reduction=1
template <typename tile_shape_acc_, typename tile_shape_cnt_,
        typename mem_desc_acc_t_, typename mem_desc_cnt_t_,
        uint32_t counter_size_, gpu_arch arch_tag_>
class global_reduce_t<reduce_op::sum, tile_shape_acc_, tile_shape_cnt_,
        mem_desc_acc_t_, mem_desc_cnt_t_, 1, counter_size_, arch_tag_,
        std::enable_if_t<((arch_tag_ == gpu_arch::Xe))
                && (tile_shape_acc_::dim == 4)>> {
public:
    static constexpr gpu_arch arch_tag = arch_tag_;
    using tile_shape_acc = tile_shape_acc_;
    using tile_shape_cnt = tile_shape_cnt_;
    using mem_desc_acc_t = mem_desc_acc_t_;
    using mem_desc_cnt_t = mem_desc_cnt_t_;
    using dtype_acc = typename mem_desc_acc_t::dtype;

private:
    using work_group_t = typename tile_shape_acc::work_group_t;

    static constexpr uint32_t acc_sg_tile_n = tile_shape_acc::sg_tile_size_n;
    static constexpr uint32_t acc_sg_tile_p = tile_shape_acc::sg_tile_size_p;
    static constexpr uint32_t acc_sg_tile_q = tile_shape_acc::sg_tile_size_q;
    static constexpr uint32_t acc_sg_tile_k = tile_shape_acc::sg_tile_size_k;

public:
    static constexpr uint32_t barrier_count = 0;
    static constexpr uint32_t slm_size = 0;
    inline bool is_last_group() { return true; }

    template <typename matAcc_t, int sg_tile_n, int sg_tile_p>
    inline KERNEL_FUNC void operator()(work_group_t &g,
            matAcc_t matAcc[sg_tile_n][sg_tile_p], mem_desc_acc_t mem_desc_acc,
            mem_desc_cnt_t mem_desc_cnt, uint32_t slm_base = 0,
            uint32_t nbarrier_base = 0) {}
};
} // namespace gpu::xetla::group
