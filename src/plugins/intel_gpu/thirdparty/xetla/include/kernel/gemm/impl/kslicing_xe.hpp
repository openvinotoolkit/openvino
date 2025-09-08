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

#include "kernel/gemm/api.hpp"
#include "kernel/gemm/common.hpp"
#include "kernel/gemm/dispatch_policy.hpp"

namespace gpu::xetla::kernel {

/// @addtogroup xetla_gemm_universal
/// @{

/// @brief Is the gemm_universal functor, specialized in kslicing dispatch policy and Xe architecture.
///
/// @tparam num_global_kslicing_ Is the k dim split ratio between groups.
/// @tparam num_local_kslicing_ Is the k dim split ratio within a group.
/// @tparam gemm_t_ Is the gemm functor to compose a GEMM_UNIVERSAL.
/// @tparam epilogue_t_ Is the epilogue functor to compose a GEMM_UNIVERSAL.
template <int num_global_kslicing_, int num_local_kslicing_, typename gemm_t_,
        typename epilogue_t_, typename group_swizzle_>
class gemm_universal_t<dispatch_policy_kslicing<group_swizzle_,
                               num_global_kslicing_, num_local_kslicing_>,
        gemm_t_, epilogue_t_,
        std::enable_if_t<(group_swizzle_::arch_tag == gpu_arch::Xe)>> {
    using gemm_t = gemm_t_;
    using epilogue_t = epilogue_t_;
    using gemm_args_t = typename gemm_t::arguments_t;
    using epilogue_args_t = typename epilogue_t::arguments_t;
    using tile_shape = typename gemm_t::tile_shape;
    using group_swizzle_t = group_swizzle_;

    static constexpr uint32_t wg_tile_m = tile_shape::wg_tile_size_y;
    static constexpr uint32_t wg_tile_n = tile_shape::wg_tile_size_x;
    static constexpr uint32_t sg_tile_m = tile_shape::sg_tile_size_y;
    static constexpr uint32_t sg_tile_n = tile_shape::sg_tile_size_x;
    static constexpr uint32_t wg_size_y = tile_shape::wg_size_y;
    static constexpr uint32_t wg_size_x = tile_shape::wg_size_x;
    static constexpr uint32_t real_wg_tile_m = sg_tile_m * wg_size_y;
    static constexpr uint32_t real_wg_tile_n = sg_tile_n * wg_size_x;

    static constexpr uint32_t k_stride = gemm_t::k_stride;
    using work_group_t = typename gemm_t::work_group_t;
    static constexpr uint32_t work_group_size = work_group_t::size;

    static constexpr gpu_arch arch_tag = group_swizzle_t::arch_tag;
    static_assert(arch_tag == gemm_t::arch_tag, "arch_tag should be the same");
    static_assert(
            arch_tag == epilogue_t::arch_tag, "arch_tag should be the same");
    static_assert(std::is_same<typename gemm_t::tile_shape,
                          typename epilogue_t::tile_shape>::value,
            "tile_shape should be the same");

    using mem_desc_a_t = typename gemm_t::mem_desc_a_t;
    using mem_desc_b_t = typename gemm_t::mem_desc_b_t;
    using mem_desc_c_t = typename epilogue_t::mem_desc_c_t;
    using matA_base_t = typename mem_desc_a_t::base_t;
    using matB_base_t = typename mem_desc_b_t::base_t;
    using matC_base_t = typename mem_desc_c_t::base_t;
    using dtype_a = typename mem_desc_a_t::dtype;
    using dtype_b = typename mem_desc_b_t::dtype;
    using dtype_c = typename mem_desc_c_t::dtype;
    using matAcc_t = typename gemm_t::matAcc_t;
    using dtype_acc = typename matAcc_t::dtype;
    using mem_desc_acc_t
            = mem_desc_t<dtype_acc, mem_layout::row_major, mem_space::global>;
    using mem_desc_cnt_t
            = mem_desc_t<uint32_t, mem_layout::row_major, mem_space::global>;
    using acc_base_t = typename mem_desc_acc_t::base_t;
    using cnt_base_t = typename mem_desc_cnt_t::base_t;

    static constexpr uint32_t num_global_kslicing = num_global_kslicing_;
    static constexpr uint32_t num_local_kslicing = num_local_kslicing_;
    static_assert((num_global_kslicing > 0) && (num_local_kslicing > 0),
            "min slicing ratio is 1");

    static_assert((num_local_kslicing & (num_local_kslicing - 1)) == 0,
            "num_local_kslicing should be power of 2!");

    using kslicing_t = group::cooperative_reduce_t<reduce_op::sum, tile_shape,
            matAcc_t, num_local_kslicing, arch_tag>;
    using mat_slice_t = typename kslicing_t::mat_slice_t;
    static constexpr uint32_t ks_coop_num_x = kslicing_t::coop_num_x;
    static constexpr uint32_t ks_coop_num_y = kslicing_t::coop_num_y;

    static constexpr uint32_t gemm_nbarr_count = gemm_t::barrier_count;
    static constexpr uint32_t gemm_slm_size = gemm_t::slm_size;

    static constexpr uint32_t epilogue_nbarr_count = epilogue_t::barrier_count;
    static constexpr uint32_t epilogue_slm_size = epilogue_t::slm_size;

    static constexpr uint32_t kslicing_nbarr_count = kslicing_t::barrier_count;
    static constexpr uint32_t kslicing_slm_size = kslicing_t::slm_size;

    static constexpr uint32_t counter_size = 8;

    static constexpr uint32_t alignment = 8 / sizeof(dtype_acc);

    using tile_shape_cnt = group::tile_shape_t<ks_coop_num_x * wg_size_x,
            ks_coop_num_y * wg_size_y, ks_coop_num_x, ks_coop_num_y>;

    using global_group_reduce_t = group::global_reduce_t<reduce_op::sum,
            tile_shape, tile_shape_cnt, mem_desc_acc_t, mem_desc_cnt_t,
            num_global_kslicing, counter_size, arch_tag>;

public:
    /// @brief GEMM_UNIVERSAL arguments.
    /// This is the interface for users to pass the application-related runtime variables.
    struct arguments_t {
        /// @brief Is the size of the m dimension of the matrix multiplication (m x k x n).
        uint32_t matrix_m;
        /// @brief Is the size of the k dimension of the matrix multiplication (m x k x n).
        uint32_t matrix_k;
        /// @brief Is the size of the n dimension of the matrix multiplication (m x k x n).
        uint32_t matrix_n;
        /// @brief Is the base address of matrix A.
        matA_base_t matA_base;
        /// @brief Is the leading dimension (pitch) size of the matrix A in memory.
        uint32_t matA_ld;
        /// @brief Is the base address of matrix B.
        matB_base_t matB_base;
        /// @brief Is the leading dimension (pitch) size of the matrix B in memory.
        uint32_t matB_ld;
        /// @brief Is the base address of matrix C.
        matC_base_t matC_base;
        /// @brief Is the leading dimension (pitch) size of the matrix C in memory.
        uint32_t matC_ld;
        /// @brief Is the base address of accumulation buffer.
        acc_base_t acc_base;
        /// @brief Is the base address of counter buffer.
        cnt_base_t cnt_base;
        /// @brief Is the epilogue arguments.
        epilogue_args_t epilogue_args;

        /// @brief Constructs arguments with default method.
        inline arguments_t() = default;

        /// @brief Set for device copyable
        static constexpr bool host_callable = true;

        // Be aware of the risks: Rule of three (copy constructor, copy assignment, destructor)
        // Please check if you need to add self-define destructor
        // ~arguments_t(){}

        /// @brief Constructs arguments with initialization list.
        /// @param matrix_m_ Is the size of the m dimension of the matrix multiplication (m x k x n).
        /// @param matrix_k_ Is the size of the k dimension of the matrix multiplication (m x k x n).
        /// @param matrix_n_ Is the size of the n dimension of the matrix multiplication (m x k x n).
        /// @param matA_base_ Is the base address of matrix A.
        /// @param matA_ld_ Is the leading dimension (pitch) size of the matrix A in memory.
        /// @param matB_base_ Is the base address of matrix B.
        /// @param matB_ld_ Is the leading dimension (pitch) size of the matrix B in memory.
        /// @param matC_base_ Is the base address of matrix C.
        /// @param matC_ld_ Is the leading dimension (pitch) size of the matrix C in memory.
        /// @param epilogue_args_ Is the epilogue arguments.
        inline arguments_t(uint32_t matrix_m_, uint32_t matrix_k_,
                uint32_t matrix_n_, matA_base_t matA_base_, uint32_t matA_ld_,
                matB_base_t matB_base_, uint32_t matB_ld_,
                matC_base_t matC_base_, uint32_t matC_ld_,
                acc_base_t acc_base_ = {}, cnt_base_t cnt_base_ = {},
                epilogue_args_t epilogue_args_ = {})
            : matrix_m(matrix_m_)
            , matrix_k(matrix_k_)
            , matrix_n(matrix_n_)
            , matA_base(matA_base_)
            , matA_ld(matA_ld_)
            , matB_base(matB_base_)
            , matB_ld(matB_ld_)
            , matC_base(matC_base_)
            , matC_ld(matC_ld_)
            , acc_base(acc_base_)
            , cnt_base(cnt_base_)
            , epilogue_args(epilogue_args_) {}
        inline arguments_t(const arguments_t &args)
            : matrix_m(args.matrix_m)
            , matrix_k(args.matrix_k)
            , matrix_n(args.matrix_n)
            , matA_base(args.matA_base)
            , matA_ld(args.matA_ld)
            , matB_base(args.matB_base)
            , matB_ld(args.matB_ld)
            , matC_base(args.matC_base)
            , matC_ld(args.matC_ld)
            , acc_base(args.acc_base)
            , cnt_base(args.cnt_base)
            , epilogue_args(args.epilogue_args) {}
        // Be aware of the risks: Rule of three (copy constructor, copy assignment, destructor)
        // Please check if you need to add self-define destructor
        // inline ~arguments_t(){}
        inline arguments_t &operator=(const arguments_t &args) {
            this->matrix_m = args.matrix_m;
            this->matrix_k = args.matrix_k;
            this->matrix_n = args.matrix_n;
            this->matA_base = args.matA_base;
            this->matA_ld = args.matA_ld;
            this->matB_base = args.matB_base;
            this->matB_ld = args.matB_ld;
            this->matC_base = args.matC_base;
            this->matC_ld = args.matC_ld;
            this->acc_base = args.acc_base;
            this->cnt_base = args.cnt_base;
            this->epilogue_args = args.epilogue_args;
            return *this;
        }
    };

    /// @brief Gets named_barrier id consumption count.
    /// Users query and get a named_barrier id consumption count in compile time.
    /// @return The count of named barriers required.
    __XETLA_API static constexpr uint32_t get_barrier_count() {
        constexpr uint32_t count = gemm_nbarr_count * num_local_kslicing
                + kslicing_nbarr_count
                + epilogue_nbarr_count * num_local_kslicing;
        static_assert(
                count <= 32, "The named_barrier count should be less than 32!");
        return count;
    }

    /// @brief Gets local memory size consumption.
    /// Users query and get a local memory consumption size in compile time.
    /// @return The size of local memory required.
    __XETLA_API static constexpr uint32_t get_slm_size() {
        constexpr uint32_t size = gemm_slm_size * num_local_kslicing
                + kslicing_slm_size + epilogue_slm_size * num_local_kslicing;
        static_assert(size <= (128 * 1024),
                "The local memory size should be less than 128KB!");
        return size;
    }

    /// @brief Main execution function for GEMM_UNIVERSAL.
    /// The processing order is 1) set group-level base and boundary, split group to workgroups ->
    /// 2) num_local_kslicing x gemms -> 3) local kslicing -> 4) num_local_kslicing x epilogues.
    /// @param item Is the sycl::nd_item, returns execution related information, such as workgroup id, subgroup id...
    /// @param args Is the GEMM_UNIVERSAL arguments for application-related runtime variables.
    /// @param slm_base Is the slm base address.
    /// @param nbarrier_base Is the named barrier base.
    __XETLA_API KERNEL_FUNC void operator()(sycl::nd_item<3> &item,
            const arguments_t &args, uint32_t slm_base = 0,
            uint32_t nbarrier_base = 0) const {
        // set up workgroup level coordinates and boundaries
        work_group_t g(item.get_local_linear_id() % work_group_size);
        uint32_t wg_id = item.get_local_linear_id() / work_group_size;
        group_swizzle_t group_swizzle;
        int start_m = group_swizzle.template get_tile_idx<1>(item) * wg_tile_m;
        int start_n = group_swizzle.template get_tile_idx<2>(item) * wg_tile_n;
        int start_k = 0;
        uint32_t wg_tile_k = args.matrix_k;
        uint32_t boundary_n = (start_n + wg_tile_n) > args.matrix_n
                ? args.matrix_n
                : (start_n + wg_tile_n);
        uint32_t boundary_m = (start_m + wg_tile_m) > args.matrix_m
                ? args.matrix_m
                : (start_m + wg_tile_m);
        uint32_t boundary_k = wg_tile_k;
        if constexpr (num_global_kslicing > 1) {
            wg_tile_k = (wg_tile_k + num_global_kslicing - 1)
                    / num_global_kslicing;
            start_k = start_k
                    + group_swizzle.template get_tile_idx<0>(item) * wg_tile_k;
            boundary_k = (start_k + wg_tile_k) > boundary_k
                    ? boundary_k
                    : (start_k + wg_tile_k);
        }
        if constexpr (num_local_kslicing > 1) {
            wg_tile_k
                    = (wg_tile_k + num_local_kslicing - 1) / num_local_kslicing;
            start_k = start_k + wg_id * wg_tile_k;
            boundary_k = (start_k + wg_tile_k) > boundary_k
                    ? boundary_k
                    : (start_k + wg_tile_k);
        }

        // set up arguments
        uint32_t gemm_slm_base = slm_base;
        uint32_t gemm_nbarr_base = nbarrier_base;
        if constexpr (num_local_kslicing > 1) {
            gemm_slm_base = slm_base + wg_id * gemm_slm_size;
            gemm_nbarr_base = nbarrier_base + wg_id * gemm_nbarr_count;
        }
        uint32_t kslicing_slm_base
                = slm_base + num_local_kslicing * gemm_slm_size;
        uint32_t kslicing_nbarr_base
                = nbarrier_base + num_local_kslicing * gemm_nbarr_count;
        uint32_t epilogue_slm_base = kslicing_slm_base + kslicing_slm_size;
        uint32_t epilogue_nbarr_base
                = kslicing_nbarr_base + kslicing_nbarr_count;

        mem_desc_a_t mem_desc_a;
        mem_desc_b_t mem_desc_b;
        mem_desc_c_t mem_desc_c;
        //setup for matA
        if constexpr (mem_desc_a_t::is_local) {
            mem_desc_a.init(args.matA_base,
                    {wg_tile_k, real_wg_tile_m, wg_tile_k}, {0, 0});
        } else {
            mem_desc_a.init(args.matA_base,
                    {boundary_k, boundary_m, args.matA_ld}, {start_k, start_m});
        }
        //setup for matB
        if constexpr (mem_desc_b_t::is_local) {
            mem_desc_b.init(args.matB_base,
                    {real_wg_tile_n, wg_tile_k, real_wg_tile_n}, {0, 0});
        } else {
            mem_desc_b.init(args.matB_base,
                    {boundary_n, boundary_k, args.matB_ld}, {start_n, start_k});
        }

        uint32_t inner_loop_count = (wg_tile_k + k_stride - 1) / k_stride;
        gemm_args_t gemm_args(mem_desc_a, mem_desc_b, inner_loop_count);
        matAcc_t matAcc;
        matAcc.init(0);
        gemm_t gemm;
        gemm(g, matAcc, gemm_args, gemm_slm_base, gemm_nbarr_base);

        kslicing_t kslicing(wg_id);
        mat_slice_t mat_slice;
        kslicing(g, mat_slice, matAcc, kslicing_slm_base, kslicing_nbarr_base);
        if (kslicing.is_valid_post_process_wg()) {
            //setup for matC
            //set up cooperative offset for matC store
            int32_t coop_offset_x
                    = kslicing.coop_id_x * mat_slice_t::tile_size_x;
            int32_t coop_offset_y
                    = kslicing.coop_id_y * mat_slice_t::tile_size_y;
            int32_t acc_start_x = start_n + coop_offset_x;
            int32_t acc_start_y = start_m + coop_offset_y;
            int32_t cnt_start_x = group_swizzle.template get_tile_idx<2>(item)
                            * tile_shape_cnt::wg_tile_size_x
                    + kslicing.coop_id_x;
            int32_t cnt_start_y = group_swizzle.template get_tile_idx<1>(item)
                            * tile_shape_cnt::wg_tile_size_y
                    + kslicing.coop_id_y;
            uint32_t group_range_x = item.get_group_range(2);
            uint32_t group_range_y = item.get_group_range(1);
            uint32_t cnt_size_x
                    = group_range_x * tile_shape_cnt::wg_tile_size_x;
            uint32_t cnt_size_y
                    = group_range_y * tile_shape_cnt::wg_tile_size_y;

            uint32_t acc_aligned_n
                    = (args.matrix_n + alignment - 1) / alignment * alignment;

            uint32_t acc_boundary_n = (start_n + wg_tile_n) > acc_aligned_n
                    ? acc_aligned_n
                    : start_n + wg_tile_n;

            mem_desc_acc_t mem_desc_acc(args.acc_base,
                    {acc_boundary_n, boundary_m, acc_aligned_n},
                    {acc_start_x, acc_start_y});
            mem_desc_cnt_t mem_desc_cnt(args.cnt_base,
                    {cnt_size_x, cnt_size_y, cnt_size_x},
                    {cnt_start_x, cnt_start_y});

            global_group_reduce_t global_group_reduce;
            global_group_reduce(g, mat_slice, mem_desc_acc, mem_desc_cnt);

            if (global_group_reduce.is_last_group()) {
                if constexpr (mem_desc_c_t::is_local) {
                    mem_desc_c.init(args.matC_base,
                            {real_wg_tile_n, real_wg_tile_m, real_wg_tile_n},
                            {coop_offset_x, coop_offset_y});
                } else {
                    mem_desc_c.init(args.matC_base,
                            {boundary_n, boundary_m, args.matC_ld},
                            {start_n + coop_offset_x, start_m + coop_offset_y});
                }
                epilogue_t epilogue;
                epilogue(g, mat_slice, mem_desc_c, args.epilogue_args,
                        epilogue_slm_base, epilogue_nbarr_base);
            }
        }
    }
};

/// @} xetla_gemm_universal

} // namespace gpu::xetla::kernel
