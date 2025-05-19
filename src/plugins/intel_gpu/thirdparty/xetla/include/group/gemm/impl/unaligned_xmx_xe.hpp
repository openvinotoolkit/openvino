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

#include "group/gemm/api.hpp"
#include "group/gemm/compute_policy.hpp"

namespace gpu::xetla::group {

/// @addtogroup xetla_gemm
/// @{

/// @brief Is the gemm functor for unaligned input, Xe architecture and matrix engine.
template <typename compute_attr_, typename perf_tuning_knob_,
        typename tile_shape_, typename mem_desc_a_t_, typename mem_desc_b_t_,
        typename pre_processing_t_, gpu_arch arch_tag_>
class gemm_t<compute_policy_unaligned_xmx<compute_attr_, perf_tuning_knob_,
                     arch_tag_>,
        tile_shape_, // tile shape of workgroup-level gemm
        mem_desc_a_t_, // memory attribute of matA
        mem_desc_b_t_, // memory attribute of matB
        pre_processing_t_, // pre_processing functor
        std::enable_if_t<(arch_tag_ == gpu_arch::Xe)
                >> {
public:
    using mem_desc_a_t = mem_desc_a_t_;
    using mem_desc_b_t = mem_desc_b_t_;
    using tile_shape = tile_shape_;
    using pre_processing_t = pre_processing_t_;
    using compute_policy = compute_policy_unaligned_xmx<compute_attr_,
            perf_tuning_knob_, arch_tag_>;

    static constexpr uint32_t num_cyclic = 3;

    static constexpr uint32_t k_stride = compute_policy::k_stride;
    static constexpr uint32_t sg_tile_m = tile_shape::sg_tile_size_y;
    static constexpr uint32_t sg_tile_n = tile_shape::sg_tile_size_x;
    static constexpr uint32_t wg_size_x = tile_shape::wg_size_x;
    static constexpr uint32_t wg_size_y = tile_shape::wg_size_y;
    using work_group_t = typename tile_shape::work_group_t;

    constexpr static gpu_arch arch_tag = compute_policy::arch_tag;

    static constexpr mem_layout mem_layout_a = mem_desc_a_t::layout;
    static constexpr mem_layout mem_layout_b = mem_desc_b_t::layout;
    static constexpr bool is_col_major_a
            = mem_layout_a == mem_layout::col_major;
    static constexpr bool is_col_major_b
            = mem_layout_b == mem_layout::col_major;

private:
    /******** set data type **********/
    using dtype_a = typename mem_desc_a_t::dtype;
    using dtype_b = typename mem_desc_b_t::dtype;
    using dtype_mma_acc = typename compute_policy::dtype_mma_acc;
    using dtype_mma_a = typename compute_policy::dtype_mma_a;
    using dtype_mma_b = typename compute_policy::dtype_mma_b;

    using check_dtype
            = group::gemm<gpu_arch::Xe>::default_xmx::check_dtype_default<
                    dtype_a, dtype_b, dtype_mma_a, dtype_mma_b>;

    /******** set memory attribute **********/
    static constexpr mem_space mem_space_a = mem_desc_a_t::space;
    static constexpr mem_space mem_space_b = mem_desc_b_t::space;

    static constexpr bool is_local_a = mem_space_a == mem_space::local;
    static constexpr bool is_local_b = mem_space_b == mem_space::local;
    static constexpr tdesc_update_dir update_dir_a = is_col_major_a
            ? tdesc_update_dir::y_dir
            : tdesc_update_dir::x_dir;
    static constexpr tdesc_update_dir update_dir_b = is_col_major_b
            ? tdesc_update_dir::x_dir
            : tdesc_update_dir::y_dir;

    using check_memory
            = group::gemm<gpu_arch::Xe>::default_xmx::check_memory_default<
                    mem_layout_a, mem_layout_b, mem_space_a, mem_space_b>;

    static constexpr uint32_t stages = compute_policy::stages;
    static constexpr uint32_t sync_freq = compute_policy::sync_freq;

    /******** set tile layout && worker scope **********/
    static constexpr uint32_t tile_size_x_a = k_stride;
    static constexpr uint32_t tile_size_y_a = sg_tile_m;
    static constexpr uint32_t tile_size_x_b = sg_tile_n;
    static constexpr uint32_t tile_size_y_b = k_stride;
    static constexpr uint32_t tile_size_x_c = sg_tile_n;
    static constexpr uint32_t tile_size_y_c = sg_tile_m;
    static constexpr uint32_t block_size_x_a = compute_policy::block_size_x_a;
    static constexpr uint32_t block_size_y_a
            = (compute_policy::block_size_y_a > tile_size_y_a)
            ? tile_size_y_a
            : compute_policy::block_size_y_a;
    static constexpr uint32_t block_size_x_b = compute_policy::block_size_x_b;
    static constexpr uint32_t block_size_y_b = compute_policy::block_size_y_b;

    using check_tile_size = group::gemm<
            gpu_arch::Xe>::default_xmx::check_tile_size_default<dtype_mma_a,
            tile_size_x_a, tile_size_y_a, block_size_x_a, block_size_y_a,
            tile_size_x_b, tile_size_y_b, block_size_x_b, block_size_y_b>;

    /******** set tile  **********/
    static constexpr reg_layout reg_layout_a = reg_layout::tiled;
    using matA_tile_desc_t = subgroup::tile_desc_t<tile_size_x_a, tile_size_y_a,
            block_size_x_a, block_size_y_a, reg_layout_a>;

    using matA_t = subgroup::tile_t<dtype_a, matA_tile_desc_t>;

    using cooperative_helper_A_t = subgroup::cooperative_load_helper_t<matA_t,
            mem_layout_a, tile_shape::wg_size_x, arch_tag>;
    using cooperative_tile_desc_A_t =
            typename cooperative_helper_A_t::co_tile_desc_t;
    using partial_matA_t = subgroup::tile_t<dtype_a, cooperative_tile_desc_A_t>;
    using matA_payload_t = subgroup::mem_payload_t<mem_desc_a_t,
            cooperative_tile_desc_A_t,
            is_local_a ? msg_type::scatter : msg_type::unaligned_2d, arch_tag>;

    using matA_payload_local_st_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_a, mem_layout::row_major, mem_space::local>,
            cooperative_tile_desc_A_t, msg_type::scatter, arch_tag>;
    using matA_payload_local_ld_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_a, mem_layout::row_major, mem_space::local>,
            matA_tile_desc_t, msg_type::scatter, arch_tag>;

    using matA_acc_t = subgroup::tile_t<dtype_mma_a, matA_tile_desc_t>;
    using matA_prefetch_payload_t = subgroup::prefetch_payload_t<mem_desc_a_t,
            subgroup::tile_desc_t<tile_size_x_a, tile_size_y_a, 1, 1>,
            wg_size_x, arch_tag>;
    static constexpr reg_layout reg_layout_b
            = sizeof(dtype_b) < sizeof(uint32_t) ? reg_layout::vnni_tiled
                                                 : reg_layout::tiled;
    using matB_tile_desc_t = subgroup::tile_desc_t<tile_size_x_b, tile_size_y_b,
            block_size_x_b, block_size_y_b, reg_layout_b>;
    using matB_t = subgroup::tile_t<dtype_b, matB_tile_desc_t>;

    using cooperative_helper_B_t = subgroup::cooperative_load_helper_t<matB_t,
            mem_layout_b, tile_shape::wg_size_y, arch_tag>;
    using cooperative_tile_desc_B_t =
            typename cooperative_helper_B_t::co_tile_desc_t;

    using partial_matB_t = subgroup::tile_t<dtype_b, cooperative_tile_desc_B_t>;

    using matB_payload_t = subgroup::mem_payload_t<mem_desc_b_t,
            cooperative_tile_desc_B_t,
            is_local_b ? msg_type::scatter : msg_type::unaligned_2d, arch_tag>;

    using matB_payload_local_st_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_b, mem_layout::row_major, mem_space::local>,
            cooperative_tile_desc_B_t, msg_type::scatter, arch_tag>;
    using matB_payload_local_ld_t = subgroup::mem_payload_t<
            mem_desc_t<dtype_b, mem_layout::row_major, mem_space::local>,
            matB_tile_desc_t, msg_type::scatter, arch_tag>;

    using matB_acc_t = subgroup::tile_t<dtype_mma_b, matB_tile_desc_t>;
    using matB_prefetch_payload_t = subgroup::prefetch_payload_t<mem_desc_b_t,
            subgroup::tile_desc_t<tile_size_x_b, tile_size_y_b, 1, 1>,
            wg_size_y, arch_tag>;

public:
    using matAcc_tile_desc_t = subgroup::tile_desc_t<tile_size_x_c,
            tile_size_y_c, block_size_x_b, block_size_y_a, reg_layout::tiled>;
    using matAcc_t = subgroup::tile_t<dtype_mma_acc, matAcc_tile_desc_t>;

private:
    using tile_mma = subgroup::tile_mma_t<matAcc_t, matAcc_t, matB_acc_t,
            matA_acc_t, mma_engine::xmx, arch_tag>;
    // static constexpr bool enable_periodic_sync = (sync_freq != 0);
    static constexpr uint32_t barrier_count_x = wg_size_y > 1 ? wg_size_x : 0;
    static constexpr uint32_t barrier_count_y = wg_size_x > 1 ? wg_size_y : 0;
    static constexpr uint32_t tile_size_a
            = tile_size_x_a * tile_size_y_a * sizeof(dtype_a);
    static constexpr uint32_t tile_size_b
            = tile_size_x_b * tile_size_y_b * sizeof(dtype_b);
    static constexpr uint32_t slm_size_a = wg_size_y * tile_size_a;
    static constexpr uint32_t slm_size_b = wg_size_x * tile_size_b;

public:
    static constexpr uint32_t barrier_count = barrier_count_x + barrier_count_y;

    static constexpr uint32_t slm_size = (slm_size_a + slm_size_b) * num_cyclic;
    static constexpr uint32_t slm_base_a = 0;
    static constexpr uint32_t slm_base_b = 0 + slm_size_a * num_cyclic;

    static constexpr msg_type msg_type_a = matA_payload_t::message_type;
    static constexpr msg_type msg_type_b = matB_payload_t::message_type;

    using pre_processing_arg_t = typename pre_processing_t::arguments_t;

    /// @brief Arguments for gemm.
    /// User should prepare matA_base_desc, matB_base_desc, inner_loop_count...
    struct arguments_t {
        /// @brief Is the memory description of matA, including base, shape and coordinate.
        mem_desc_a_t matA_base_desc;
        /// @brief Is the memory description of matB, including base, shape and coordinate.
        mem_desc_b_t matB_base_desc;
        /// @brief Is the total inner loop count required to compute the entire K-dim.
        uint32_t inner_loop_count;
        /// @brief Is the arguments for pre-processing functor.
        pre_processing_arg_t pre_processing_args;

        /// @brief Default construct.
        inline arguments_t() = default;
        // Be aware of the risks: Rule of three (copy constructor, copy assignment, destructor)
        // Please check if you need to add self-define destructor
        // ~arguments_t(){}

        /// @brief Constructs a new arguments t object.
        /// @param matA_desc Is the memory description of matA, including base, shape and coordinate.
        /// @param matB_desc Is the memory description of matB, including base, shape and coordinate.
        /// @param loop_count Is the total inner loop count required to compute the entire K-dim.
        /// @param args Is the arguments for pre-processing functor.
        inline arguments_t(mem_desc_a_t matA_desc, mem_desc_b_t matB_desc,
                uint32_t loop_count, pre_processing_arg_t args = {})
            : matA_base_desc(matA_desc)
            , matB_base_desc(matB_desc)
            , inner_loop_count(loop_count)
            , pre_processing_args(args) {}
        // Be aware of the risks: Rule of three (copy constructor, copy assignment, destructor)
        // Please check if you need to add self-define destructor
        // inline ~arguments_t(){}
        inline arguments_t(const arguments_t &args)
            : matA_base_desc(args.matA_base_desc)
            , matB_base_desc(args.matB_base_desc)
            , inner_loop_count(args.inner_loop_count)
            , pre_processing_args(args.pre_processing_args) {}
        inline arguments_t &operator=(const arguments_t &args) {
            this->matA_base_desc = args.matA_base_desc;
            this->matB_base_desc = args.matB_base_desc;
            this->inner_loop_count = args.inner_loop_count;
            this->pre_processing_args = args.pre_processing_args;
            return *this;
        }

        /// @brief Explicit initialization function.
        /// @param matA_desc Is the memory description of matA, including base, shape and coordinate.
        /// @param matB_desc Is the memory description of matB, including base, shape and coordinate.
        /// @param loop_count Is the total inner loop count required to compute the entire K-dim.
        /// @param args Is the arguments for pre-processing functor.
        inline void init(mem_desc_a_t matA_desc, mem_desc_b_t matB_desc,
                uint32_t loop_count, pre_processing_arg_t args = {}) {
            matA_base_desc = matA_desc;
            matB_base_desc = matB_desc;
            inner_loop_count = loop_count;
            pre_processing_args = args;
        }
    };

    /// @brief Gets the subgroup-level tile offset x.
    /// @param g Is the workgroup of the current tile.
    /// @return Subgroup-level tile offset x.
    __XETLA_API static int get_matC_offset_x(work_group_t &g) {
        int32_t sg_idx = g.get_id() % wg_size_x;
        return sg_idx * sg_tile_n;
    }

    /// @brief Gets the subgroup-level tile offset y.
    /// @param g Is the workgroup of the current tile.
    /// @return Subgroup-level tile offset y.
    __XETLA_API static int get_matC_offset_y(work_group_t &g) {
        int32_t sg_idy = g.get_id() / wg_size_x;
        return sg_idy * sg_tile_m;
    }

    XETLA_MARKER(
            "This release function will wait until all the  r/w and nbarrier "
            "id used in this gemm have been committed. By default, it will "
            "use barrier_id 0 to do the entire workgroup sync if wg_size > 1. "
            "If you call this function, please set a free barrier id or make "
            "sure barrier_id 0 is not being occupied and you need to allocate "
            "one more barrier count in addition to the gemm barrier counts.")
    __XETLA_API static void release(uint8_t nbarrier_id = 0) {
        static constexpr bool need_local_fence
                = (mem_space_a == mem_space::local)
                || (mem_space_b == mem_space::local);
        if constexpr (need_local_fence) {
            xetla_fence<memory_kind::shared_local>();
        }
        xetla_fence<memory_kind::untyped_global>();
        static constexpr uint32_t wg_size = wg_size_x * wg_size_y;
        if constexpr (wg_size > 1) {
            xetla_nbarrier_t<wg_size, wg_size, arch_tag> nbarrier;
            nbarrier.init_nbarrier(
                    nbarrier_id, nbarrier_role::producer_consumer);
            nbarrier.arrive_wait();
        }
    }

    /// @brief Main execution function for gemm.
    /// The basic process is load data -> matrix multiply.
    /// @param g Is the workgroup of the current tile.
    /// @param matAcc Is the reference of the accumulation buffer.
    /// @param args Is the gemm::arguments_t.
    /// @param slm_base Is the slm base address.
    /// @param nbarrier_base Is the named barrier base.
    __XETLA_API KERNEL_FUNC void operator()(work_group_t &g, matAcc_t &matAcc,
            arguments_t args, uint32_t slm_base = 0,
            uint32_t nbarrier_base = 0) {
        int32_t sg_idx = g.get_id() % wg_size_x;
        int32_t sg_idy = g.get_id() / wg_size_x;

        XETLA_ASSERT(g.get_id() < (wg_size_x * wg_size_y),
                "Thread id(%d) should less than wg_size(%d)", g.get_id(),
                wg_size_x * wg_size_y);

        update_sg_tile_tdesc(args, sg_idx, sg_idy);
        pre_processing_t pre_processing;
        matA_t matA;
        matB_t matB;
        partial_matA_t partial_matA;
        partial_matB_t partial_matB;
        //  >>>>>>>>>>>>>>>>>> pre_processing init
        pre_processing.init(g, args.pre_processing_args);
        uint32_t base_A = slm_base_a + sg_idy * tile_size_a;
        uint32_t base_B = slm_base_b + sg_idx * tile_size_b;

        uint32_t store_idx = 0;
        uint32_t load_idx = 0;

        matA_payload_t matA_payload(args.matA_base_desc);
        matA_payload_local_st_t matA_local_st_payload(base_A, tile_size_x_a,
                tile_size_y_a, tile_size_x_a,
                cooperative_helper_A_t::get_offset_x(sg_idx),
                cooperative_helper_A_t::get_offset_y(sg_idx));
        matA_payload_local_ld_t matA_local_ld_payload(
                base_A, tile_size_x_a, tile_size_y_a, tile_size_x_a, 0, 0);

        matB_payload_t matB_payload(args.matB_base_desc);
        matB_payload_local_st_t matB_local_st_payload(base_B, tile_size_x_b,
                tile_size_y_b, tile_size_x_b,
                cooperative_helper_B_t::get_offset_x(sg_idy),
                cooperative_helper_B_t::get_offset_y(sg_idy));
        matB_payload_local_ld_t matB_local_ld_payload(
                base_B, tile_size_x_b, tile_size_y_b, tile_size_x_b, 0, 0);

        matA_prefetch_payload_t matA_prefetch_payload(
                args.matA_base_desc, sg_idx);
        matB_prefetch_payload_t matB_prefetch_payload(
                args.matB_base_desc, sg_idy);

        xetla_nbarrier_t<wg_size_x, wg_size_x, arch_tag> nbarrier_a;
        nbarrier_a.init_nbarrier(
                sg_idy + nbarrier_base, nbarrier_role::producer_consumer);
        xetla_nbarrier_t<wg_size_y, wg_size_y, arch_tag> nbarrier_b;
        nbarrier_b.init_nbarrier(sg_idx + barrier_count_y + nbarrier_base,
                nbarrier_role::producer_consumer);

        tile_load(partial_matA, matA_payload);
        tile_load(partial_matB, matB_payload);

        tile_store(partial_matA, matA_local_st_payload);
        tile_store(partial_matB, matB_local_st_payload);
        store_idx++;

        matA_payload.template update_tdesc<update_dir_a>(matA_t::tile_size_x);
        matB_payload.template update_tdesc<update_dir_b>(matB_t::tile_size_y);
        xetla_fence<memory_kind::shared_local>();
        nbarrier_a.arrive();
        nbarrier_b.arrive();
#pragma unroll
        for (int i = 1; i < num_cyclic - 1; i++) {
            tile_load(partial_matA, matA_payload);
            tile_load(partial_matB, matB_payload);

            matA_payload.template update_tdesc<update_dir_a>(
                    matA_t::tile_size_x);
            matB_payload.template update_tdesc<update_dir_b>(
                    matB_t::tile_size_y);

            matA_local_st_payload
                    .template update_tdesc<tdesc_update_dir::y_dir>(
                            wg_size_y * matA_t::tile_size_y);
            matB_local_st_payload
                    .template update_tdesc<tdesc_update_dir::y_dir>(
                            wg_size_x * matB_t::tile_size_y);

            tile_store(partial_matA, matA_local_st_payload);
            tile_store(partial_matB, matB_local_st_payload);
            store_idx++;
        }

        matA_prefetch_payload.template update_tdesc<update_dir_a>(
                matA_t::tile_size_x * (num_cyclic - 1));
        matB_prefetch_payload.template update_tdesc<update_dir_b>(
                matB_t::tile_size_y * (num_cyclic - 1));
#pragma unroll
        for (int i = 0; i < stages; i++) {
            subgroup::tile_prefetch<cache_hint::cached, cache_hint::cached>(
                    matA_prefetch_payload);
            subgroup::tile_prefetch<cache_hint::cached, cache_hint::cached>(
                    matB_prefetch_payload);
            matA_prefetch_payload.template update_tdesc<update_dir_a>(
                    matA_t::tile_size_x);
            matB_prefetch_payload.template update_tdesc<update_dir_b>(
                    matB_t::tile_size_y);
        }

        for (int i = 0; i < args.inner_loop_count; i++) {
            tile_load(partial_matA, matA_payload);
            tile_load(partial_matB, matB_payload);

            matA_payload.template update_tdesc<update_dir_a>(
                    matA_t::tile_size_x);
            matB_payload.template update_tdesc<update_dir_b>(
                    matB_t::tile_size_y);

            if constexpr (stages != 0) {
                subgroup::tile_prefetch<cache_hint::cached, cache_hint::cached>(
                        matA_prefetch_payload);
                subgroup::tile_prefetch<cache_hint::cached, cache_hint::cached>(
                        matB_prefetch_payload);
            }

            nbarrier_a.wait();
            nbarrier_b.wait();

            tile_load(matA, matA_local_ld_payload);
            tile_load(matB, matB_local_ld_payload);

            load_idx = (load_idx < num_cyclic - 1) ? (load_idx + 1) : 0;

            if (load_idx != 0) {
                matA_local_ld_payload
                        .template update_tdesc<tdesc_update_dir::y_dir>(
                                wg_size_y * matA_t::tile_size_y);
                matB_local_ld_payload
                        .template update_tdesc<tdesc_update_dir::y_dir>(
                                wg_size_x * matB_t::tile_size_y);
            } else {
                matA_local_ld_payload
                        .template update_tdesc<tdesc_update_dir::y_dir>(
                                (1 - num_cyclic) * wg_size_y
                                * matA_t::tile_size_y);
                matB_local_ld_payload
                        .template update_tdesc<tdesc_update_dir::y_dir>(
                                (1 - num_cyclic) * wg_size_x
                                * matB_t::tile_size_y);
            }
            xetla_fence<memory_kind::shared_local>();

            if constexpr (stages != 0) {
                matA_prefetch_payload.template update_tdesc<update_dir_a>(
                        matA_t::tile_size_x);
                matB_prefetch_payload.template update_tdesc<update_dir_b>(
                        matB_t::tile_size_y);
            }

            nbarrier_a.arrive();
            nbarrier_b.arrive();
            SW_BARRIER();
            matA_acc_t matA_acc;
            matB_acc_t matB_acc;
            subgroup::elemwise_cvt(matA_acc, matA);
            subgroup::vnni_transform(matB_acc, matB);
            pre_processing(matA_acc, matB_acc, matA, matB);
            SW_BARRIER();
            tile_mma::mma(matAcc, matAcc, matB_acc, matA_acc);
            SW_BARRIER();

            if (store_idx != 0) {
                matA_local_st_payload
                        .template update_tdesc<tdesc_update_dir::y_dir>(
                                wg_size_y * matA_t::tile_size_y);
                matB_local_st_payload
                        .template update_tdesc<tdesc_update_dir::y_dir>(
                                wg_size_x * matB_t::tile_size_y);
            } else {
                matA_local_st_payload
                        .template update_tdesc<tdesc_update_dir::y_dir>(
                                (1 - num_cyclic) * wg_size_y
                                * matA_t::tile_size_y);
                matB_local_st_payload
                        .template update_tdesc<tdesc_update_dir::y_dir>(
                                (1 - num_cyclic) * wg_size_x
                                * matB_t::tile_size_y);
            }

            tile_store(partial_matA, matA_local_st_payload);
            tile_store(partial_matB, matB_local_st_payload);
            store_idx = (store_idx < num_cyclic - 1) ? (store_idx + 1) : 0;
        }
        SW_BARRIER();
        nbarrier_a.wait();
        nbarrier_b.wait();
    }

private:
    /// @brief Updates tile base descriptor based on the tid.
    __XETLA_API static void update_sg_tile_tdesc(
            arguments_t &args, int32_t sg_idx, int32_t sg_idy) {
        int32_t tile_offset_n = sg_idx * sg_tile_n;
        int32_t tile_offset_m = sg_idy * sg_tile_m;

        args.matA_base_desc.update_coord_y(
                tile_offset_m + cooperative_helper_A_t::get_offset_y(sg_idx));
        args.matA_base_desc.update_coord_x(
                cooperative_helper_A_t::get_offset_x(sg_idx));
        args.matB_base_desc.update_coord_x(
                tile_offset_n + cooperative_helper_B_t::get_offset_x(sg_idy));
        args.matB_base_desc.update_coord_y(
                cooperative_helper_B_t::get_offset_y(sg_idy));
    }
};

/// @} xetla_gemm

} // namespace gpu::xetla::group
