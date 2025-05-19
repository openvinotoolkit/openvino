/*******************************************************************************
* Copyright (c) 2022-2025 Intel Corporation
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

using namespace gpu::xetla;
using namespace gpu::xetla::group;
using namespace gpu::xetla::kernel;
using namespace gpu::xetla::subgroup;

template <typename compute_attr_, typename perf_tuning_knob_,
        typename tile_shape_, typename mem_desc_a_t_, typename mem_desc_b_t_,
        typename pre_processing_t_, gpu_arch arch_tag_, typename matA_in_t>
class gemm_lora_b_aligned_t {
public:
    using mem_desc_a_t = mem_desc_a_t_;
    using mem_desc_b_t = mem_desc_b_t_;
    using tile_shape = tile_shape_;
    using pre_processing_t = pre_processing_t_;
    using compute_policy = compute_policy_default_xmx<compute_attr_,
            perf_tuning_knob_, arch_tag_>;
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

    static_assert(
            matA_in_t::tile_desc::tile_size_x % matA_tile_desc_t::tile_size_x
                    == 0,
            "matA_in_t tile size x should be the same as matAcc_tile_desc_t");
    static_assert(
            matA_tile_desc_t::tile_size_y == matA_in_t::tile_desc::tile_size_y,
            "matA_in_t tile size y should be the same as matAcc_tile_desc_t");
    static_assert(matA_tile_desc_t::block_size_x
                    == matA_in_t::tile_desc::block_size_x,
            "matA_in_t block size x should be the same as matAcc_tile_desc_t");
    static_assert(matA_tile_desc_t::block_size_y
                    == matA_in_t::tile_desc::block_size_y,
            "matA_in_t block size y should be the same as matAcc_tile_desc_t");

    using matA_t = subgroup::tile_t<dtype_a, matA_tile_desc_t>;
    using matA_payload_t = subgroup::mem_payload_t<mem_desc_a_t,
            matA_tile_desc_t,
            is_local_a ? msg_type::scatter : msg_type::block_2d, arch_tag>;
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
    using matB_payload_t = subgroup::mem_payload_t<mem_desc_b_t,
            matB_tile_desc_t,
            is_local_b ? msg_type::scatter : msg_type::block_2d, arch_tag>;
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
    static constexpr bool enable_periodic_sync = (sync_freq != 0);
    static constexpr uint32_t barrier_count_x = wg_size_y > 1 ? wg_size_x : 0;
    static constexpr uint32_t barrier_count_y = wg_size_x > 1 ? wg_size_y : 0;

public:
    static constexpr uint32_t barrier_count
            = enable_periodic_sync ? barrier_count_x + barrier_count_y : 0;

    static constexpr uint32_t slm_size = 0;

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
    inline static int get_matC_offset_x(work_group_t &g) {
        int32_t sg_idx = g.get_id() % wg_size_x;
        return sg_idx * sg_tile_n;
    }

    /// @brief Gets the subgroup-level tile offset y.
    /// @param g Is the workgroup of the current tile.
    /// @return Subgroup-level tile offset y.
    inline static int get_matC_offset_y(work_group_t &g) {
        int32_t sg_idy = g.get_id() / wg_size_x;
        return sg_idy * sg_tile_m;
    }

    inline static void release(uint8_t nbarrier_id = 0) {
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
    inline void operator()(work_group_t &g, matAcc_t &matAcc,
            arguments_t args, matA_in_t &matA_in, uint32_t slm_base = 0,
            uint32_t nbarrier_base = 0) {
        int32_t sg_idx = g.get_id() % wg_size_x;
        int32_t sg_idy = g.get_id() / wg_size_x;

        update_sg_tile_tdesc(args, sg_idx, sg_idy);
        pre_processing_t pre_processing;
        matA_t matA;
        matB_t matB;
        //  >>>>>>>>>>>>>>>>>> pre_processing init
        pre_processing.init(g, args.pre_processing_args);
        matB_payload_t matB_payload(args.matB_base_desc);
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

#pragma unroll
        for (int i = 0; i < stages; i++) {
            subgroup::tile_prefetch<cache_hint::cached, cache_hint::cached>(
                    matB_prefetch_payload);
            matB_prefetch_payload.template update_tdesc<update_dir_b>(
                    matB_t::tile_size_y);
        }

        for (int i = 0; i < args.inner_loop_count; i++) {
            if constexpr (enable_periodic_sync) {
                if ((i % sync_freq) == 0) {
                    if constexpr (wg_size_x > 1) { nbarrier_a.arrive(); }
                    if constexpr (wg_size_y > 1) { nbarrier_b.arrive(); }
                }
            }
            subgroup::tile_load<cache_hint::cached, cache_hint::cached>(
                    matB, matB_payload);

            static_assert(sizeof(dtype_a) == 2);
            static_assert(matA_t::tile_desc::tile_size_y
                            % matA_in_t::tile_desc::block_size_y
                    == 0);
            static_assert(matA_t::tile_desc::block_size_y
                    == matA_in_t::tile_desc::block_size_y);
            static_assert(matA_t::tile_desc::block_size_x
                    == matA_in_t::tile_desc::block_size_x);
            static constexpr uint32_t block_size
                    = matA_in_t::tile_desc::block_size_x
                    * matA_in_t::tile_desc::block_size_y;
            static_assert(matA_t::tile_desc::tile_size_x
                            % matA_in_t::tile_desc::block_size_x
                    == 0);
            static constexpr uint32_t copy_n_blocks
                    = matA_t::tile_desc::tile_size_x
                    / matA_in_t::tile_desc::block_size_x;
            static_assert(matA_in_t::tile_desc::tile_size_x
                            % matA_in_t::tile_desc::block_size_x
                    == 0);
            if constexpr (matA_t::tile_desc::tile_size_x
                    < matA_in_t::tile_desc::tile_size_x) {
#pragma unroll
                for (int j = 0; j < matA_t::tile_desc::tile_size_y
                                / matA_in_t::tile_desc::block_size_y;
                        j++) {
                    matA.reg.template xetla_select<block_size * copy_n_blocks,
                            1>(j * block_size * copy_n_blocks)
                            = matA_in.reg.template xetla_select<
                                    block_size * copy_n_blocks, 1>(j
                                            * block_size
                                            * matA_in_t::tile_desc::tile_size_x
                                            / matA_in_t::tile_desc::block_size_x
                                    + i * block_size * copy_n_blocks);
                }
            } else {
                matA.reg = matA_in.reg;
            }

            if constexpr (stages != 0) {
                subgroup::tile_prefetch<cache_hint::cached, cache_hint::cached>(
                        matB_prefetch_payload);
            }
            cm_fence(CM_SW_BARRIER);
            matB_payload.template update_tdesc<update_dir_b>(
                    matB_t::tile_size_y);
            if constexpr (stages != 0) {
                matB_prefetch_payload.template update_tdesc<update_dir_b>(
                        matB_t::tile_size_y);
            }
            cm_fence(CM_SW_BARRIER);
            matA_acc_t matA_acc;
            matB_acc_t matB_acc;
            subgroup::elemwise_cvt(matA_acc, matA);
            subgroup::vnni_transform(matB_acc, matB);
            pre_processing(matA_acc, matB_acc, matA, matB);
            cm_fence(CM_SW_BARRIER);
            tile_mma::mma(matAcc, matAcc, matB_acc, matA_acc);
            cm_fence(CM_SW_BARRIER);
            if constexpr (enable_periodic_sync) {
                if ((i % sync_freq) == 0) {
                    if constexpr (wg_size_x > 1) { nbarrier_a.wait(); }
                    if constexpr (wg_size_y > 1) { nbarrier_b.wait(); }
                }
            }
        }
        cm_fence(CM_SW_BARRIER);
    }

private:
    /// @brief Updates tile base descriptor based on the tid.
    inline static void update_sg_tile_tdesc(
            arguments_t &args, int32_t sg_idx, int32_t sg_idy) {
        int32_t tile_offset_n = sg_idx * sg_tile_n;
        int32_t tile_offset_m = sg_idy * sg_tile_m;

        args.matA_base_desc.update_coord_y(tile_offset_m);
        args.matB_base_desc.update_coord_x(tile_offset_n);
    }
};

template <typename gemm_t_, typename epilogue_t_, typename group_swizzle_,
        typename matA_in_t, uint32_t wg_tile_n_total>
class gemm_kernel_lora_b_t {
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
        /// @brief Is the leading dimension (pitch) size of the matrix A in memory.
        uint32_t matA_ld;
        /// @brief Is the leading dimension (pitch) size of the matrix B in memory.
        uint32_t matB_ld;
        /// @brief Is the leading dimension (pitch) size of the matrix C in memory.
        uint32_t matC_ld;
        /// @brief Is the base address of matrix A.
        matA_base_t matA_base;
        /// @brief Is the base address of matrix B.
        matB_base_t matB_base;
        /// @brief Is the base address of matrix C.
        matC_base_t matC_base;
        /// @brief Is the epilogue arguments.
        epilogue_args_t epilogue_args;

        /// @brief Constructs arguments with default method.
        inline arguments_t() = default;

        /// @brief Set for device copyable
        static constexpr bool host_callable = true;

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
            , epilogue_args(epilogue_args_) {}
        // Be aware of the risks: Rule of three (copy constructor, copy assignment, destructor)
        // Please check if you need to add self-define destructor
        // inline ~arguments_t(){}
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
            , epilogue_args(args.epilogue_args) {}
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
            this->epilogue_args = args.epilogue_args;
            return *this;
        }
    };

    /// @brief Gets named_barrier id consumption count.
    /// Users query and get a named_barrier id consumption count in compile time.
    /// @return The count of named barriers required.
    inline static constexpr uint32_t get_barrier_count() {
        constexpr uint32_t count
                = gemm_t::barrier_count + epilogue_t::barrier_count;
        static_assert(
                count <= 32, "The named_barrier count should be less than 32!");
        return count;
    }

    /// @brief Gets local memory size consumption.
    /// Users query and get a local memory consumption size in compile time.
    /// @return The size of local memory required.
    inline static constexpr uint32_t get_slm_size() {
        constexpr uint32_t size = gemm_t::slm_size + epilogue_t::slm_size;
        static_assert(size <= (128 * 1024),
                "The local memory size should be less than 128KB!");
        return size;
    };

    /// @brief Main execution function for GEMM_UNIVERSAL.
    /// The processing order is 1) set group-level base and boundary -> 2) gemm -> 3) epilogue.
    /// @param item Is the sycl::nd_item, returns execution related information, such as workgroup id, subgroup id...
    /// @param args Is the GEMM_UNIVERSAL arguments for application-related runtime variables.
    /// @param slm_base Is the slm base address.
    /// @param nbarrier_base Is the named barrier base.
    inline void operator()(sycl::nd_item<3> &item,
            const arguments_t &args, matA_in_t &matA, uint32_t iter = 0,
            uint32_t slm_base = 0, uint32_t nbarrier_base = 0) const {
        // set up workgroup level coordinates and boundaries
        group_swizzle_t group_swizzle;
        int start_m = group_swizzle.template get_tile_idx<1>(item) * wg_tile_m;
        int start_n
                = group_swizzle.template get_tile_idx<2>(item) * wg_tile_n_total
                + wg_tile_n * iter;
        int start_k = 0;
        uint32_t wg_tile_k = args.matrix_k;
        uint32_t boundary_n = (start_n + wg_tile_n) > args.matrix_n
                ? args.matrix_n
                : (start_n + wg_tile_n);
        uint32_t boundary_m = (start_m + wg_tile_m) > args.matrix_m
                ? args.matrix_m
                : (start_m + wg_tile_m);
        uint32_t boundary_k = wg_tile_k;

        uint32_t gemm_slm_base = slm_base;
        uint32_t gemm_nbarr_base = nbarrier_base;
        uint32_t epilogue_slm_base = gemm_slm_base + gemm_t::slm_size;
        uint32_t epilogue_nbarr_base = gemm_nbarr_base + gemm_t::barrier_count;

        // set up arguments
        work_group_t g;
        g.init(item.get_local_linear_id());
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
        //setup for matC
        if constexpr (mem_desc_c_t::is_local) {
            mem_desc_c.init(args.matC_base,
                    {real_wg_tile_n, real_wg_tile_m, real_wg_tile_n}, {0, 0});
        } else {
            mem_desc_c.init(args.matC_base,
                    {boundary_n, boundary_m, args.matC_ld}, {start_n, start_m});
        }
        uint32_t inner_loop_count = (wg_tile_k + k_stride - 1) / k_stride;
        gemm_args_t gemm_args(mem_desc_a, mem_desc_b, inner_loop_count);
        gemm_t gemm;
        epilogue_t epilogue;

        matAcc_t matAcc(0);
        gemm(g, matAcc, gemm_args, matA, gemm_slm_base, gemm_nbarr_base);
        epilogue(g, matAcc, mem_desc_c, args.epilogue_args, epilogue_slm_base,
                epilogue_nbarr_base);
    }
};

template <typename dtype_a, typename dtype_b, typename dtype_c,
        typename dtype_acc, uint32_t wg_m, uint32_t wg_n, uint32_t sg_m,
        uint32_t sg_n, uint32_t sg_k, uint32_t wg_tile_n_total,
        mem_layout layout_a, mem_layout layout_b, mem_layout layout_c,
        mem_space mem_space_a, mem_space mem_space_b, mem_space mem_space_c,
        uint32_t local_kslicing, mma_engine engine,
        uint32_t periodic_sync_interval, uint32_t prefetch_distance,
        gpu_arch arch_tag, bool unaligned, bool temp_in_reg, typename matA_in_t>
struct gemm_lora_b {
    using tile_shape = group::tile_shape_t<wg_n, wg_m, sg_n, sg_m>;

    using mem_desc_a
            = mem_desc_t<dtype_a, layout_a, mem_space_a, unaligned ? 1 : 8>;
    using mem_desc_b
            = mem_desc_t<dtype_b, layout_b, mem_space_b, unaligned ? 1 : 8>;
    using mem_desc_c
            = mem_desc_t<dtype_c, layout_c, mem_space_c, unaligned ? 1 : 8>;

    using compute_attr = typename std::conditional<engine == mma_engine::fpu,
            compute_attr_t<dtype_acc, dtype_acc, dtype_acc>,
            compute_attr_t<dtype_a, dtype_b, dtype_acc>>::type;

    using perf_tuning_knob = perf_tuning_knob_t<sg_k, prefetch_distance,
            periodic_sync_interval>;

    using compute_policy_0 =
            typename std::conditional<engine == mma_engine::fpu,
                    compute_policy_default_fpu<compute_attr, perf_tuning_knob,
                            arch_tag>,
                    compute_policy_default_xmx<compute_attr, perf_tuning_knob,
                            arch_tag>>::type;
    using compute_policy = typename std::conditional<unaligned,
            compute_policy_unaligned_xmx<compute_attr, perf_tuning_knob,
                    arch_tag>,
            compute_policy_0>::type;
    using pre_processing = pre_processing_default_t<tile_shape, arch_tag>;
#if LORA_TEMP_IN_REG == 1
    using gemm = gemm_lora_b_aligned_t<compute_attr, perf_tuning_knob,
            tile_shape, mem_desc_a, mem_desc_b, pre_processing, arch_tag,
            matA_in_t>;
#else
    using gemm = gemm_t<compute_policy, tile_shape, mem_desc_a, mem_desc_b,
            pre_processing>;
#endif

    LORA_POST_OP_DEFINITIONS

    using tile_op_t = subgroup::chained_tile_op_t<LORA_POST_OP_LIST>;
    using epilogue = epilogue_t<
            epilogue_policy_tile_op<tile_op_t, arch_tag,
                    unaligned ? msg_type::unaligned_2d : msg_type::block_2d>,
            tile_shape, mem_desc_c>;

    using epilogue_args_t = typename epilogue::arguments_t;

    using group_swizzle_t = kernel::group_swizzle_default<arch_tag>;

#if LORA_TEMP_IN_REG == 1
    using gemm_op_t = gemm_kernel_lora_b_t<gemm, epilogue, group_swizzle_t,
            matA_in_t, wg_tile_n_total>;
#else
    using gemm_op_t = kernel::gemm_universal_t<
            kernel::dispatch_policy_kslicing<group_swizzle_t, 1,
                    local_kslicing>,
            gemm, epilogue>;
#endif
    static constexpr uint32_t barrier_count = gemm_op_t::get_barrier_count();
    static constexpr uint32_t slm_size = gemm_op_t::get_slm_size();

    inline static void run(sycl::nd_item<3> &item, uint32_t mat_m,
            uint32_t mat_k, uint32_t mat_n, dtype_a *a, dtype_b *b, dtype_c *c,
            uint32_t iter
#if LORA_TEMP_IN_REG == 1
            ,
            matA_in_t &matA
#endif
                    LORA_POST_OP_ARGS) {

        gemm_op_t gemm_op;

        uint32_t lda = layout_a == mem_layout::col_major ? mat_m : mat_k;
        uint32_t ldb = layout_b == mem_layout::col_major ? mat_k : mat_n;
        uint32_t ldc = layout_c == mem_layout::col_major ? mat_m : mat_n;

        LORA_POST_OP_SHAPE_DEFINITIONS
        epilogue_args_t epilogue_args;
        epilogue_args.init({LORA_POST_OP_EPILOGUE_INIT_ARGS});

        typename gemm_op_t::arguments_t arg(mat_m, mat_k, mat_n, a, lda, b, ldb,
                c, ldc,
#if LORA_TEMP_IN_REG == 0
                nullptr, nullptr,
#endif
                epilogue_args);

        gemm_op(item, arg
#if LORA_TEMP_IN_REG == 1

                ,
                matA, iter
#endif
        );
    }
};
