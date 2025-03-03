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

template <typename dtype_a, typename dtype_b, typename dtype_c,
        typename dtype_acc, uint32_t wg_m, uint32_t wg_n, uint32_t sg_m,
        uint32_t sg_n, uint32_t sg_k, mem_layout layout_a, mem_layout layout_b,
        mem_layout layout_c, mem_space mem_space_a, mem_space mem_space_b,
        mem_space mem_space_c, mma_engine engine, uint32_t local_kslicing,
        gpu_arch arch_tag>
struct gemm_universal {
    using tile_shape = group::tile_shape_t<wg_n, wg_m, sg_n, sg_m>;
    static constexpr uint32_t periodic_sync_interval = 0;
    static constexpr uint32_t prefetch_distance = 2;

    using gemm_t = typename group::gemm_selector_t<dtype_a, dtype_b, layout_a,
            layout_b, mem_space_a, mem_space_b, 8, 8, dtype_acc, tile_shape,
            sg_k, mma_engine::xmx, arch_tag, prefetch_distance,
            periodic_sync_interval>::gemm;

    using bias_op_t = subgroup::bias_add_op_t<dtype_b, arch_tag>;
    using tile_op_t = subgroup::chained_tile_op_t<bias_op_t>;
    using epilogue_t = epilogue_t<epilogue_policy_tile_op<tile_op_t, arch_tag>,
            tile_shape, mem_desc_t<dtype_c, layout_c, mem_space_c>>;
    using epilogue_args_t = typename epilogue_t::arguments_t;

    using group_swizzle_t = kernel::group_swizzle_default<arch_tag>;

    using gemm_op_t = kernel::gemm_universal_t<
            kernel::dispatch_policy_kslicing<group_swizzle_t, 1,
                    local_kslicing>,
            gemm_t, epilogue_t>;

    using gemm_acc_t = typename gemm_t::matAcc_t;

    static constexpr uint32_t barrier_count = gemm_op_t::get_barrier_count();
    static constexpr uint32_t slm_size = gemm_op_t::get_slm_size();

    inline void operator()(sycl::nd_item<3> &item, dtype_a *a,
            dtype_b *b, dtype_b *bias,
            typename epilogue_t::mem_desc_c_t::base_t c, uint32_t mat_m,
            uint32_t mat_n, uint32_t mat_k, uint32_t lda, uint32_t ldb,
            uint32_t ldc) {
        gemm_op_t gemm_op;

        typename bias_op_t::shape_t bias_add_shape(mat_n, 1, mat_n);
        epilogue_args_t epilogue_args;

        uint64_t matrix_size_a = mat_m * mat_k;
        uint64_t matrix_size_b = mat_k * mat_n;
        uint64_t matrix_size_c = mat_m * mat_n;

        epilogue_args.init({{bias, bias_add_shape}});
        typename gemm_op_t::arguments_t arg(mat_m, mat_k, mat_n, a,
                layout_a == mem_layout::col_major ? mat_m : mat_k, b,
                layout_b == mem_layout::col_major ? mat_k : mat_n, c.base,
                mat_n, nullptr, nullptr, epilogue_args);
        gemm_op(item, arg);
    }
};

template <typename dtype_a, typename dtype_b, typename dtype_c,
        typename dtype_acc, uint32_t wg_m, uint32_t wg_n, uint32_t sg_m,
        uint32_t sg_n, uint32_t sg_k, mem_layout mem_layout_a,
        mem_layout mem_layout_b, mem_layout mem_layout_c, mem_space mem_space_a,
        mem_space mem_space_b, mem_space mem_space_c, mma_engine engine,
        uint32_t k_size, gpu_arch arch_tag>
struct gemm_persistent {
    static_assert(engine == mma_engine::fpu);
    static_assert(group::detail::check_2d_block_pitch_alignment<dtype_a,
            dtype_b, 8, 8, arch_tag>::value);

    static constexpr uint32_t periodic_sync_interval = 0;
    static constexpr uint32_t prefetch_distance = 0;
    static constexpr uint32_t barrier_count = 0;
    static constexpr uint32_t slm_size = 0;

    using tile_shape = group::tile_shape_t<wg_n, wg_m, sg_n, sg_m>;
    using work_group_t = typename tile_shape::work_group_t;

    using mem_desc_a_t = mem_desc_t<dtype_a, mem_layout_a, mem_space_a, 8>;
    using mem_desc_b_t = mem_desc_t<dtype_b, mem_layout_b, mem_space_b, 8>;

    using compute_attr_t
            = group::compute_attr_t<dtype_acc, dtype_acc, dtype_acc>;
    using perf_tuning_knob_t = group::perf_tuning_knob_t<sg_k,
            prefetch_distance, periodic_sync_interval>;
    using compute_policy_t = group::compute_policy_default_fpu<compute_attr_t,
            perf_tuning_knob_t, arch_tag>;

    using group_swizzle_t = kernel::group_swizzle_default<arch_tag>;

    using dtype_mma_acc = typename compute_policy_t::dtype_mma_acc;
    using dtype_mma_a = typename compute_policy_t::dtype_mma_a;
    using dtype_mma_b = typename compute_policy_t::dtype_mma_b;

    using check_dtype
            = group::gemm<gpu_arch::Xe>::default_fpu::check_dtype_default<
                    dtype_a, dtype_b, dtype_mma_a, dtype_mma_b, dtype_mma_acc>;

    using check_memory
            = group::gemm<gpu_arch::Xe>::default_fpu::check_memory_default<
                    mem_layout_a, mem_layout_b, mem_space_a, mem_space_b>;

    static constexpr uint32_t tile_size_x_a = compute_policy_t::k_stride;
    static constexpr uint32_t tile_size_y_a = tile_shape::sg_tile_size_y;
    static constexpr uint32_t tile_size_x_b = tile_shape::sg_tile_size_x;
    static constexpr uint32_t tile_size_y_b = compute_policy_t::k_stride;
    static constexpr uint32_t tile_size_x_c = tile_shape::sg_tile_size_x;
    static constexpr uint32_t tile_size_y_c = tile_shape::sg_tile_size_y;

    static constexpr uint32_t block_size_x_a
            = (compute_policy_t::block_size_x_a > tile_size_x_a)
            ? tile_size_x_a
            : compute_policy_t::block_size_x_a;
    static constexpr uint32_t block_size_y_a
            = (compute_policy_t::block_size_y_a > tile_size_y_a)
            ? tile_size_y_a
            : compute_policy_t::block_size_y_a;
    static constexpr uint32_t block_size_x_b
            = (compute_policy_t::block_size_x_b > tile_size_x_b)
            ? tile_size_x_b
            : compute_policy_t::block_size_x_b;
    static constexpr uint32_t block_size_y_b
            = (compute_policy_t::block_size_y_b > tile_size_y_b)
            ? tile_size_y_b
            : compute_policy_t::block_size_y_b;

    static constexpr reg_layout reg_layout_a = reg_layout::tiled;
    using matA_tile_desc_t
            = subgroup::tile_desc_t<k_size, 1, k_size, 1, reg_layout_a>;
    using matA_t = subgroup::tile_t<dtype_a, matA_tile_desc_t>;
    using matA_payload_t = subgroup::mem_payload_t<mem_desc_a_t,
            matA_tile_desc_t,
            subgroup::msg_type_v<matA_tile_desc_t, mem_space_a>, arch_tag>;
    using matA_acc_t = subgroup::tile_t<dtype_mma_a, matA_tile_desc_t>;

    static constexpr reg_layout reg_layout_b = reg_layout::tiled;
    using matB_tile_desc_t = subgroup::tile_desc_t<tile_size_x_b, tile_size_y_b,
            block_size_x_b, block_size_y_b, reg_layout_b>;
    using matB_t = subgroup::tile_t<dtype_b, matB_tile_desc_t>;
    using matB_payload_t = subgroup::mem_payload_t<mem_desc_b_t,
            matB_tile_desc_t, msg_type::block_2d, arch_tag>;
    using matB_acc_t = subgroup::tile_t<dtype_mma_b, matB_tile_desc_t>;

    using matAcc_tile_desc_t = subgroup::tile_desc_t<tile_size_x_c,
            tile_size_y_c, block_size_x_b, block_size_y_a, reg_layout::tiled>;
    using matAcc_t = subgroup::tile_t<dtype_mma_acc, matAcc_tile_desc_t>;

    using tile_mma = subgroup::tile_mma_t<matAcc_t, matAcc_t, matB_acc_t,
            matA_acc_t, mma_engine::fpu, arch_tag>;

    static constexpr bool is_col_major_a
            = mem_layout_a == mem_layout::col_major;
    static constexpr bool is_col_major_b
            = mem_layout_b == mem_layout::col_major;

    static constexpr tdesc_update_dir update_dir_a = is_col_major_a
            ? tdesc_update_dir::y_dir
            : tdesc_update_dir::x_dir;
    static constexpr tdesc_update_dir update_dir_b = is_col_major_b
            ? tdesc_update_dir::x_dir
            : tdesc_update_dir::y_dir;

    mem_desc_a_t mem_desc_a;
    static_assert(k_size % compute_policy_t::k_stride == 0);
    static constexpr uint32_t k_steps = k_size / compute_policy_t::k_stride;
    matB_acc_t matB_acc[k_steps];

    inline void init(sycl::nd_item<3> &item,
            typename mem_desc_a_t::base_t a, dtype_b *b, uint32_t mat_m,
            uint32_t mat_n, uint32_t mat_k, uint32_t lda, uint32_t ldb,
            uint32_t ldc) {
        work_group_t g(item.get_local_linear_id() % work_group_t::size);
        uint32_t wg_id = item.get_local_linear_id() / work_group_t::size;

        group_swizzle_t group_swizzle;
        int start_m = group_swizzle.template get_tile_idx<1>(item)
                * tile_shape::wg_tile_size_y;
        int start_n = group_swizzle.template get_tile_idx<2>(item)
                * tile_shape::wg_tile_size_x;
        int start_k = 0;

        uint32_t wg_tile_k = mat_k;
        uint32_t boundary_n = (start_n + tile_shape::wg_tile_size_x)
                        > tile_shape::wg_tile_size_x
                ? mat_n
                : (start_n + tile_shape::wg_tile_size_x);
        uint32_t boundary_m = (start_m + tile_shape::wg_tile_size_y)
                        > tile_shape::wg_tile_size_y
                ? mat_m
                : (start_m + tile_shape::wg_tile_size_y);
        uint32_t boundary_k = wg_tile_k;

        mem_desc_b_t mem_desc_b;

        mem_desc_a.init(a, {boundary_k, boundary_m, lda}, {start_k, start_m});
        mem_desc_b.init(b, {boundary_n, boundary_k, ldb}, {start_n, start_k});

        int32_t sg_idx = g.get_id() % tile_shape::wg_size_x;
        int32_t sg_idy = g.get_id() / tile_shape::wg_size_x;
        int32_t tile_offset_n = sg_idx * tile_shape::sg_tile_size_x;
        int32_t tile_offset_m = sg_idy * tile_shape::sg_tile_size_y;

        mem_desc_a.update_coord_y(tile_offset_m);
        mem_desc_b.update_coord_x(tile_offset_n);

        matB_payload_t matB_payload(mem_desc_b);
        matB_t matB[k_steps];

#pragma unroll
        for (int i = 0; i < k_steps; i++) {
            subgroup::tile_load<cache_hint::uncached, cache_hint::uncached>(
                    matB[i], matB_payload);
            matB_payload.template update_tdesc<update_dir_b>(
                    matB_t::tile_size_y);
            subgroup::elemwise_cvt(matB_acc[i], matB[i]);
        }

#pragma unroll
        for (int i = 0; i < k_steps; i++) {
            subgroup::tile_store<cache_hint::uncached, cache_hint::uncached>(
                    matB_acc[i], matB_payload);
        }
    }

    inline void run(matAcc_t &result) {

        matAcc_t part_res[k_steps];
        matA_t matA;
        matA_acc_t matA_acc;
        matA_payload_t matA_payload(mem_desc_a);

        subgroup::tile_load<cache_hint::cached, cache_hint::cached>(
                matA, matA_payload);
        subgroup::elemwise_cvt(matA_acc, matA);

        cm_fence(CM_SW_BARRIER);

#pragma unroll
        for (int i = 0; i < k_steps; i++) {
            part_res[i].init(0);
#pragma unroll
            for (int j = 0; j < sg_k; j++) {
                vector<float, k_size> tempA = matA_acc.reg;
                float a = tempA[j + i * sg_k];
                vector<float, sg_n *sg_k> tempB = matB_acc[i].reg;
                vector<float, sg_n> tempB_simd
                        = tempB.select<sg_n, 1>(j * sg_n);
                part_res[i].reg += a * tempB_simd;
            }
        }
        cm_fence(CM_SW_BARRIER);
        result.init(0);
#pragma unroll
        for (int i = 0; i < k_steps; i++) {
            result.reg += part_res[i].reg;
        }
    }
};

template <typename dtype_a, typename dtype_b, typename dtype_c,
        typename dtype_acc, uint32_t input_size, uint32_t hidden_size,
        uint32_t directions, mem_layout mem_layout_x, mem_layout mem_layout_w,
        mem_layout mem_layout_out, mem_space mem_space_x, mem_space mem_space_w,
        mem_space mem_space_out, gpu_arch arch_tag>
struct __xetla_kernel_lstm_gemm {
    static_assert(directions == 1 || directions == 2);

    static constexpr uint32_t wg_m = 40;
    static constexpr uint32_t wg_n = 256;

    static constexpr uint32_t sg_m = 24;
    static constexpr uint32_t sg_n = 32;
    static constexpr uint32_t sg_k = 32;
    static constexpr uint32_t local_kslicing = 1;

    using gemm_t = gemm_universal<dtype_a, dtype_b, dtype_c, dtype_acc, wg_m,
            wg_n, sg_m, sg_n, sg_k, mem_layout_x, mem_layout_w, mem_layout_x,
            mem_space_x, mem_space_w, mem_space_x, mma_engine::xmx,
            local_kslicing, arch_tag>;

    static inline void init() {
        if constexpr (gemm_t::barrier_count != 0) {
            xetla_nbarrier_init<gemm_t::barrier_count>();
        }
        if constexpr (gemm_t::slm_size != 0) {
            xetla_local_init<gemm_t::slm_size>();
        }
    }

    static inline void run(sycl::nd_item<3> &item, dtype_a *x, dtype_b *W,
            dtype_b *B, int *sequence_lengths, dtype_acc *out) {
        uint32_t seq_len = sequence_lengths[0];
        uint32_t matrix_m = seq_len;
        uint32_t matrix_k = input_size;
        uint32_t matrix_n = hidden_size * 4;

        size_t lda
                = mem_layout_x == mem_layout::col_major ? matrix_m : matrix_k;
        size_t ldb
                = mem_layout_w == mem_layout::col_major ? matrix_k : matrix_n;
        size_t ldc = matrix_n;

        int dir_id = item.get_group(0);

        dtype_b *w_dir = W + matrix_k * matrix_n * dir_id;
        dtype_b *bias_dir = B + matrix_n * dir_id;
        dtype_c *x_dir = out + matrix_m * matrix_n * dir_id;

        gemm_t gemm = gemm_t();
        gemm(item, x, w_dir, bias_dir, x_dir, matrix_m, matrix_n, matrix_k, lda,
                ldb, ldc);
    }
};

template <typename dtype_a, typename dtype_b, typename dtype_c,
        typename dtype_acc, uint32_t input_size, uint32_t hidden_size,
        uint32_t directions, mem_layout mem_layout_x, mem_layout mem_layout_w,
        mem_layout mem_layout_out, mem_space mem_space_x, mem_space mem_space_w,
        mem_space mem_space_out, gpu_arch arch_tag>
struct __xetla_kernel_lstm_loop {
    static_assert(directions == 1 || directions == 2);
    static_assert(hidden_size == 128);
    static_assert(hidden_size % 4 == 0);

    static constexpr uint32_t SIMD = 16;
    static constexpr uint32_t num_threads = 32;
    static constexpr uint32_t gates = 4;

    static constexpr uint32_t wg_m = 1;
    static constexpr uint32_t wg_n = hidden_size * gates;

    static constexpr uint32_t sg_m = 1;
    static constexpr uint32_t sg_n = SIMD;
    static constexpr uint32_t sg_k = hidden_size / 4;

    static constexpr uint32_t matrix_m = 1;
    static constexpr uint32_t matrix_k = hidden_size;
    static constexpr uint32_t matrix_n = hidden_size * gates;

    static constexpr size_t lda
            = mem_layout_x == mem_layout::col_major ? matrix_m : matrix_k;
    static constexpr size_t ldb
            = mem_layout_w == mem_layout::col_major ? matrix_k : matrix_n;
    static constexpr size_t ldc = matrix_n;

    using gemm_t = gemm_persistent<dtype_a, dtype_b, dtype_acc, dtype_acc, wg_m,
            wg_n, sg_m, sg_n, sg_k, mem_layout::row_major, mem_layout_w,
            mem_layout::row_major, mem_space::local, mem_space_w,
            mem_space::local, mma_engine::fpu, hidden_size, arch_tag>;

    static constexpr uint32_t barrier_count = gemm_t::barrier_count + 1;
    static constexpr uint32_t context_size_bytes
            = hidden_size * sizeof(dtype_acc);
    static constexpr uint32_t figo_size = gates * hidden_size;
    static constexpr uint32_t figo_size_bytes = figo_size * sizeof(dtype_acc);
    static constexpr uint32_t slm_size
            = context_size_bytes * 2 + figo_size_bytes + gemm_t::slm_size;

    static inline void init() {
        if constexpr (barrier_count != 0) {
            xetla_nbarrier_init<barrier_count>();
        }
        if constexpr (slm_size != 0) { xetla_local_init<slm_size>(); }
    }

    static inline void run(sycl::nd_item<3> &item, dtype_a *x,
            dtype_b *initial_hidden_state, dtype_b *initial_cell_state,
            dtype_b *R, int *sequence_lengths, dtype_c *hidden_history,
            dtype_c *hidden_state, dtype_c *cell_state) {
        xetla_nbarrier_t<num_threads, num_threads, arch_tag> nbarrier;
        nbarrier.init_nbarrier(
                barrier_count - 1, nbarrier_role::producer_consumer);

        int dir_id = item.get_group(0);
        uint32_t seq_len = sequence_lengths[0];

        dtype_a *x_dir = x + seq_len * hidden_size * gates * dir_id;
        dtype_b *r_dir = R + matrix_k * matrix_n * dir_id;
        dtype_b *context_h_init = initial_hidden_state + hidden_size * dir_id;
        dtype_b *context_c_init = initial_cell_state + hidden_size * dir_id;
        dtype_c *context_h_out = hidden_state + hidden_size * dir_id;
        dtype_c *context_c_out = cell_state + hidden_size * dir_id;

        constexpr uint32_t slm_figo_offset = gemm_t::slm_size;
        constexpr uint32_t slm_h_offset = slm_figo_offset + figo_size_bytes;
        constexpr uint32_t slm_c_offset = slm_h_offset + context_size_bytes;

        gemm_t gemm = gemm_t();
        gemm.init(item, slm_h_offset, r_dir, matrix_m, matrix_n, matrix_k, lda,
                ldb, ldc);

        using tile_desc_t
                = subgroup::tile_desc_t<SIMD, 1, SIMD, 1, reg_layout::tiled>;

        using tile_acc_t = subgroup::tile_t<dtype_acc, tile_desc_t>;
        using tile_a_t = subgroup::tile_t<dtype_a, tile_desc_t>;
        using tile_b_t = subgroup::tile_t<dtype_b, tile_desc_t>;
        using tile_c_t = subgroup::tile_t<dtype_c, tile_desc_t>;

        using local_acc_payload_t = subgroup::mem_payload_t<
                mem_desc_t<dtype_acc, mem_layout::row_major, mem_space::local>,
                tile_desc_t,
                subgroup::msg_type_v<tile_desc_t, mem_space::local>, arch_tag>;

        using global_acc_payload_t = subgroup::mem_payload_t<
                mem_desc_t<dtype_acc, mem_layout::row_major, mem_space::global>,
                tile_desc_t,
                subgroup::msg_type_v<tile_desc_t, mem_space::global>, arch_tag>;

        using global_a_payload_t = subgroup::mem_payload_t<
                mem_desc_t<dtype_a, mem_layout::row_major, mem_space::global>,
                tile_desc_t,
                subgroup::msg_type_v<tile_desc_t, mem_space::global>, arch_tag>;

        using global_b_payload_t = subgroup::mem_payload_t<
                mem_desc_t<dtype_b, mem_layout::row_major, mem_space::global>,
                tile_desc_t,
                subgroup::msg_type_v<tile_desc_t, mem_space::global>, arch_tag>;

        using global_c_payload_t = subgroup::mem_payload_t<
                mem_desc_t<dtype_c, mem_layout::row_major, mem_space::global>,
                tile_desc_t,
                subgroup::msg_type_v<tile_desc_t, mem_space::global>, arch_tag>;

        const int seq_inc = 1 - (2 * dir_id);
        const int seq_start_y = (seq_len - 1) * dir_id;
        const uint32_t local_id = item.get_local_linear_id();
        const uint32_t minigroup_offset
                = (local_id % ((hidden_size + SIMD - 1) / SIMD)) * SIMD;

        tile_a_t x_tile;
        global_a_payload_t x_tile_payload;
        x_tile_payload.init(x_dir, figo_size, seq_len, figo_size,
                local_id * SIMD, seq_start_y);

        tile_acc_t h_tile;
        local_acc_payload_t h_tile_payload;
        h_tile_payload.init(
                slm_h_offset, hidden_size, 1, hidden_size, minigroup_offset, 0);

        tile_acc_t c_tile;
        local_acc_payload_t c_tile_payload;
        c_tile_payload.init(
                slm_c_offset, hidden_size, 1, hidden_size, minigroup_offset, 0);

        if (local_id < (hidden_size / SIMD)) {
            tile_b_t h_tile_init;
            global_b_payload_t h_tile_payload_init;
            h_tile_payload_init.init(context_h_init, hidden_size, 1,
                    hidden_size, minigroup_offset, 0);

            tile_b_t c_tile_init;
            global_b_payload_t c_tile_payload_init;
            c_tile_payload_init.init(context_c_init, hidden_size, 1,
                    hidden_size, minigroup_offset, 0);

            tile_load<cache_hint::uncached, cache_hint::uncached>(
                    h_tile_init, h_tile_payload_init);
            tile_load<cache_hint::uncached, cache_hint::uncached>(
                    c_tile_init, c_tile_payload_init);
            h_tile.reg = xetla_cvt<dtype_acc, dtype_b, SIMD>(h_tile_init.reg);
            c_tile.reg = xetla_cvt<dtype_acc, dtype_b, SIMD>(c_tile_init.reg);
            tile_store(h_tile, h_tile_payload);
            tile_store(c_tile, c_tile_payload);
        }
        xetla_fence<memory_kind::untyped_global>();
        nbarrier.arrive_wait();

        tile_c_t out_tile;
        global_c_payload_t out_tile_payload;
        out_tile_payload.init(hidden_history, hidden_size, seq_len,
                hidden_size * directions,
                hidden_size * dir_id + minigroup_offset, seq_start_y);

        tile_acc_t figo_tile;
        local_acc_payload_t figo_tile_payload;
        figo_tile_payload.init(
                slm_figo_offset, figo_size, 1, figo_size, local_id * SIMD, 0);

        tile_acc_t f_tile;
        local_acc_payload_t f_tile_payload;
        f_tile_payload.init(slm_figo_offset, hidden_size, 1, hidden_size,
                minigroup_offset, 0);

        tile_acc_t i_tile;
        local_acc_payload_t i_tile_payload;
        i_tile_payload.init(slm_figo_offset + hidden_size * sizeof(dtype_acc),
                hidden_size, 1, hidden_size, minigroup_offset, 0);

        tile_acc_t g_tile;
        local_acc_payload_t g_tile_payload;
        g_tile_payload.init(
                slm_figo_offset + hidden_size * 2 * sizeof(dtype_acc),
                hidden_size, 1, hidden_size, minigroup_offset, 0);

        tile_acc_t o_tile;
        local_acc_payload_t o_tile_payload;
        o_tile_payload.init(
                slm_figo_offset + hidden_size * 3 * sizeof(dtype_acc),
                hidden_size, 1, hidden_size, minigroup_offset, 0);

#ifdef UNROLL_SEQ_LEN
        for (int iii = 0; iii < seq_len / UNROLL_SEQ_LEN; iii++) {
#pragma unroll
            for (int ii = 0; ii < UNROLL_SEQ_LEN; ii++) {
                int i = ii * UNROLL_SEQ_LEN + iii;
#else
        for (int i = 0; i < seq_len; i++) {
#endif
                tile_load<cache_hint::uncached, cache_hint::cached>(
                        x_tile, x_tile_payload);
                x_tile_payload.template update_tdesc<tdesc_update_dir::y_dir>(
                        seq_inc);

                typename gemm_t::matAcc_t matAcc;
                gemm.run(matAcc);

                figo_tile.reg = matAcc.reg + x_tile.reg;
                if (local_id < (hidden_size * 2 / SIMD)
                        || local_id >= (hidden_size * 3 / SIMD)) {
                    figo_tile.reg
                            = xetla_sigmoid<dtype_acc, SIMD>(figo_tile.reg);
                } else {
                    figo_tile.reg = xetla_tanh<dtype_acc, SIMD>(figo_tile.reg);
                }
                tile_store(figo_tile, figo_tile_payload);
                xetla_fence<memory_kind::shared_local>();
                nbarrier.arrive_wait();

                if (local_id < (hidden_size / SIMD)) {
                    tile_load(f_tile, f_tile_payload);
                    tile_load(i_tile, i_tile_payload);
                    tile_load(g_tile, g_tile_payload);
                    tile_load(o_tile, o_tile_payload);

                    tile_load(c_tile, c_tile_payload);

                    c_tile.reg *= f_tile.reg;
                    i_tile.reg *= g_tile.reg;
                    c_tile.reg += i_tile.reg;

                    tile_store(c_tile, c_tile_payload);

                    h_tile.reg = xetla_tanh<dtype_acc, SIMD>(c_tile.reg)
                            * o_tile.reg;

                    tile_store(h_tile, h_tile_payload);

                    out_tile.reg
                            = xetla_cvt<dtype_c, dtype_acc, SIMD>(h_tile.reg);
                    tile_store<cache_hint::write_back, cache_hint::write_back>(
                            out_tile, out_tile_payload);
                    out_tile_payload
                            .template update_tdesc<tdesc_update_dir::y_dir>(
                                    seq_inc);
                }
                xetla_fence<memory_kind::shared_local>();
                nbarrier.arrive_wait();
            }
#ifdef UNROLL_SEQ_LEN
        }
#endif
        if (local_id < (hidden_size / SIMD)) {
            tile_load(h_tile, h_tile_payload);
            tile_load(c_tile, c_tile_payload);

            tile_c_t h_tile_out;
            global_c_payload_t h_tile_payload_out;
            h_tile_payload_out.init(context_h_out, hidden_size, 1, hidden_size,
                    minigroup_offset, 0);

            tile_c_t c_tile_out;
            global_c_payload_t c_tile_payload_out;
            c_tile_payload_out.init(context_c_out, hidden_size, 1, hidden_size,
                    minigroup_offset, 0);

            h_tile_out.reg = xetla_cvt<dtype_c, dtype_acc, SIMD>(h_tile.reg);
            c_tile_out.reg = xetla_cvt<dtype_c, dtype_acc, SIMD>(c_tile.reg);

            tile_store<cache_hint::write_back, cache_hint::write_back>(
                    h_tile_out, h_tile_payload_out);
            tile_store<cache_hint::write_back, cache_hint::write_back>(
                    c_tile_out, c_tile_payload_out);
        }
    }
};
