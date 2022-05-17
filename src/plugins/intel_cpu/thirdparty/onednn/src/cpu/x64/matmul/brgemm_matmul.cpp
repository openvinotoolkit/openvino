/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/tag_traits.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_primitive.hpp"

#include "cpu/x64/amx_tile_configure.hpp"
#include "cpu/x64/injectors/jit_uni_binary_injector.hpp"
#include "cpu/x64/matmul/brgemm_matmul.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace matmul {

using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;

using namespace nstl;

using namespace data_type;

template <cpu_isa_t isa>
status_t brgemm_matmul_t<isa>::pd_t::init(engine_t *engine) {
    const auto src_dt = src_md_.data_type;
    const auto wei_dt = weights_md_.data_type;
    const auto dst_dt = dst_md_.data_type;

    const bool is_f32 = everyone_is(f32, src_dt, wei_dt, dst_dt);
    const bool is_int8 = one_of(src_dt, u8, s8) && wei_dt == s8
            && one_of(dst_dt, u8, s8, s32, f32, bf16);
    const bool is_bf16
            = everyone_is(bf16, src_dt, wei_dt) && one_of(dst_dt, bf16, f32);

    auto check_bias = [&]() -> bool {
        const bool is_bia_dt_correct
                = (is_int8
                          && one_of(weights_md(1)->data_type, f32, s32, s8, u8,
                                  bf16))
                || (is_bf16 && one_of(weights_md(1)->data_type, f32, bf16))
                || (is_f32 && weights_md(1)->data_type == f32);
        return IMPLICATION(with_bias(), is_bia_dt_correct && is_bias_1xN());
    };

    auto check_attr_oscale = [&]() -> bool {
        const auto &oscale = attr()->output_scales_;
        return IMPLICATION(
                oscale.mask_ != 0, oscale.mask_ == (1 << (dst_md_.ndims - 1)));
    };

    auto check_attr_zero_points
            = [&]() -> bool { return attr()->zero_points_.common(); };

    const bool problem_dt_correct = is_int8 || is_bf16 || is_f32;
    bool ok = mayiuse(isa) && problem_dt_correct
            && !has_runtime_dims_or_strides()
            && attr()->has_default_values(primitive_attr_t::skip_mask_t::oscale
                            | primitive_attr_t::skip_mask_t::zero_points_runtime
                            | primitive_attr_t::skip_mask_t::post_ops
                            | primitive_attr_t::skip_mask_t::sum_dt,
                    dst_dt)
            && attr()->post_ops_.check_sum_consistent_dt(dst_dt)
            && check_attr_oscale() && check_attr_zero_points() && check_bias();
    if (!ok) return status::unimplemented;

    CHECK(init_brgemm_matmul_conf(isa, bgmmc_, *desc(), src_md_, weights_md_,
            dst_md_, bias_md_, attr_));

    const float alpha = 1.0;
    const float beta = 1.0;
    const float beta_init = 0.0;
    for_(int i_init = 0; i_init < 2; i_init++)
    for_(int i_M = 0; i_M < 2; i_M++)
    for_(int i_N = 0; i_N < 2; i_N++)
    for (int i_K = 0; i_K < 2; i_K++) {
        auto vbeta = (i_init) ? beta_init : beta;
        auto vM = (i_M) ? bgmmc_.M_tail : bgmmc_.M_blk;
        auto vN = (i_N) ? bgmmc_.N_tail : bgmmc_.N_blk;
        auto vK = (i_K) ? bgmmc_.K_tail : bgmmc_.K_blk;

        int idx = get_brg_kernel_idx(i_init, i_M, i_N, i_K);
        if (idx < 0) continue;
        brgemm_t &brg = brg_descs_[idx];
        auto LDA = i_K && bgmmc_.use_buffer_a_tail_only
                ? (dim_t)bgmmc_.wei_k_blk
                : bgmmc_.LDA;
        CHECK(brgemm_desc_init(&brg, isa, bgmmc_.brg_type, bgmmc_.src_dt,
                bgmmc_.wei_dt, false, false, brgemm_row_major, alpha, vbeta,
                LDA, bgmmc_.LDB, bgmmc_.LDC, vM, vN, vK));

        auto LDD = bgmmc_.N;
        CHECK(brgemm_desc_set_postops(
                &brg, attr(), &dst_md_, LDD, bgmmc_.bia_dt));

        brgemm_attr_t brgattr;
        constexpr bool is_amx = one_of(
                isa, avx512_core_bf16_amx_int8, avx512_core_bf16_amx_bf16);
        if (is_amx) {
            brgattr.max_bs = bgmmc_.brgemm_batch_size;
            brgattr.wary_tail_read = false;

            // TODO: change expected sizes to local chunks wrt L2 blocking
            brgattr.hint_expected_A_size = vM * vK * bgmmc_.brgemm_batch_size;
            brgattr.hint_expected_B_size = vN * vK * bgmmc_.brgemm_batch_size;
            brgattr.hint_expected_C_size = vM * vN * bgmmc_.brgemm_batch_size;
            brgattr.hint_innermost_loop = brgemm_ld_loop_innermost;
        }

        brgattr.generate_skip_accumulation
                = bgmmc_.post_ops_applicable && bgmmc_.nthr_k > 1;

        CHECK(brgemm_desc_set_attr(&brg, brgattr));
    }

    auto scratchpad = scratchpad_registry().registrar();
    init_scratchpad(scratchpad, bgmmc_);

    return status::success;
}

template <cpu_isa_t isa>
status_t brgemm_matmul_t<isa>::init(engine_t *engine) {
    for_(int i_M = 0; i_M < 2; i_M++)
    for_(int i_N = 0; i_N < 2; i_N++)
    for_(int i_K = 0; i_K < 2; i_K++)
    for (int i_init = 0; i_init < 2; i_init++) {
        int idx = pd()->get_brg_kernel_idx(i_init, i_M, i_N, i_K);
        if (idx < 0) continue;

        brgemm_kernel_t *ker = nullptr;
        CHECK(brgemm_kernel_create(&ker, pd()->get_brg_desc(idx)));
        CHECK(safe_ptr_assign(brg_kernels_[idx], ker));
        if (one_of(isa, avx512_core_bf16_amx_int8, avx512_core_bf16_amx_bf16))
            CHECK(brgemm_init_tiles(
                    pd()->get_brg_desc(idx), &brg_kernel_palettes_[idx][0]));
    }

    const auto &bgmmc = pd()->get_brgemm_matmul_conf();
    if (bgmmc.use_buffer_b)
        CHECK(create_brgemm_matmul_copy_b(copy_B_kernel_, &bgmmc));

    if (bgmmc.use_buffer_a || bgmmc.use_buffer_a_tail_only)
        CHECK(create_brgemm_matmul_copy_a(copy_A_kernel_, &bgmmc));

    if (bgmmc.nthr_k > 1 && bgmmc.acc_dt == f32) {
        CHECK(safe_ptr_assign(
                acc_ker_f32_, new cpu_accumulator_1d_t<data_type::f32>()));
        CHECK(acc_ker_f32_->create_kernel());
    } else if (bgmmc.nthr_k > 1 && bgmmc.acc_dt == s32) {
        CHECK(safe_ptr_assign(
                acc_ker_s32_, new cpu_accumulator_1d_t<data_type::s32>()));
        CHECK(acc_ker_s32_->create_kernel());
    }

    return status::success;
}

template <cpu_isa_t isa>
status_t brgemm_matmul_t<isa>::execute_body(const exec_ctx_t &ctx) const {
    DEFINE_ZERO_POINT_VALUE(src_zero_point, DNNL_ARG_SRC);
    DEFINE_ZERO_POINT_VALUE(wei_zero_point, DNNL_ARG_WEIGHTS);
    DEFINE_ZERO_POINT_VALUE(dst_zero_point, DNNL_ARG_DST);

    brg_matmul_exec_ctx_t brgmm_ctx(
            ctx, pd(), src_zero_point, wei_zero_point, dst_zero_point);

    const auto &bgmmc = pd()->get_brgemm_matmul_conf();
    const bool use_buffer_a
            = bgmmc.use_buffer_a || bgmmc.use_buffer_a_tail_only;
    constexpr bool is_amx
            = one_of(isa, avx512_core_bf16_amx_int8, avx512_core_bf16_amx_bf16);
    const int num_threads = brgmm_ctx.get_num_threads_for_parallelization();

    parallel(num_threads, [&](const int ithr, const int nthr) {
        const int ithr_bmn = brgmm_ctx.get_thread_idx_for_bmn(ithr);
        const int ithr_k = brgmm_ctx.get_thread_idx_for_k(ithr);
        if (ithr_bmn < 0 || ithr_k < 0) return;
        int start {0}, end {0};
        balance211(brgmm_ctx.get_parallel_work_amount(),
                brgmm_ctx.get_num_threads_for_bmn(), ithr_bmn, start, end);
        int kc_start {0}, kc_end {bgmmc.K_chunks};
        if (brgmm_ctx.parallel_reduction_is_used())
            balance211((int)bgmmc.K_chunks, brgmm_ctx.get_num_threads_for_k(),
                    ithr_k, kc_start, kc_end);

        if (is_amx) {
            const auto base_ker_idx = brgmm_ctx.get_base_brgemm_kernel_idx();
            amx_tile_configure(&brg_kernel_palettes_[base_ker_idx][0]);
        }

        int b {0}, mc {0}, nc {0};
        nd_iterator_init(
                start, b, bgmmc.batch, mc, bgmmc.M_chunks, nc, bgmmc.N_chunks);
        while (start < end) {
            auto m_start = mc * bgmmc.M_chunk_size;
            auto m_end = nstl::min(
                    (mc + 1) * bgmmc.M_chunk_size, bgmmc.num_M_blocks);
            auto n_start = nc * bgmmc.N_chunk_size;
            auto n_end = nstl::min(
                    (nc + 1) * bgmmc.N_chunk_size, bgmmc.num_N_blocks);
            for_(int kc = kc_start; kc < kc_end; kc++)
            for (int nb = n_start; nb < n_end; nb++) {
                if (bgmmc.use_buffer_b)
                    copy_b_chunk_in_buffer(brgmm_ctx, ithr, b, nb, kc);
                for (int mb = m_start; mb < m_end; mb++) {
                    if (use_buffer_a && nb == n_start)
                        copy_a_chunk_in_buffer(brgmm_ctx, ithr, b, mb, kc);
                    compute_kernel(
                            brgmm_ctx, ithr, b, mb, nb, kc, kc == kc_start);
                }
            }
            ++start;
            nd_iterator_step(
                    b, bgmmc.batch, mc, bgmmc.M_chunks, nc, bgmmc.N_chunks);
        }
        if (is_amx) { amx_tile_release(); }
    });

    maybe_reduce_partial_results_and_apply_postops(brgmm_ctx);

    return status::success;
}

template <cpu_isa_t isa>
void brgemm_matmul_t<isa>::compute_kernel(
        const brg_matmul_exec_ctx_t &brgmm_ctx, int ithr, int b_idx,
        int m_blk_idx, int n_blk_idx, int k_chunk_idx, bool do_init) const {
    constexpr bool is_amx
            = one_of(isa, avx512_core_bf16_amx_int8, avx512_core_bf16_amx_bf16);
    const auto &bgmmc = pd()->get_brgemm_matmul_conf();
    const auto addr_batch = brgmm_ctx.get_batch_elem_ptr(ithr);
    const int base_brg_ker_idx = brgmm_ctx.get_base_brgemm_kernel_idx();

    const auto wsp_tile = brgmm_ctx.get_tile_workspace(ithr);
    const int m = m_blk_idx * bgmmc.M_blk;
    const int n = n_blk_idx * bgmmc.N_blk;
    const int k_blk_idx = k_chunk_idx * bgmmc.brgemm_batch_size;

    const bool is_M_tail = (bgmmc.M - m < bgmmc.M_blk);
    const bool is_N_tail = (bgmmc.N - n < bgmmc.N_blk);
    const bool is_last_K_chunk = brgmm_ctx.is_last_K_chunk(k_chunk_idx);
    const bool is_K_tail = is_last_K_chunk && bgmmc.K_tail > 0;

    const int gemm_batch = brgmm_ctx.get_brgemm_batch_size(k_chunk_idx);
    const int brg_ker_idx
            = pd()->get_brg_kernel_idx(do_init, is_M_tail, is_N_tail, false);
    const auto brg_kernel = brg_kernels_[brg_ker_idx].get();
    const auto ptr_bias = brgmm_ctx.get_bias_ptr(n);
    auto ptr_D = brgmm_ctx.get_data_C_ptr(b_idx, m, n);
    auto ptr_C = (bgmmc.use_buffer_c)
            ? brgmm_ctx.get_buf_C_ptr(ithr, m_blk_idx, n_blk_idx)
            : ptr_D;

    const auto zp_comp_a = brgmm_ctx.get_zp_a_compensation_ptr(ithr, n_blk_idx);
    const auto zp_comp_b
            = brgmm_ctx.get_zp_b_compensation_result_ptr(ithr, m_blk_idx);
    const auto zp_c_val_ptr = brgmm_ctx.get_zp_c_val_ptr();
    const auto &post_ops_binary_rhs_arg_vec
            = brgmm_ctx.get_post_ops_binary_rhs_arg_vec();
    const bool post_ops_applicable = bgmmc.post_ops_applicable
            && (bgmmc.nthr_k <= 1 || bgmmc.K_chunks == 1);

    if (gemm_batch > 0 && brg_kernel != nullptr) {
        const bool is_tile_reconf_required = is_amx && (is_M_tail || is_N_tail);
        if (is_tile_reconf_required)
            amx_tile_configure(&brg_kernel_palettes_[brg_ker_idx][0]);

        brgmm_ctx.init_brgemm_batch_elements_values(
                ithr, 0, gemm_batch, b_idx, m_blk_idx, k_blk_idx, n_blk_idx);

        if (post_ops_applicable && is_last_K_chunk && !is_K_tail) {
            void *scratch = is_amx
                    ? static_cast<void *>(wsp_tile)
                    : static_cast<void *>(brgmm_ctx.get_s8s8_comp_ptr(
                            ithr, b_idx, n_blk_idx));

            const size_t dst_row_logical_off = m_blk_idx * bgmmc.M_blk;
            const size_t batch_first_dim_idx = bgmmc.batch_ndims > 1
                    ? b_idx / bgmmc.batch_without_first_dim
                    : 0;
            const size_t first_mb_matrix_addr_off
                    = batch_first_dim_idx * (bgmmc.M * bgmmc.N)
                    + (m * bgmmc.N + n);
            const brgemm_post_ops_data_t post_ops_data {
                    static_cast<const void *>(ptr_bias),
                    brgmm_ctx.get_oscales_ptr(n),
                    post_ops_binary_rhs_arg_vec.data(), static_cast<size_t>(n),
                    dst_row_logical_off, brgmm_ctx.get_data_C_ptr(0, 0, 0),
                    first_mb_matrix_addr_off,
                    static_cast<const void *>(zp_comp_a),
                    static_cast<const void *>(zp_comp_b),
                    static_cast<const void *>(zp_c_val_ptr)};

            brgemm_kernel_execute_postops(brg_kernel, gemm_batch, addr_batch,
                    (void *)ptr_C, (void *)ptr_D, post_ops_data, scratch);
        } else {
            brgemm_kernel_execute(brg_kernel, gemm_batch, addr_batch,
                    (void *)ptr_C, is_amx ? (void *)wsp_tile : nullptr);
        }

        if (is_tile_reconf_required)
            amx_tile_configure(&brg_kernel_palettes_[base_brg_ker_idx][0]);
    }
    if (is_K_tail) {
        brgmm_ctx.init_brgemm_batch_elements_values(
                ithr, gemm_batch, 1, b_idx, m_blk_idx, k_blk_idx, n_blk_idx);

        const bool use_init_ker = (do_init && gemm_batch == 0);
        const int brg_ker_idx = pd()->get_brg_kernel_idx(
                use_init_ker, is_M_tail, is_N_tail, true);
        const auto brg_kernel_k_tail = brg_kernels_[brg_ker_idx].get();
        const bool is_tile_reconf_required
                = is_amx && bgmmc.K_tail != bgmmc.K_blk;
        if (is_tile_reconf_required)
            amx_tile_configure(&brg_kernel_palettes_[brg_ker_idx][0]);
        if (post_ops_applicable) {
            void *scratch = is_amx
                    ? static_cast<void *>(wsp_tile)
                    : static_cast<void *>(brgmm_ctx.get_s8s8_comp_ptr(
                            ithr, b_idx, n_blk_idx));

            const size_t dst_row_logical_off = m_blk_idx * bgmmc.M_blk;
            const size_t batch_first_dim_idx = bgmmc.batch_ndims > 1
                    ? b_idx / bgmmc.batch_without_first_dim
                    : 0;
            const size_t first_mb_matrix_addr_off
                    = batch_first_dim_idx * (bgmmc.M * bgmmc.N)
                    + (m * bgmmc.N + n);
            const brgemm_post_ops_data_t post_ops_data {
                    static_cast<const void *>(ptr_bias),
                    brgmm_ctx.get_oscales_ptr(n),
                    post_ops_binary_rhs_arg_vec.data(), static_cast<size_t>(n),
                    dst_row_logical_off, brgmm_ctx.get_data_C_ptr(0, 0, 0),
                    first_mb_matrix_addr_off,
                    static_cast<const void *>(zp_comp_a),
                    static_cast<const void *>(zp_comp_b),
                    static_cast<const void *>(zp_c_val_ptr)};

            brgemm_kernel_execute_postops(brg_kernel_k_tail, 1, addr_batch,
                    (void *)ptr_C, (void *)ptr_D, post_ops_data, scratch);
        } else {
            brgemm_kernel_execute(brg_kernel_k_tail, 1, addr_batch,
                    (void *)ptr_C, is_amx ? (void *)wsp_tile : nullptr);
        }
        if (is_tile_reconf_required)
            amx_tile_configure(&brg_kernel_palettes_[base_brg_ker_idx][0]);
    }
}

template <cpu_isa_t isa>
void brgemm_matmul_t<isa>::maybe_reduce_partial_results_and_apply_postops(
        const brg_matmul_exec_ctx_t &brgmm_ctx) const {
    if (!brgmm_ctx.parallel_reduction_is_used()) return;

    const auto &bgmmc = pd()->get_brgemm_matmul_conf();
    const int num_threads = brgmm_ctx.get_num_threads_for_parallelization();

    parallel(num_threads, [&](const int ithr, const int nthr) {
        const int nthr_k = brgmm_ctx.get_num_threads_for_k();
        const int ithr_bmn = brgmm_ctx.get_thread_idx_for_bmn(ithr);
        const int ithr_k = brgmm_ctx.get_thread_idx_for_k(ithr);
        if (ithr_bmn < 0 || ithr_k < 0) return;

        const int num_reduction_buffers = nstl::min(nthr_k, bgmmc.K_chunks);

        int bmn_start {0}, bmn_end {0};
        int start {0}, end {0};
        balance211(brgmm_ctx.get_parallel_work_amount(),
                brgmm_ctx.get_num_threads_for_bmn(), ithr_bmn, bmn_start,
                bmn_end);
        balance211(bmn_end - bmn_start, nthr_k, ithr_k, start, end);

        int b {0}, mc {0}, nc {0};

        assert(bgmmc.batch == 1);
        nd_iterator_init(bmn_start + start, b, bgmmc.batch, mc, bgmmc.M_chunks,
                nc, bgmmc.N_chunks);
        while (start < end) {
            auto mb_start = mc * bgmmc.M_chunk_size;
            auto mb_end = nstl::min(
                    (mc + 1) * bgmmc.M_chunk_size, bgmmc.num_M_blocks);
            auto nb_start = nc * bgmmc.N_chunk_size;
            auto nb_end = nstl::min(
                    (nc + 1) * bgmmc.N_chunk_size, bgmmc.num_N_blocks);
            for (int mb = mb_start; mb < mb_end; mb++) {
                const int curr_M_blk
                        = nstl::min(bgmmc.M - mb * bgmmc.M_blk, bgmmc.M_blk);
                const bool is_M_tail = curr_M_blk < bgmmc.M_blk;
                const int curr_N_chunk_size
                        = nstl::min(bgmmc.N, nb_end * bgmmc.N_blk)
                        - nb_start * bgmmc.N_blk;
                char *buf_reduced_base = brgmm_ctx.get_buf_C_par_reduction_ptr(
                        0, mb, nb_start);
                const size_t m_offset = bgmmc.LDC * bgmmc.acc_dt_sz;
                for (int r = 1; r < num_reduction_buffers; r++) {
                    const char *buf_to_reduce_base
                            = brgmm_ctx.get_buf_C_par_reduction_ptr(
                                    r, mb, nb_start);
                    for (int m = 0; m < curr_M_blk; m++) {
                        accumulate(buf_reduced_base + m * m_offset,
                                buf_to_reduce_base + m * m_offset,
                                curr_N_chunk_size);
                    }
                }
                if (bgmmc.post_ops_applicable) {
                    for (int nb = nb_start; nb < nb_end; nb++) {
                        const bool is_N_tail
                                = (bgmmc.N - nb * bgmmc.N_blk < bgmmc.N_blk);
                        const int brg_ker_idx = pd()->get_brg_kernel_idx(
                                false, is_M_tail, is_N_tail, false);
                        const auto brg_kernel = brg_kernels_[brg_ker_idx].get();
                        const int m = mb * bgmmc.M_blk;
                        const int n = nb * bgmmc.N_blk;
                        const auto ptr_bias = brgmm_ctx.get_bias_ptr(n);
                        auto ptr_D = brgmm_ctx.get_data_C_ptr(b, m, n);
                        auto ptr_C = brgmm_ctx.get_buf_C_par_reduction_ptr(
                                0, mb, nb);

                        // TODO: support reduction for zp/s8s8 compensations
                        // computed in copy routines
                        const auto zp_comp_a
                                = brgmm_ctx.get_zp_a_compensation_ptr(ithr, nb);
                        const auto zp_comp_b
                                = brgmm_ctx.get_zp_b_compensation_result_ptr(
                                        ithr, mb);
                        const auto zp_c_val_ptr = brgmm_ctx.get_zp_c_val_ptr();
                        const auto &post_ops_binary_rhs_arg_vec
                                = brgmm_ctx.get_post_ops_binary_rhs_arg_vec();

                        const size_t dst_row_logical_off = mb * bgmmc.M_blk;
                        const size_t batch_first_dim_idx = bgmmc.batch_ndims > 1
                                ? b / bgmmc.batch_without_first_dim
                                : 0;
                        const size_t first_mb_matrix_addr_off
                                = batch_first_dim_idx * (bgmmc.M * bgmmc.N)
                                + (m * bgmmc.N + n);
                        // apply post-ops and convert to dst data type only
                        constexpr bool skip_accumulation = true;
                        const brgemm_post_ops_data_t post_ops_data {
                                static_cast<const void *>(ptr_bias),
                                brgmm_ctx.get_oscales_ptr(n),
                                post_ops_binary_rhs_arg_vec.data(),
                                static_cast<size_t>(n), dst_row_logical_off,
                                brgmm_ctx.get_data_C_ptr(0, 0, 0),
                                first_mb_matrix_addr_off,
                                static_cast<const void *>(zp_comp_a),
                                static_cast<const void *>(zp_comp_b),
                                static_cast<const void *>(zp_c_val_ptr),
                                skip_accumulation};

                        brgemm_kernel_execute_postops(brg_kernel, 0, nullptr,
                                (void *)ptr_C, (void *)ptr_D, post_ops_data,
                                nullptr);
                    }
                }
            }
            ++start;
            nd_iterator_step(
                    b, bgmmc.batch, mc, bgmmc.M_chunks, nc, bgmmc.N_chunks);
        }
    });
}

template <cpu_isa_t isa>
void brgemm_matmul_t<isa>::copy_a_chunk_in_buffer(
        const brg_matmul_exec_ctx_t &brgmm_ctx, int ithr, int b_idx,
        int m_blk_idx, int k_chunk_idx) const {
    const auto &bgmmc = pd()->get_brgemm_matmul_conf();

    auto ctx = jit_brgemm_matmul_copy_a_t::ctx_t();
    const int k_start = k_chunk_idx * bgmmc.K_chunk_elems;
    const bool is_K_tail
            = brgmm_ctx.is_last_K_chunk(k_chunk_idx) && bgmmc.K_tail > 0;
    const int gemm_batch = brgmm_ctx.get_brgemm_batch_size(k_chunk_idx);
    const int gemm_batch_iters = bgmmc.use_buffer_a_tail_only ? 0 : gemm_batch;

    const int m = m_blk_idx * bgmmc.M_blk;
    const bool is_M_tail = (bgmmc.M - m < bgmmc.M_blk);
    ctx.current_M_blk = is_M_tail ? bgmmc.M_tail : bgmmc.M_blk;
    ctx.zp_b_compensation_buffer_ptr
            = (void *)brgmm_ctx.get_zp_b_compensation_buffer_ptr(
                    ithr, m_blk_idx);
    ctx.zp_a_compensation_result_ptr
            = (void *)brgmm_ctx.get_zp_b_compensation_result_ptr(
                    ithr, m_blk_idx);
    ctx.zp_b_neg_value_ptr = (void *)brgmm_ctx.get_zp_b_neg_val_ptr();
    ctx.zp_ab_comp_ptr = (void *)brgmm_ctx.get_zp_ab_mixed_comp_ptr();

    for (int gb = 0; gb < gemm_batch_iters; gb++) {
        const int k = k_start + gb * bgmmc.K_blk;
        ctx.src = (void *)brgmm_ctx.get_data_A_ptr(b_idx, m, k);
        ctx.tr_src = (void *)brgmm_ctx.get_buf_A_ptr(ithr, m_blk_idx, gb);
        ctx.current_K_blk = nstl::min(bgmmc.K_blk, bgmmc.K);
        ctx.current_K_start = k;

        (*copy_A_kernel_)(&ctx);
    }
    if (is_K_tail) {
        const auto K_tail = bgmmc.K % bgmmc.K_blk;
        const int k = k_start + gemm_batch * bgmmc.K_blk;
        ctx.src = (void *)brgmm_ctx.get_data_A_ptr(b_idx, m, k);
        ctx.tr_src = (void *)brgmm_ctx.get_buf_A_ptr(
                ithr, m_blk_idx, gemm_batch_iters);
        ctx.current_K_blk = K_tail;
        ctx.current_K_start = k;

        (*copy_A_kernel_)(&ctx);
    }
}

template <cpu_isa_t isa>
void brgemm_matmul_t<isa>::copy_b_chunk_in_buffer(
        const brg_matmul_exec_ctx_t &brgmm_ctx, int ithr, int b_idx,
        int n_blk_idx, int k_chunk_idx) const {
    const auto &bgmmc = pd()->get_brgemm_matmul_conf();

    const int k_start = k_chunk_idx * bgmmc.K_chunk_elems;
    const bool is_K_tail
            = brgmm_ctx.is_last_K_chunk(k_chunk_idx) && bgmmc.K_tail > 0;
    const int gemm_batch = brgmm_ctx.get_brgemm_batch_size(k_chunk_idx);
    auto ctx = jit_brgemm_matmul_copy_b_t::ctx_t();

    const int n = n_blk_idx * bgmmc.N_blk;
    const bool is_N_tail = (bgmmc.N - n < bgmmc.N_blk);
    ctx.current_N_blk = is_N_tail ? bgmmc.N_tail : bgmmc.N_blk;
    ctx.zp_a_compensation_ptr
            = (void *)brgmm_ctx.get_zp_a_compensation_ptr(ithr, n_blk_idx);
    ctx.zp_a_neg_value_ptr = (void *)brgmm_ctx.get_zp_a_neg_val_ptr();

    int gb = 0;
    for (; gb < gemm_batch; gb++) {
        const int k = k_start + gb * bgmmc.K_blk;
        ctx.src = (void *)brgmm_ctx.get_data_B_ptr(b_idx, k, n);
        ctx.tr_src = (void *)brgmm_ctx.get_buf_B_ptr(ithr, gb, n_blk_idx);
        ctx.compensation_ptr
                = (void *)brgmm_ctx.get_s8s8_comp_ptr(ithr, b_idx, n_blk_idx);
        ctx.current_K_start = k;
        ctx.current_K_iters = nstl::min(bgmmc.K_blk, bgmmc.K);

        (*copy_B_kernel_)(&ctx);
    }

    if (is_K_tail) {
        const int k = k_start + gb * bgmmc.K_blk;
        ctx.src = (void *)brgmm_ctx.get_data_B_ptr(b_idx, k, n);
        ctx.tr_src = (void *)brgmm_ctx.get_buf_B_ptr(ithr, gb, n_blk_idx);
        ctx.compensation_ptr
                = (void *)brgmm_ctx.get_s8s8_comp_ptr(ithr, b_idx, n_blk_idx);
        ctx.current_K_start = k;
        ctx.current_K_iters = bgmmc.K % bgmmc.K_blk;

        (*copy_B_kernel_)(&ctx);
    }
}

template <cpu_isa_t isa>
void brgemm_matmul_t<isa>::accumulate(
        char *result_ptr, const char *reduce_ptr, size_t size) const {
    if (pd()->get_brgemm_matmul_conf().acc_dt == f32)
        acc_ker_f32_->accumulate(
                (float *)result_ptr, (const float *)reduce_ptr, size);
    else if (pd()->get_brgemm_matmul_conf().acc_dt == s32)
        acc_ker_s32_->accumulate(
                (int32_t *)result_ptr, (const int32_t *)reduce_ptr, size);
    else
        assert(!"unsupported accumulation data type");
}

template <cpu_isa_t isa>
struct brgemm_matmul_t<isa>::brg_matmul_exec_ctx_t {
    brg_matmul_exec_ctx_t(const exec_ctx_t &ctx, const pd_t *pd, int32_t src_zp,
            int32_t wei_zp, int32_t dst_zp)
        : bgmmc_(pd->get_brgemm_matmul_conf()) {

        data_A_ptr_ = CTX_IN_MEM(const char *, DNNL_ARG_SRC);
        data_B_ptr_ = CTX_IN_MEM(const char *, DNNL_ARG_WEIGHTS);
        data_C_ptr_ = CTX_OUT_MEM(char *, DNNL_ARG_DST);

        bias_ptr_ = CTX_IN_MEM(const char *, DNNL_ARG_BIAS);
        oscales_ptr_ = pd->attr()->output_scales_.scales_;
        memory_tracking::grantor_t scratchpad = ctx.get_scratchpad_grantor();
        const auto &bgmmc = pd->get_brgemm_matmul_conf();

        batch_element_ptr_ = scratchpad.template get<brgemm_batch_element_t>(
                key_brgemm_primitive_batch);

        const bool use_buffer_a
                = bgmmc.use_buffer_a || bgmmc.use_buffer_a_tail_only;
        buf_A_ptr_ = (use_buffer_a)
                ? scratchpad.template get<char>(key_brgemm_primitive_buffer_a)
                : nullptr;

        buf_B_ptr_ = (bgmmc.use_buffer_b)
                ? scratchpad.template get<char>(key_brgemm_primitive_buffer_b)
                : nullptr;

        buf_C_ptr_ = (bgmmc.use_buffer_c)
                ? scratchpad.template get<char>(key_brgemm_primitive_buffer)
                : nullptr;

        is_amx_ = one_of(
                isa, avx512_core_bf16_amx_int8, avx512_core_bf16_amx_bf16);
        wsp_tile_ptr_ = is_amx_
                ? ctx.get_scratchpad_grantor().template get<char>(
                        key_conv_amx_tile_buffer)
                : nullptr;

        const memory_desc_wrapper weights_d(pd->weights_md(0));
        const dim_t comp_offset = bgmmc_.b_dt_sz
                * (weights_d.size() - weights_d.additional_buffer_size());
        s8s8_compensation_ptr_ = (bgmmc.s8s8_compensation_required)
                ? ((bgmmc.use_buffer_b)
                                ? scratchpad.template get<int32_t>(
                                        key_brgemm_primitive_buffer_comp)
                                : const_cast<int32_t *>(
                                        reinterpret_cast<const int32_t *>(
                                                &data_B_ptr_[comp_offset])))
                : nullptr;

        zero_point_a_compensations_ptr_ = bgmmc.has_zero_point_a
                ? scratchpad.template get<int32_t>(
                        key_brgemm_primitive_zp_comp_a)
                : nullptr;
        zero_point_b_compensations_ptr_ = bgmmc.has_zero_point_b
                ? scratchpad.template get<int32_t>(
                        key_brgemm_primitive_zp_comp_b)
                : nullptr;

        zero_point_a_negative_val_ = -src_zp;
        zero_point_b_negative_val_ = -wei_zp;
        zero_point_mixed_ab_compensation_component_
                = bgmmc.K * zero_point_a_negative_val_;

        zero_point_c_val_ = dst_zp;

        post_ops_binary_rhs_arg_vec_ = binary_injector::prepare_binary_args(
                pd->attr()->post_ops_, ctx);
        base_brg_ker_idx_ = pd->get_brg_kernel_idx(true, false, false, false);
        vnni_factor = isa == avx512_core_bf16_amx_int8
                ? 4
                : isa == avx512_core_bf16_amx_bf16 ? 2 : 1;

        reorder_zp_a_comp_ptr_ = nullptr;
        if (bgmmc_.has_zero_point_a && bgmmc_.blocked_B) {
            // Store the pointer to computed in reorder compensation values to
            // scale them locally by zp_a value just before usage in post-ops.
            // Using the single global scaling before parallel section might
            // produce significant overhead for small problems running in
            // multitreaded execution mode
            const size_t reorder_zp_a_comp_offset
                    = weights_d.size() - weights_d.additional_buffer_size();
            const size_t s8s8_buffer_sz = bgmmc.s8s8_compensation_required
                    ? bgmmc.s8s8_comp_b_str * sizeof(int32_t)
                    : 0;
            reorder_zp_a_comp_ptr_
                    = const_cast<int32_t *>(reinterpret_cast<const int32_t *>(
                            &data_B_ptr_[reorder_zp_a_comp_offset
                                    + s8s8_buffer_sz]));
        }

        // parallelization
        parallel_work_amount_ = bgmmc.batch * bgmmc.M_chunks * bgmmc.N_chunks;

        // The number of threads available during primitive execution may
        // increase (ex. Eigen threadpool implementation) or decrease
        // (ex. nested parallelism) compared to the
        // number of threads available during primitive creation.
        // So we limit the total number of threads to the
        // minimum of these two values to prevent potential OOM issues.
        nthr_ = nstl::min(dnnl_get_current_num_threads(), bgmmc.nthr);

        nthr_k_ = bgmmc.nthr_k > 0 && bgmmc.nthr_k <= nthr_ ? bgmmc.nthr_k : 1;
        nthr_bmn_ = nthr_ / nthr_k_;
        num_threads_used_ = nthr_k_ * nthr_bmn_;

        // If parallel_work_amount_ == 1 and parallel reduction is not used, we
        // limit num threads to 1 as parallel(1, ...) does not create parallel
        // section at all. We do not limit number of threads for case
        // 1 < parallel_work_amount_ < dnnl_get_max_threads() to avoid potential
        // overhead on spawning different number of OMP threads from layer to
        // layer.
        if (parallel_work_amount_ == 1 && !parallel_reduction_is_used())
            nthr_ = nthr_bmn_ = nthr_k_ = 1;

        const bool need_to_calculate_compensation_for_a
                = bgmmc.has_zero_point_b;
        const bool need_to_calculate_compensation_for_b = !IMPLICATION(
                (bgmmc.has_zero_point_a || bgmmc.s8s8_compensation_required),
                bgmmc.blocked_B);
        const bool calculate_compensations_in_copy_routines
                = need_to_calculate_compensation_for_a
                || need_to_calculate_compensation_for_b;
        // currently parallel reduction is supported only for case of
        // non-batched problems without computation of any compensations in
        // copy routines
        assert(IMPLICATION(parallel_reduction_is_used(),
                bgmmc.batch == 1 && !calculate_compensations_in_copy_routines));
        MAYBE_UNUSED(need_to_calculate_compensation_for_a);
        MAYBE_UNUSED(need_to_calculate_compensation_for_b);
        MAYBE_UNUSED(calculate_compensations_in_copy_routines);
    }

    // NOTE: gb --> generalized batch, bb --> broadcast batch
    int get_bb_idx(int gb_idx, const brgemm_matmul_bcast_desc_t &bd) const {
        if (!bd.bcast_mask) // no broadcast
            return gb_idx;

        int gb_off_before_bcast = utils::rnd_dn(
                gb_idx, bd.first_bcast_dim_to_last_batch_dim_prod);
        int bb_idx = gb_off_before_bcast / (bd.bcast_dims_prod);

        dim_t cur_bcast_dims_prod = bd.bcast_dims_prod;
        int mask = 1 << (bgmmc_.batch_ndims - bd.first_bcast_dim - 1);
        for (int d = bd.first_bcast_dim; d < bd.last_bcast_dim; ++d) {
            if (bd.bcast_mask & mask) // broadcast
                cur_bcast_dims_prod /= bd.batch_dims[d];
            else {
                int cur_b = (gb_idx / bd.gb_off[d]) % bd.batch_dims[d];
                bb_idx += cur_b * (bd.gb_off[d] / cur_bcast_dims_prod);
            }
            mask >>= 1;
        }
        bb_idx += gb_idx % bd.gb_off[bd.last_bcast_dim];
        return bb_idx;
    }

    const char *get_data_A_ptr(int b, int m, int k) const {
        int cur_b = get_bb_idx(b, bgmmc_.bcast_A_desc);
        return data_A_ptr_ + get_data_A_off(cur_b, m, k);
    }

    const char *get_data_B_ptr(int b, int k, int n) const {
        int cur_b = get_bb_idx(b, bgmmc_.bcast_B_desc);
        return data_B_ptr_ + get_data_B_off(cur_b, k, n);
    }

    char *get_data_C_ptr(int b, int m, int n) const {
        return data_C_ptr_ + get_data_C_off(b, m, n);
    }

    brgemm_batch_element_t *get_batch_elem_ptr(int ithr) const {
        return batch_element_ptr_
                + ithr * bgmmc_.brgemm_batch_element_per_thr_sz;
    }

    void init_brgemm_batch_elements_values(int ithr, int brg_batch_start,
            int brg_batch_iters, int b_idx, int m_blk_idx, int k_blk_idx,
            int n_blk_idx) const {
        auto addr_batch = get_batch_elem_ptr(ithr);

        const int m = m_blk_idx * bgmmc_.M_blk;
        const int n = n_blk_idx * bgmmc_.N_blk;

        for (int b_iter = 0; b_iter < brg_batch_iters; b_iter++) {
            const int brg_batch_idx = brg_batch_start + b_iter;
            const int k = (k_blk_idx + brg_batch_idx) * bgmmc_.K_blk;
            addr_batch[b_iter].ptr.A = bgmmc_.use_buffer_a
                    ? get_buf_A_ptr(ithr, m_blk_idx, brg_batch_idx)
                    : get_data_A_ptr(b_idx, m, k);
            addr_batch[b_iter].ptr.B = (bgmmc_.use_buffer_b)
                    ? get_buf_B_ptr(ithr, brg_batch_idx, n_blk_idx)
                    : get_data_B_ptr(b_idx, k, n);
        }
    }

    char *get_buf_A_ptr(int ithr, int m_blk_idx, int k_blk_idx) const {
        if (!bgmmc_.use_buffer_a && !bgmmc_.use_buffer_a_tail_only)
            return nullptr;

        const int k_blk_local = bgmmc_.use_buffer_a_tail_only ? 0 : k_blk_idx;
        const int m_blk_local = m_blk_idx % bgmmc_.M_chunk_size;
        return buf_A_ptr_ + ithr * bgmmc_.buffer_a_per_thread_sz
                + m_blk_local * bgmmc_.buffer_a_chunk_shift_along_m
                + k_blk_local * bgmmc_.buffer_a_chunk_sz;
    }

    char *get_buf_B_ptr(int ithr, int k_blk_idx, int n_blk_idx) const {
        UNUSED(n_blk_idx);
        if (!bgmmc_.use_buffer_b) return nullptr;

        return buf_B_ptr_ + ithr * bgmmc_.buffer_b_per_thread_sz
                + k_blk_idx * bgmmc_.buffer_b_chunk_sz;
    }

    char *get_buf_C_ptr(int ithr, int m_blk_idx, int n_blk_idx) const {
        if (!bgmmc_.use_buffer_c) return nullptr;

        if (bgmmc_.nthr_k > 1) {
            const int nthr_k = bgmmc_.nthr_k <= nthr_ ? bgmmc_.nthr_k : 1;
            const int nthr_bmn = nthr_ / nthr_k;
            const int ithr_k = ithr / nthr_bmn;
            return get_buf_C_par_reduction_ptr(ithr_k, m_blk_idx, n_blk_idx);
        }

        const int m_blk_local = m_blk_idx % bgmmc_.M_chunk_size;
        const int n_blk_local = n_blk_idx % bgmmc_.N_chunk_size;
        const int buf_idx = bgmmc_.N_chunk_size * m_blk_local + n_blk_local;

        return buf_C_ptr_ + ithr * bgmmc_.buffer_c_per_thread_sz
                + buf_idx * bgmmc_.buffer_c_chunk_sz;
    }

    char *get_buf_C_par_reduction_ptr(
            int ithr_k, int m_blk_idx, int n_blk_idx) const {
        if (bgmmc_.nthr_k <= 1) return nullptr;

        const int m = m_blk_idx * bgmmc_.M_blk;
        const int n = n_blk_idx * bgmmc_.N_blk;

        if (!bgmmc_.post_ops_applicable && ithr_k == 0)
            return get_data_C_ptr(0, m, n);

        int k_buf_idx = ithr_k - (!bgmmc_.post_ops_applicable ? 1 : 0);
        return buf_C_ptr_ + k_buf_idx * bgmmc_.buffer_c_per_thread_sz
                + get_data_C_off(0, m, n) * bgmmc_.acc_dt_sz / bgmmc_.c_dt_sz;
    }

    // Auxiliary functions for getting offsets with pre-calculated memory
    // strides for each tensor to get general sulution for all possible
    // dimension without significant overhead
    dim_t get_data_A_off(int b, int m, int k) const {
        return bgmmc_.A_strides[2] * b + bgmmc_.A_strides[1] * m
                + bgmmc_.A_strides[0] * k;
    }
    dim_t get_data_B_off(int b, int k, int n) const {
        int k_idx = bgmmc_.blocked_B ? k / bgmmc_.wei_k_blk : k;
        int n_idx = bgmmc_.blocked_B ? n / bgmmc_.wei_n_blk : n;

        return bgmmc_.B_strides[2] * b + bgmmc_.B_strides[1] * k_idx
                + bgmmc_.B_strides[0] * n_idx
                + get_data_B_off_within_block(k, n);
    }

    dim_t get_data_B_off_within_block(int k, int n) const {
        using namespace format_tag;

        if (!bgmmc_.blocked_B) return 0;

        int x0 = k % bgmmc_.wei_k_blk;
        int x1 = n % bgmmc_.wei_n_blk;
        dim_t offset = (x0 / vnni_factor) * vnni_factor * bgmmc_.wei_n_blk
                + x1 * vnni_factor + x0 % vnni_factor;
        return bgmmc_.b_dt_sz * offset;
    }

    dim_t get_data_C_off(int b, int m, int n) const {
        return bgmmc_.C_strides[2] * b + bgmmc_.C_strides[1] * m
                + bgmmc_.C_strides[0] * n;
    }

    const char *get_bias_ptr(int n) const {
        if (!bgmmc_.with_bias) return nullptr;

        return bias_ptr_ + n * bgmmc_.bias_dt_sz;
    }

    int32_t *get_s8s8_comp_ptr(int ithr, int b, int n_blk_idx) const {
        if (!bgmmc_.s8s8_compensation_required) return nullptr;

        const int n_blk_local = bgmmc_.use_buffer_b
                ? n_blk_idx % bgmmc_.N_chunk_size
                : n_blk_idx;
        return s8s8_compensation_ptr_ + ithr * bgmmc_.s8s8_comp_ithr_str
                + b * bgmmc_.s8s8_comp_b_str
                + n_blk_local * bgmmc_.s8s8_comp_n_str;
    }

    const float *get_oscales_ptr(int n) const {
        return oscales_ptr_ + bgmmc_.is_oscale_per_n * n;
    }

    const int32_t *get_zp_a_neg_val_ptr() const {
        return &zero_point_a_negative_val_;
    }

    const int32_t *get_zp_b_neg_val_ptr() const {
        return &zero_point_b_negative_val_;
    }

    const int32_t *get_zp_ab_mixed_comp_ptr() const {
        return &zero_point_mixed_ab_compensation_component_;
    }

    const int32_t *get_zp_c_val_ptr() const { return &zero_point_c_val_; }

    int32_t *get_zp_a_compensation_ptr(int ithr, int n_blk_idx) const {
        if (!bgmmc_.has_zero_point_a) return nullptr;

        const int n_blk_local = n_blk_idx % bgmmc_.N_chunk_size;
        int32_t *zp_comp = zero_point_a_compensations_ptr_
                + ithr * bgmmc_.zp_a_comp_elems_per_thr
                + n_blk_local * bgmmc_.zp_a_comp_shift_n;

        if (bgmmc_.blocked_B) {
            // Scale computed in reorder compensation values by zp_a value
            // locally just before usage. Using the single global scaling before
            // parallel section might produce significant overhead for small
            // problems running in multitreaded execution mode
            const int base_offset = n_blk_idx * bgmmc_.wei_n_blk;
            PRAGMA_OMP_SIMD()
            for (int b = 0; b < bgmmc_.wei_n_blk; b++)
                zp_comp[b] = -zero_point_a_negative_val_
                        * reorder_zp_a_comp_ptr_[base_offset + b];
        }
        return zp_comp;
    }

    int32_t *get_zp_b_compensation_result_ptr(int ithr, int m_blk_idx) const {
        if (!bgmmc_.has_zero_point_b) return nullptr;

        const int m_blk_local = m_blk_idx % bgmmc_.M_chunk_size;
        return zero_point_b_compensations_ptr_
                + ithr * bgmmc_.zp_b_comp_elems_per_thr
                + m_blk_local * bgmmc_.zp_b_comp_result_shift_m;
    }

    int32_t *get_zp_b_compensation_buffer_ptr(int ithr, int m_blk_idx) const {
        if (!bgmmc_.has_zero_point_b) return nullptr;

        const int m_blk_local = m_blk_idx % bgmmc_.M_chunk_size;
        return get_zp_b_compensation_result_ptr(ithr, 0)
                + bgmmc_.zp_b_comp_buffer_start
                + m_blk_local * bgmmc_.zp_b_comp_buffer_shift_m;
    }

    char *get_tile_workspace(int ithr) const {
        return is_amx_ ? wsp_tile_ptr_ + ithr * bgmmc_.wsp_tile_per_thr_bytes
                       : nullptr;
    }

    const std::vector<const void *> &get_post_ops_binary_rhs_arg_vec() const {
        return post_ops_binary_rhs_arg_vec_;
    }

    int get_base_brgemm_kernel_idx() const { return base_brg_ker_idx_; }

    bool is_last_K_chunk(int k_chunk_idx) const {
        return k_chunk_idx == bgmmc_.K_chunks - 1;
    }

    int get_brgemm_batch_size(int k_chunk_idx) const {
        const int last_brgemm_batch_size
                = (nstl::max(bgmmc_.K, bgmmc_.K_blk)
                          - k_chunk_idx * bgmmc_.K_chunk_elems)
                / bgmmc_.K_blk;
        return is_last_K_chunk(k_chunk_idx) ? last_brgemm_batch_size
                                            : bgmmc_.brgemm_batch_size;
    }

    int get_parallel_work_amount() const { return parallel_work_amount_; }
    int get_num_threads_for_k() const { return nthr_k_; }
    bool parallel_reduction_is_used() const {
        return nthr_k_ > 1 && bgmmc_.K_chunks > 1;
    }
    int get_num_threads_for_bmn() const { return nthr_bmn_; }
    // ithr = ithr_k * nthr_bmn + ithr_bmn
    int get_thread_idx_for_k(int ithr) const {
        if (ithr >= num_threads_used_) return -1;
        const int ithr_k = ithr / nthr_bmn_;
        return ithr_k < bgmmc_.K_chunks ? ithr_k : -1;
    }
    int get_thread_idx_for_bmn(int ithr) const {
        if (ithr >= num_threads_used_) return -1;
        const int ithr_bmn = ithr % nthr_bmn_;
        return ithr_bmn < parallel_work_amount_ ? ithr_bmn : -1;
    }
    int get_num_threads_for_parallelization() const { return nthr_; }

private:
    bool is_amx_;
    const brgemm_matmul_conf_t &bgmmc_;
    const char *data_A_ptr_;
    const char *data_B_ptr_;
    char *data_C_ptr_;
    brgemm_batch_element_t *batch_element_ptr_;

    char *buf_A_ptr_;
    char *buf_B_ptr_;
    char *buf_C_ptr_;

    char *wsp_tile_ptr_;
    const char *bias_ptr_;
    const float *oscales_ptr_;
    int32_t *s8s8_compensation_ptr_;

    int32_t *zero_point_a_compensations_ptr_;
    int32_t *zero_point_b_compensations_ptr_;
    int32_t *reorder_zp_a_comp_ptr_;

    int32_t zero_point_a_negative_val_;
    int32_t zero_point_b_negative_val_;
    int32_t zero_point_mixed_ab_compensation_component_;
    int32_t zero_point_c_val_;
    std::vector<const void *> post_ops_binary_rhs_arg_vec_;

    int base_brg_ker_idx_;
    int vnni_factor;

    // parallelization parameters
    int parallel_work_amount_;
    int nthr_, nthr_k_, nthr_bmn_, num_threads_used_;
};

template struct brgemm_matmul_t<avx512_core_bf16_amx_int8>;
template struct brgemm_matmul_t<avx512_core_bf16_amx_bf16>;
template struct brgemm_matmul_t<avx512_core_bf16>;
template struct brgemm_matmul_t<avx512_core_vnni>;
template struct brgemm_matmul_t<avx512_core>;

} // namespace matmul
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
