/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/x64/amx_tile_configure.hpp"
#include "cpu/x64/cpu_barrier.hpp"
#include "cpu/x64/injectors/jit_uni_binary_injector.hpp"
#include "cpu/x64/jit_brgemm_inner_product.hpp"
#include "cpu/x64/jit_transpose_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::data_type;
using namespace dnnl::impl::format_tag;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::status;
using namespace dnnl::impl::utils;

using namespace nstl;

#define get_blk_off(d, dt, ...) \
    (types::data_type_size((dt)) * (d).blk_off(__VA_ARGS__))

namespace {
template <typename ker_type>
void copy_data_chunk(ker_type &ker, char *tr_data, const char *data,
        int os_work, bool is_last_blk) {
    auto ctx = jit_brgemm_copy_to_coarse_t::ctx_t();
    ctx.data = (void *)data;
    ctx.tr_data = (void *)tr_data;
    ctx.os_work = os_work;
    ctx.last_row_blk = is_last_blk ? 1 : 0;
    (*ker)(&ctx);
}
} // namespace

template <cpu_isa_t isa>
void brgemm_inner_product_fwd_t<isa>::execute_forward(
        const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const char *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const char *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const char *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(char *, DNNL_ARG_DST);
    const auto post_ops_binary_rhs_arg_vec
            = binary_injector::prepare_binary_args(
                    pd()->attr()->post_ops_, ctx);

    memory_tracking::grantor_t scratchpad = ctx.get_scratchpad_grantor();
    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));

    const float *oscales = pd()->attr()->output_scales_.scales_;

    const auto &jbgp = pd()->jbgp_;
    const bool is_f32 = everyone_is(f32, jbgp.src_dt, jbgp.wei_dt, jbgp.dst_dt);

    const size_t src_dt_size = types::data_type_size(jbgp.src_dt);
    const size_t bia_dt_size
            = jbgp.with_bias ? types::data_type_size(jbgp.bia_dt) : 0;
    const size_t acc_dt_size = types::data_type_size(jbgp.acc_dt);
    const size_t dst_dt_size = types::data_type_size(jbgp.dst_dt);

    auto addr_batch_global = scratchpad.template get<brgemm_batch_element_t>(
            key_brgemm_primitive_batch);
    auto a_buffer_global = (jbgp.use_buffer_a)
            ? scratchpad.template get<char>(key_brgemm_primitive_buffer_a)
            : nullptr;
    auto c_buffer_global = (jbgp.use_buffer)
            ? scratchpad.template get<char>(key_brgemm_primitive_buffer)
            : nullptr;
    static constexpr bool is_amx = (isa == avx512_core_bf16_amx_int8
            || isa == avx512_core_bf16_amx_bf16);
    auto wsp_tile_base = is_amx
            ? ctx.get_scratchpad_grantor().template get<char>(
                    key_conv_amx_tile_buffer)
            : nullptr;

    const int ic_chunks = div_up(jbgp.nb_ic, jbgp.nb_ic_blocking);

    const bool are_post_ops_applicable = one_of(true, jbgp.with_sum,
            jbgp.with_bias, jbgp.with_scales, jbgp.with_eltwise,
            jbgp.with_binary, jbgp.acc_dt != jbgp.dst_dt, jbgp.signed_input);

    size_t offset = types::data_type_size(jbgp.wei_dt)
            * (weights_d.size() - weights_d.additional_buffer_size());
    auto compensation = (jbgp.signed_input)
            ? reinterpret_cast<const int32_t *>(&weights[offset])
            : nullptr;

    bool is_os_tail = (jbgp.mb < jbgp.os_block);
    bool is_oc_tail = (jbgp.oc < jbgp.oc_block);
    int base_brg_ker_idx = brgemm_inner_product_utils::
            get_brg_kernel_index( // TODO: Can be calculated on initialization stage
                    jbgp, false, is_os_tail, is_oc_tail, false);

    const auto ker = [&](int ithr_oc_mb, int nthr_oc_mb, int ithr_ic, int n,
                             int ocb, int icc, bool do_init, int buffer_a_osb,
                             bool copy_buffer_a) {
        const int ithr = nthr_oc_mb * ithr_ic + ithr_oc_mb;
        auto addr_batch = addr_batch_global + ithr * jbgp.adjusted_batch_size;

        const size_t a_buffer_osb_stride
                = src_dt_size * jbgp.LDA * jbgp.os_block;
        const size_t a_buffer_per_thr
                = a_buffer_osb_stride * jbgp.nb_os_blocking;
        auto a_buffer = (jbgp.use_buffer_a)
                ? a_buffer_global + ithr * a_buffer_per_thr
                        + buffer_a_osb * a_buffer_osb_stride
                : nullptr;

        const int oc = ocb * jbgp.oc_block;
        const size_t dst_off = get_blk_off(dst_d, jbgp.dst_dt, n, oc);

        const bool use_c_buffer = (jbgp.with_sum)
                || (jbgp.use_buffer && (jbgp.nthr_ic_b == 1 || ithr_ic > 0));

        char *c_buffer = nullptr;
        if (use_c_buffer) {
            const size_t c_buf_thr_idx = jbgp.nthr_ic_b <= 1
                    ? ithr
                    : (jbgp.acc_dt != jbgp.dst_dt || jbgp.with_sum
                                    ? ithr_ic
                                    : ithr_ic - 1);
            const size_t c_buf_num_rows = jbgp.nthr_ic_b > 1 ? jbgp.mb : jbgp.M;
            const size_t c_buffer_shift
                    = c_buf_thr_idx * c_buf_num_rows * jbgp.LDC;
            const size_t c_buffer_off = acc_dt_size * c_buffer_shift
                    + (jbgp.nthr_ic_b > 1 ? acc_dt_size * dst_off / dst_dt_size
                                          : 0);
            c_buffer = c_buffer_global + c_buffer_off;
        }

        char *wsp_tile = is_amx ? wsp_tile_base + ithr * 1024 : nullptr;
        int icb = icc * jbgp.nb_ic_blocking;
        int ic = icb * jbgp.ic_block;

        bool kernel_init = do_init;

        bool is_os_tail = (jbgp.mb - n < jbgp.os_block);
        bool is_oc_tail = (jbgp.oc - oc < jbgp.oc_block);
        bool is_last_ic_chunk = icc == ic_chunks - 1;
        bool is_ic_tail = is_last_ic_chunk && jbgp.K_tail > 0;
        const int remaining_ic_blks
                = (jbgp.use_buffer_a ? utils::rnd_up(jbgp.ic, jbgp.ic_block)
                                     : jbgp.ic)
                - ic;
        const int gemm_batch = nstl::min(
                jbgp.gemm_batch_size, remaining_ic_blks / jbgp.ic_block);

        int brg_ker_idx = brgemm_inner_product_utils::get_brg_kernel_index(
                jbgp, kernel_init, is_os_tail, is_oc_tail, false);
        auto brg_kernel = brg_kernels_[brg_ker_idx].get();

        if (copy_buffer_a) {
            auto src_ptr = src + get_blk_off(src_d, jbgp.src_dt, n, ic);
            copy_data_chunk(copy_src_kernel_, a_buffer, src_ptr,
                    is_os_tail ? jbgp.mb - n : jbgp.os_block, is_last_ic_chunk);
        }
        if (gemm_batch > 0 && brg_kernel != nullptr) {
            if (is_amx && (is_os_tail || is_oc_tail))
                amx_tile_configure(&brg_kernel_palettes_[brg_ker_idx][0]);
            for (int b = 0; b < gemm_batch; b++) {
                auto A_ptr = jbgp.use_buffer_a
                        ? (a_buffer + src_dt_size * b * jbgp.ic_block)
                        : (src
                                + get_blk_off(src_d, jbgp.src_dt, n,
                                        ic + b * jbgp.ic_block));
                addr_batch[b].ptr.A = A_ptr;
                addr_batch[b].ptr.B = weights
                        + get_blk_off(weights_d, jbgp.wei_dt, ocb, icb + b);
            }

            auto ptr_D = dst + dst_off;
            auto ptr_C = use_c_buffer ? c_buffer : ptr_D;

            if (jbgp.nthr_ic_b == 1 && are_post_ops_applicable
                    && is_last_ic_chunk && !is_ic_tail) {
                void *scratch = is_amx
                        ? static_cast<void *>(wsp_tile)
                        : (jbgp.signed_input ? static_cast<void *>(
                                   const_cast<int *>(&compensation[oc]))
                                             : nullptr);
                auto ptr_bias
                        = jbgp.with_bias ? bias + bia_dt_size * oc : nullptr;
                const brgemm_post_ops_data_t post_ops_data {
                        static_cast<const void *>(ptr_bias),
                        &oscales[jbgp.is_oc_scale * oc],
                        post_ops_binary_rhs_arg_vec.data(),
                        static_cast<size_t>(oc), 0, dst};

                brgemm_kernel_execute_postops(brg_kernel, gemm_batch,
                        addr_batch, (void *)ptr_C, (void *)ptr_D, post_ops_data,
                        scratch);
            } else {
                brgemm_kernel_execute(brg_kernel, gemm_batch, addr_batch,
                        (void *)ptr_C, is_amx ? (void *)wsp_tile : nullptr);
            }

            if (is_amx && (is_os_tail || is_oc_tail))
                amx_tile_configure(&brg_kernel_palettes_[base_brg_ker_idx][0]);
        }

        if (is_ic_tail) {
            assert(!jbgp.use_buffer_a);
            int ic_block = gemm_batch * jbgp.K / jbgp.ic_block;
            addr_batch[0].ptr.A = src
                    + get_blk_off(src_d, jbgp.src_dt, n,
                            ic + ic_block * jbgp.ic_block);
            addr_batch[0].ptr.B = weights
                    + get_blk_off(weights_d, jbgp.wei_dt, ocb, icb + ic_block);

            auto use_init_ker = (kernel_init && gemm_batch == 0);
            int brg_ker_idx = brgemm_inner_product_utils::get_brg_kernel_index(
                    jbgp, use_init_ker, is_os_tail, is_oc_tail, true);
            auto brg_kernel_ic_tail = brg_kernels_[brg_ker_idx].get();
            if (is_amx)
                amx_tile_configure(&brg_kernel_palettes_[brg_ker_idx][0]);
            auto ptr_D = dst + dst_off;
            auto ptr_C = use_c_buffer ? c_buffer : ptr_D;
            if (jbgp.nthr_ic_b == 1 && are_post_ops_applicable) {
                void *scratch = is_amx
                        ? static_cast<void *>(wsp_tile)
                        : (jbgp.signed_input ? static_cast<void *>(
                                   const_cast<int *>(&compensation[oc]))
                                             : nullptr);
                auto ptr_bias
                        = jbgp.with_bias ? bias + bia_dt_size * oc : nullptr;
                const brgemm_post_ops_data_t post_ops_data {
                        static_cast<const void *>(ptr_bias),
                        &oscales[jbgp.is_oc_scale * oc],
                        post_ops_binary_rhs_arg_vec.data(),
                        static_cast<size_t>(oc)};

                brgemm_kernel_execute_postops(brg_kernel_ic_tail, 1, addr_batch,
                        (void *)ptr_C, (void *)ptr_D, post_ops_data, scratch);
            } else {
                brgemm_kernel_execute(brg_kernel_ic_tail, 1, addr_batch,
                        (void *)ptr_C, is_amx ? (void *)wsp_tile : nullptr);
            }
            if (is_amx)
                amx_tile_configure(&brg_kernel_palettes_[base_brg_ker_idx][0]);
        }
    };

    const int os_chunks = div_up(jbgp.nb_os, jbgp.nb_os_blocking);
    const int oc_chunks = div_up(jbgp.nb_oc, jbgp.nb_oc_blocking);
    const int work_amount = oc_chunks * os_chunks;

    const auto init_thr_groups
            = [&](const int ithr, const int nthr, int &nthr_ic, int &nthr_oc_mb,
                      int &ithr_ic, int &ithr_oc_mb) {
                  nthr_ic = jbgp.nthr_ic_b <= nthr ? jbgp.nthr_ic_b : 1;
                  nthr_oc_mb = nthr / nthr_ic;
                  ithr_ic = ithr / nthr_oc_mb;
                  ithr_oc_mb = ithr % nthr_oc_mb;
                  if (ithr_oc_mb >= work_amount || ithr_ic >= ic_chunks
                          || ithr >= rnd_dn(nthr, nthr_ic))
                      return false;
                  return true;
              };

    // If work_amount == 1 we limit num_threads to 1 as parallel(1, ...) does
    // not create parallel section at all. We do not limit num_threads
    // for 1 < work_amount < dnnl_get_max_threads() case to avoid potential
    // overhead on spawning different number of OMP threads from layer to layer.
    const int num_threads = (work_amount == 1 ? 1 : 0);
    parallel(num_threads, [&](const int ithr, const int nthr) {
        int nthr_ic {1}, nthr_oc_mb {1}, ithr_ic {0}, ithr_oc_mb {0};
        bool ok = init_thr_groups(
                ithr, nthr, nthr_ic, nthr_oc_mb, ithr_ic, ithr_oc_mb);
        if (!ok) return;

        int start {0}, end {0};
        balance211(work_amount, nthr_oc_mb, ithr_oc_mb, start, end);

        int icc_start {0}, icc_end {ic_chunks};
        if (nthr_ic > 1)
            balance211(ic_chunks, nthr_ic, ithr_ic, icc_start, icc_end);

        const int icc_work = icc_end - icc_start;

        if (is_amx)
            amx_tile_configure(&brg_kernel_palettes_[base_brg_ker_idx][0]);

        int occ {0}, osc {0};
        nd_iterator_init(start, osc, os_chunks, occ, oc_chunks);
        while (start < end) {
            int ocb_s = occ * jbgp.nb_oc_blocking;
            int ocb_e = nstl::min(ocb_s + jbgp.nb_oc_blocking, jbgp.nb_oc);
            int ocb_work = ocb_e - ocb_s;

            int osb_s = osc * jbgp.nb_os_blocking;
            int osb_e = nstl::min(osb_s + jbgp.nb_os_blocking, jbgp.nb_os);
            int osb_work = osb_e - osb_s;

            // Each thread runs the below loops:
            int loop_start = 0, loop_end = icc_work * osb_work * ocb_work;
            int icc = 0, osb = 0, ocb = 0;

            // If buffer is required, then inner-most loop will be over icc_work
            const bool ocb_inner_most = is_f32 && !jbgp.use_buffer;
            if (ocb_inner_most)
                nd_iterator_init(
                        0, icc, icc_work, osb, osb_work, ocb, ocb_work);
            else
                nd_iterator_init(
                        0, osb, osb_work, ocb, ocb_work, icc, icc_work);

            while (loop_start < loop_end) {
                const int n = (osb + osb_s) * jbgp.os_block;
                const int cur_icc = icc + icc_start;
                const bool copy_buffer_a = jbgp.use_buffer_a
                        && IMPLICATION(ocb_inner_most, ocb == 0);
                ker(ithr_oc_mb, nthr_oc_mb, ithr_ic, n, ocb + ocb_s, cur_icc,
                        cur_icc == icc_start, osb, copy_buffer_a);

                ++loop_start;
                if (ocb_inner_most)
                    nd_iterator_step(
                            icc, icc_work, osb, osb_work, ocb, ocb_work);
                else
                    nd_iterator_step(
                            osb, osb_work, ocb, ocb_work, icc, icc_work);
            }

            ++start;
            nd_iterator_step(osc, os_chunks, occ, oc_chunks);
        }
        if (is_amx) amx_tile_release();
    });

    if (jbgp.nthr_ic_b > 1) {
        assert(jbgp.use_buffer && is_f32);

        const auto get_dst_reduced_off = [&](int ithr_ic, int osb, int ocb) {
            assert(jbgp.nthr_ic_b > 1);
            int os = osb * jbgp.os_block;
            int oc = ocb * jbgp.oc_block;
            const size_t dst_off = get_blk_off(dst_d, jbgp.dst_dt, os, oc);
            if (ithr_ic == 0) return dst_off;
            assert(ithr_ic > 0);
            const size_t ic_buf_idx = jbgp.with_sum ? ithr_ic : ithr_ic - 1;
            return dst_off + (ic_buf_idx * jbgp.mb * jbgp.LDC * acc_dt_size);
        };

        parallel(num_threads, [&](const int ithr, const int nthr) {
            int nthr_ic {1}, nthr_oc_mb {1}, ithr_ic {0}, ithr_oc_mb {0};
            bool ok = init_thr_groups(
                    ithr, nthr, nthr_ic, nthr_oc_mb, ithr_ic, ithr_oc_mb);
            if (!ok) return;

            int ocmb_start {0}, ocmb_end {0};
            int start {0}, end {0};
            balance211(
                    work_amount, nthr_oc_mb, ithr_oc_mb, ocmb_start, ocmb_end);
            balance211(ocmb_end - ocmb_start, nthr_ic, ithr_ic, start, end);

            int occ {0}, osc {0};
            nd_iterator_init(
                    ocmb_start + start, osc, os_chunks, occ, oc_chunks);
            while (start < end) {
                int ocb_s = occ * jbgp.nb_oc_blocking;
                int ocb_e = nstl::min(ocb_s + jbgp.nb_oc_blocking, jbgp.nb_oc);

                int osb_s = osc * jbgp.nb_os_blocking;
                int osb_e = nstl::min(osb_s + jbgp.nb_os_blocking, jbgp.nb_os);

                for (int osb = osb_s; osb < osb_e; ++osb) {
                    int cur_os_block = nstl::min(
                            jbgp.os - osb * jbgp.os_block, jbgp.os_block);
                    const bool is_os_tail = cur_os_block < jbgp.os_block;
                    const int cur_oc_chunk_size
                            = nstl::min(jbgp.LDC, ocb_e * jbgp.oc_block)
                            - ocb_s * jbgp.oc_block;
                    char *dst_reduced = (jbgp.with_sum ? c_buffer_global : dst)
                            + get_dst_reduced_off(0, osb, ocb_s);
                    const size_t os_offset = jbgp.LDC * acc_dt_size;
                    for (int ic_buf = 0; ic_buf < nthr_ic - 1; ++ic_buf) {
                        const char *c_buffer = c_buffer_global
                                + get_dst_reduced_off(ic_buf + 1, osb, ocb_s);
                        for (int os = 0; os < cur_os_block; ++os) {
                            acc_ker_->accumulate(
                                    (float *)(dst_reduced + os * os_offset),
                                    (float *)(c_buffer + os * os_offset),
                                    cur_oc_chunk_size);
                        }
                    }
                    if (are_post_ops_applicable) {
                        for (int ocb = ocb_s; ocb < ocb_e; ++ocb) {
                            const bool is_oc_tail
                                    = (jbgp.oc - ocb * jbgp.oc_block
                                            < jbgp.oc_block);
                            const int brg_ker_idx = brgemm_inner_product_utils::
                                    get_brg_kernel_index(jbgp, false,
                                            is_os_tail, is_oc_tail, false);
                            const auto brg_kernel
                                    = brg_kernels_[brg_ker_idx].get();
                            const int os = osb * jbgp.os_block;
                            const int oc = ocb * jbgp.oc_block;
                            const auto ptr_bias = jbgp.with_bias
                                    ? bias + bia_dt_size * oc
                                    : nullptr;
                            auto ptr_D = dst
                                    + get_blk_off(dst_d, jbgp.dst_dt, os, oc);
                            auto ptr_C = (jbgp.with_sum ? c_buffer_global : dst)
                                    + get_dst_reduced_off(0, osb, ocb);

                            char *wsp_tile = is_amx
                                    ? wsp_tile_base + ithr * 1024
                                    : nullptr;

                            void *scratch = is_amx
                                    ? static_cast<void *>(wsp_tile)
                                    : (jbgp.signed_input ? static_cast<void *>(
                                               const_cast<int *>(
                                                       &compensation[oc]))
                                                         : nullptr);

                            const brgemm_post_ops_data_t post_ops_data {
                                    static_cast<const void *>(ptr_bias),
                                    &oscales[jbgp.is_oc_scale * oc],
                                    post_ops_binary_rhs_arg_vec.data(),
                                    static_cast<size_t>(oc), 0, dst, 0, nullptr,
                                    nullptr, nullptr, true /* skip_accm */};
                            brgemm_kernel_execute_postops(brg_kernel, 0,
                                    nullptr, (void *)ptr_C, (void *)ptr_D,
                                    post_ops_data, scratch);
                        }
                    }
                }
                ++start;
                nd_iterator_step(osc, os_chunks, occ, oc_chunks);
            }
        });
    }
}

template struct brgemm_inner_product_fwd_t<avx512_core>;
template struct brgemm_inner_product_fwd_t<avx512_core_bf16>;
template struct brgemm_inner_product_fwd_t<avx512_core_vnni>;
template struct brgemm_inner_product_fwd_t<avx512_core_bf16_amx_bf16>;
template struct brgemm_inner_product_fwd_t<avx512_core_bf16_amx_int8>;

template <cpu_isa_t isa>
void brgemm_inner_product_bwd_data_t<isa>::execute_backward_data(
        const exec_ctx_t &ctx) const {

    auto diff_dst_ = CTX_IN_MEM(const char *, DNNL_ARG_DIFF_DST);
    auto weights_ = CTX_IN_MEM(const char *, DNNL_ARG_WEIGHTS);
    auto diff_src_ = CTX_OUT_MEM(char *, DNNL_ARG_DIFF_SRC);

    auto diff_src = const_cast<char *>(diff_src_);
    auto weights = const_cast<char *>(weights_);
    auto diff_dst = const_cast<char *>(diff_dst_);

    const memory_desc_wrapper diff_src_d(pd()->diff_src_md());
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));

    const auto &jbgp = pd()->jbgp_;

    static constexpr bool is_amx = (isa == avx512_core_bf16_amx_bf16);
    const bool is_f32 = everyone_is(f32, jbgp.src_dt, jbgp.wei_dt, jbgp.dst_dt);
    const bool is_bf16_bf16_out = everyone_is(bf16, jbgp.wei_dt, jbgp.dst_dt)
            && everyone_is(bf16, jbgp.src_dt);
    const bool is_bf16_f32_out = everyone_is(bf16, jbgp.wei_dt, jbgp.dst_dt)
            && everyone_is(f32, jbgp.src_dt);

    const dim_t wei_dt_size = types::data_type_size(jbgp.wei_dt);

    memory_tracking::grantor_t scratchpad = ctx.get_scratchpad_grantor();
    brgemm_batch_element_t *addr_batch_global
            = scratchpad.template get<brgemm_batch_element_t>(
                    key_brgemm_primitive_batch);
    char *c_buffer_global = (jbgp.use_buffer)
            ? scratchpad.template get<char>(key_brgemm_primitive_buffer)
            : nullptr;
    char *b_buffer_global = jbgp.use_buffer_b
            ? scratchpad.template get<char>(key_brgemm_primitive_buffer_b)
            : nullptr;
    char *a_buffer_global = jbgp.use_buffer_a
            ? scratchpad.template get<char>(key_brgemm_primitive_buffer_a)
            : nullptr;
    auto wsp_tile_base = is_amx
            ? ctx.get_scratchpad_grantor().template get<char>(
                    key_conv_amx_tile_buffer)
            : nullptr;

    const int oc_chunks = div_up(jbgp.nb_oc, jbgp.nb_oc_blocking);
    bool is_os_tail = (jbgp.mb < jbgp.os_block);
    bool is_ic_tail = (jbgp.ic < jbgp.ic_block);
    bool is_oc_tail = (jbgp.oc < jbgp.oc_block) && !jbgp.use_buffer_a;

    const dim_t acc_dt_sz = types::data_type_size(jbgp.acc_dt);
    const dim_t src_dt_sz = types::data_type_size(jbgp.src_dt);

    const int base_brg_ker_idx = brgemm_inner_product_utils::
            get_brg_kernel_index( // TODO: Can be calculated on initialization stage
                    jbgp, false, is_os_tail, is_ic_tail, is_oc_tail);

    const auto get_weights_ptr = [&](int icb, int ocb) {
        int fwd_ic_block = (is_amx) ? 2 * jbgp.simd_w : jbgp.simd_w;
        int fwd_oc_block = 0;
        switch (jbgp.wei_tag) {
            case OI16i64o:
            case OIw16i64o:
            case OIhw16i64o:
            case OIdhw16i64o:
            case OI8i64o2i:
            case OIw8i64o2i:
            case OIhw8i64o2i:
            case OIdhw8i64o2i:
            case OI16i64o2i:
            case OIw16i64o2i:
            case OIhw16i64o2i:
            case OIdhw16i64o2i: fwd_oc_block = 4 * jbgp.simd_w; break;
            case OI16i32o:
            case OIw16i32o:
            case OIhw16i32o:
            case OIdhw16i32o:
            case OI8i32o2i:
            case OIw8i32o2i:
            case OIhw8i32o2i:
            case OIdhw8i32o2i:
            case OI16i32o2i:
            case OIw16i32o2i:
            case OIhw16i32o2i:
            case OIdhw16i32o2i: fwd_oc_block = 2 * jbgp.simd_w; break;
            default: fwd_oc_block = jbgp.simd_w;
        };
        int fwd_icb = icb * jbgp.ic_block / fwd_ic_block;
        int fwd_ocb = ocb * jbgp.oc_block / fwd_oc_block;
        char *ptr_wei_local = weights
                + get_blk_off(weights_d, jbgp.wei_dt, fwd_ocb, fwd_icb);

        int fwd_ocb_simd = (ocb * jbgp.oc_block) % fwd_oc_block;
        int fwd_icb_simd = (icb * jbgp.ic_block) % fwd_ic_block;
        int blk_sz = jbgp.wei_dt == data_type::bf16 ? 2 : 1;

        return ptr_wei_local
                + wei_dt_size
                * (fwd_icb_simd / blk_sz * blk_sz * fwd_oc_block
                        + blk_sz * fwd_ocb_simd);
    };

    const auto transform_b_chunk
            = [&](char *tr_wei, const char *wei, int trans_batch, int current_N,
                      int current_K) {
                  auto ctx = jit_brgemm_trans_wei_t::ctx_t();
                  ctx.src = (void *)wei;
                  ctx.tr_src = (void *)tr_wei;
                  ctx.current_gemm_batch = trans_batch;
                  ctx.current_N = current_N;
                  ctx.current_K = current_K;
                  (*trans_B_kernel_)(&ctx);
              };

    const auto ker = [&](int ithr_ic_mb, int nthr_ic_mb, int ithr_oc,
                             int nthr_oc, int n, int icb, int occ, bool do_init,
                             bool do_b_transpose) {
        const int ithr = nthr_ic_mb * ithr_oc + ithr_ic_mb;
        brgemm_batch_element_t *addr_batch
                = addr_batch_global + ithr * jbgp.adjusted_batch_size;

        const int ic = icb * jbgp.ic_block;
        const int ocb = occ * jbgp.nb_oc_blocking;
        const int oc = ocb * jbgp.oc_block;
        const size_t dsrc_off = get_blk_off(diff_src_d, jbgp.src_dt, n, ic);
        const int adj_buffers = (jbgp.src_dt == f32) ? 1 : 0;
        const size_t c_buf_shift = jbgp.nthr_oc_b > 1
                ? (ithr_oc - adj_buffers)
                        * static_cast<size_t>(jbgp.mb * jbgp.LDC)
                : ithr * static_cast<size_t>(jbgp.LDC * jbgp.M);
        const size_t c_buf_off
                = types::data_type_size(jbgp.acc_dt) * c_buf_shift
                + (jbgp.nthr_oc_b > 1 ? acc_dt_sz * dsrc_off / src_dt_sz : 0);
        bool use_c_buf = false;
        if ((is_f32 || is_bf16_f32_out) && jbgp.use_buffer) {
            use_c_buf = (jbgp.nthr_oc_b == 1 || ithr_oc > 0);
        } else if (is_bf16_bf16_out && jbgp.use_buffer) {
            if (jbgp.nthr_oc_b > 1)
                use_c_buf = true;
            else
                use_c_buf = (jbgp.nthr_oc_b == 1 || ithr_oc > 0);
        }

        const size_t a_buffer_size_per_thr
                = jbgp.os_block * jbgp.LDA * types::data_type_size(jbgp.dst_dt);
        char *c_buffer = use_c_buf ? c_buffer_global + c_buf_off : nullptr;
        char *a_buffer = jbgp.use_buffer_a
                ? a_buffer_global + ithr * a_buffer_size_per_thr
                : diff_dst;
        char *wsp_tile = is_amx ? wsp_tile_base + ithr * 1024 : nullptr;

        bool kernel_init = do_init;

        const bool is_os_tail = (jbgp.mb - n < jbgp.os_block);
        const bool is_ic_tail = (jbgp.ic - ic < jbgp.ic_block);
        const bool is_last_oc_chunk = occ == oc_chunks - 1;
        const bool is_oc_tail = is_last_oc_chunk && jbgp.K_tail > 0;

        const int rnd_oc
                = rnd_up(jbgp.oc, jbgp.use_buffer_a ? jbgp.oc_block : 1);
        const int nb_oc_b
                = nstl::min((rnd_oc - oc) / jbgp.oc_block, jbgp.nb_oc_blocking);

        const int brg_ker_idx
                = brgemm_inner_product_utils::get_brg_kernel_index(
                        jbgp, kernel_init, is_os_tail, is_ic_tail, false);
        auto brg_kernel = brg_kernels_[brg_ker_idx].get();

        const int size_B = jbgp.LDB * rnd_up(jbgp.K, 2);

        const size_t b_buf_shift = jbgp.ip_bwd_d_global_b_transpose
                ? icb * jbgp.nb_oc + ocb
                : ithr * jbgp.gemm_batch_size;
        const size_t b_buf_off = wei_dt_size * b_buf_shift * size_B;
        char *b_buffer = b_buffer_global + b_buf_off;

        char *ptr_D = diff_src + dsrc_off;
        char *ptr_C = use_c_buf ? c_buffer : ptr_D;

        if (jbgp.use_buffer_a)
            copy_data_chunk(copy_diff_dst_kernel_, a_buffer,
                    diff_dst + get_blk_off(diff_dst_d, jbgp.dst_dt, n, oc),
                    is_os_tail ? jbgp.os - n : jbgp.os_block, is_last_oc_chunk);

        if (nb_oc_b > 0 && brg_kernel != nullptr) {
            if (is_amx && (is_os_tail || is_ic_tail))
                amx_tile_configure(&brg_kernel_palettes_[brg_ker_idx][0]);

            for (int oc_block = 0; oc_block < nb_oc_b; oc_block++) {
                addr_batch[oc_block].ptr.A = jbgp.use_buffer_a ? a_buffer
                                + oc_block * jbgp.oc_block
                                        * types::data_type_size(jbgp.dst_dt)
                                                               : diff_dst
                                + get_blk_off(diff_dst_d, jbgp.dst_dt, n,
                                        oc + oc_block * jbgp.oc_block);
                addr_batch[oc_block].ptr.B
                        = b_buffer + wei_dt_size * (oc_block * size_B);
                if (!jbgp.ip_bwd_d_global_b_transpose && do_b_transpose)
                    transform_b_chunk((char *)addr_batch[oc_block].ptr.B,
                            get_weights_ptr(icb, ocb + oc_block), 1,
                            is_ic_tail ? jbgp.ic % jbgp.ic_block
                                       : jbgp.ic_block,
                            jbgp.oc_block);
            }

            if (jbgp.use_buffer && jbgp.nthr_oc_b <= 1 && is_last_oc_chunk
                    && !is_oc_tail) {
                void *scratch
                        = is_amx ? static_cast<void *>(wsp_tile) : nullptr;
                const brgemm_post_ops_data_t empty_po_data {};
                brgemm_kernel_execute_postops(brg_kernel, nb_oc_b, addr_batch,
                        (void *)c_buffer, (void *)ptr_D, empty_po_data,
                        scratch);

            } else {
                brgemm_kernel_execute(brg_kernel, nb_oc_b, addr_batch,
                        (void *)ptr_C, is_amx ? (void *)wsp_tile : nullptr);
            }
            if (is_amx && (is_os_tail || is_ic_tail))
                amx_tile_configure(&brg_kernel_palettes_[base_brg_ker_idx][0]);
        }
        if (is_oc_tail) {
            assert(!jbgp.use_buffer_a);

            const int oc_block = nb_oc_b;
            addr_batch[0].ptr.A = diff_dst
                    + get_blk_off(diff_dst_d, jbgp.dst_dt, n,
                            oc + oc_block * jbgp.oc_block);
            addr_batch[0].ptr.B = b_buffer + wei_dt_size * (oc_block * size_B);
            if (!jbgp.ip_bwd_d_global_b_transpose && do_b_transpose) {
                transform_b_chunk((char *)addr_batch[0].ptr.B,
                        get_weights_ptr(icb, ocb + oc_block), 1,
                        is_ic_tail ? jbgp.ic % jbgp.ic_block : jbgp.ic_block,
                        jbgp.K_tail);
            }

            auto use_init_ker = (kernel_init && nb_oc_b == 0);
            const int brg_kernel_oc_tail_idx
                    = brgemm_inner_product_utils::get_brg_kernel_index(
                            jbgp, use_init_ker, is_os_tail, is_ic_tail, true);
            auto brg_kernel_oc_tail
                    = brg_kernels_[brg_kernel_oc_tail_idx].get();
            if (is_amx)
                amx_tile_configure(
                        &brg_kernel_palettes_[brg_kernel_oc_tail_idx][0]);
            if (jbgp.use_buffer && jbgp.nthr_oc_b <= 1) {
                void *scratch
                        = is_amx ? static_cast<void *>(wsp_tile) : nullptr;
                const brgemm_post_ops_data_t empty_po_data {};
                brgemm_kernel_execute_postops(brg_kernel_oc_tail, 1, addr_batch,
                        (void *)c_buffer, (void *)ptr_D, empty_po_data,
                        scratch);

            } else {
                brgemm_kernel_execute(brg_kernel_oc_tail, 1, addr_batch,
                        (void *)ptr_C, is_amx ? (void *)wsp_tile : nullptr);
            }
            if (is_amx)
                amx_tile_configure(&brg_kernel_palettes_[base_brg_ker_idx][0]);
        }
    };

    const int os_chunks = div_up(jbgp.nb_os, jbgp.nb_os_blocking);
    const int work_amount = jbgp.nb_ic * os_chunks;
    if (jbgp.ip_bwd_d_global_b_transpose && jbgp.use_buffer_b) {
        assert(IMPLICATION(
                jbgp.ip_bwd_d_global_b_transpose, jbgp.nthr_oc_b == 1));
        parallel(0, [&](const int ithr, const int nthr) {
            int start {0}, end {0};
            int max_ch_block = nstl::max(jbgp.ic_block, jbgp.oc_block);
            int ic_chunk_sz = max_ch_block / jbgp.ic_block;
            int oc_chunk_sz = max_ch_block / jbgp.oc_block;
            int nc_ic = utils::div_up(jbgp.nb_ic, ic_chunk_sz);
            int nc_oc = utils::div_up(jbgp.nb_oc, oc_chunk_sz);
            int transp_work_amount = nc_ic * nc_oc;
            balance211(transp_work_amount, nthr, ithr, start, end);
            int icc, occ;
            nd_iterator_init(start, icc, nc_ic, occ, nc_oc);
            while (start < end) {
                int icb_start = icc * ic_chunk_sz;
                int icb_end = nstl::min((icc + 1) * ic_chunk_sz, jbgp.nb_ic);
                int ocb_start = occ * oc_chunk_sz;
                int ocb_end = nstl::min((occ + 1) * oc_chunk_sz, jbgp.nb_oc);
                for_(int icb = icb_start; icb < icb_end; icb++)
                for (int ocb = ocb_start; ocb < ocb_end; ocb++) {
                    int ic = icb * jbgp.ic_block;
                    int oc = ocb * jbgp.oc_block;
                    bool is_ic_tail = (jbgp.ic - ic < jbgp.ic_block);
                    bool is_oc_tail = (jbgp.oc - oc < jbgp.oc_block);
                    const int size_B = jbgp.LDB * rnd_up(jbgp.K, 2);
                    char *b_buffer = b_buffer_global
                            + wei_dt_size
                                    * ((dim_t)icb * jbgp.nb_oc * size_B
                                            + (dim_t)ocb * size_B);

                    transform_b_chunk(b_buffer, get_weights_ptr(icb, ocb), 1,
                            is_ic_tail ? jbgp.ic % jbgp.ic_block
                                       : jbgp.ic_block,
                            is_oc_tail ? jbgp.oc % jbgp.oc_block
                                       : jbgp.oc_block);
                }
                ++start;
                nd_iterator_step(icc, nc_ic, occ, nc_oc);
            }
        });
    }

    parallel(0, [&](const int ithr, const int nthr) {
        const int nthr_oc = jbgp.nthr_oc_b <= nthr ? jbgp.nthr_oc_b : 1;
        const int nthr_ic_mb = nthr / nthr_oc;
        const int ithr_ic_mb = ithr % nthr_ic_mb;
        const int ithr_oc = ithr / nthr_ic_mb;
        if (ithr_ic_mb >= work_amount || ithr_oc >= oc_chunks
                || ithr >= rnd_dn(nthr, nthr_oc))
            return;

        int start {0}, end {0};
        balance211(work_amount, nthr_ic_mb, ithr_ic_mb, start, end);
        int occ_start {0}, occ_end {oc_chunks};
        if (nthr_oc > 1)
            balance211(oc_chunks, nthr_oc, ithr_oc, occ_start, occ_end);

        if (is_amx)
            amx_tile_configure(&brg_kernel_palettes_[base_brg_ker_idx][0]);

        int icb {0}, oss {0};
        nd_iterator_init(start, oss, os_chunks, icb, jbgp.nb_ic);
        while (start < end) {
            const int nb_os_blocking
                    = nstl::min(jbgp.nb_os - oss * jbgp.nb_os_blocking,
                            jbgp.nb_os_blocking);
            const int occ_work = occ_end - occ_start;
            const int loop_iteration = nb_os_blocking * occ_work;

            for (int iter = 0; iter < loop_iteration; ++iter) {
                int osb = 0, occ = occ_start;
                if (jbgp.use_buffer || !is_f32) {
                    osb += iter / occ_work;
                    occ += iter % occ_work;
                } else {
                    occ += iter / nb_os_blocking;
                    osb += iter % nb_os_blocking;
                }
                int n = (oss * jbgp.nb_os_blocking + osb) * jbgp.os_block;
                ker(ithr_ic_mb, nthr_ic_mb, ithr_oc, nthr_oc, n, icb, occ,
                        occ == occ_start, osb == 0 || occ_work > 1);
            }
            ++start;
            nd_iterator_step(oss, os_chunks, icb, jbgp.nb_ic);
        }
        if (is_amx) amx_tile_release();
    });

    if (jbgp.nthr_oc_b > 1) {
        parallel(0, [&](const int ithr, const int nthr) {
            const int nthr_oc = jbgp.nthr_oc_b <= nthr ? jbgp.nthr_oc_b : 1;
            if (nthr_oc <= 1) return;

            const int ddst_elems = jbgp.LDC * jbgp.os;
            const int reduce_chunk_size = 64;
            int start {0}, end {0};
            balance211(div_up(ddst_elems, reduce_chunk_size), nthr, ithr, start,
                    end);
            const dim_t reduce_start = start * reduce_chunk_size;
            const dim_t reduce_finish
                    = nstl::min(end * reduce_chunk_size, ddst_elems);
            if (reduce_finish <= reduce_start) return;
            const dim_t elems_to_reduce = reduce_finish - reduce_start;
            const dim_t acc_dt_sz = types::data_type_size(jbgp.acc_dt);

            char *dsrc_reduced = diff_src + src_dt_sz * reduce_start;
            char *c_buffer_start = c_buffer_global + acc_dt_sz * reduce_start;

            float *out_buffer = (is_f32 || is_bf16_f32_out)
                    ? (float *)dsrc_reduced
                    : (float *)c_buffer_start;
            int oc_buf_idx = is_bf16_bf16_out;
            int oc_buf_end = !is_bf16_bf16_out;
            for (int oc_buf = oc_buf_idx; oc_buf < nthr_oc - oc_buf_end;
                    oc_buf++) {
                const dim_t c_buf_offt = acc_dt_sz
                        * (oc_buf * jbgp.os * jbgp.LDC + reduce_start);
                char *c_buffer = c_buffer_global + c_buf_offt;

                acc_ker_->accumulate((float *)out_buffer, (float *)c_buffer,
                        elems_to_reduce);
                if (is_bf16_bf16_out && oc_buf == (nthr_oc - oc_buf_end) - 1) {
                    cvt_float_to_bfloat16((bfloat16_t *)dsrc_reduced,
                            (const float *)out_buffer, elems_to_reduce);
                }
            }
        });
    }
}

template struct brgemm_inner_product_bwd_data_t<avx512_core>;
template struct brgemm_inner_product_bwd_data_t<avx512_core_bf16_amx_bf16>;
template struct brgemm_inner_product_bwd_data_t<avx512_core_bf16>;

template <cpu_isa_t isa>
struct brgemm_inner_product_bwd_weights_t<isa>::thread_info_t {
    const char *src;
    const char *diff_dst;
    char *diff_weights;
    char *diff_bias;

    const memory_tracking::grantor_t scratchpad;

    char *buffer_a = nullptr;
    char *buffer_b = nullptr;
    char *buffer_c = nullptr;
    char *buffer_bias = nullptr;
    char *wsp_tile_base = nullptr;

    int ithr;
    int ithr_ic_c, ithr_oc_c, ithr_os_c;
    int nthr;
    int nthr_ic_c, nthr_oc_c, nthr_os_c;

    int os_c_start = 0, os_c_end = 0, os_c_work;
    int oc_c_start = 0, oc_c_end = 0, oc_c_work;
    int ic_c_start = 0, ic_c_end = 0, ic_c_work;
    simple_barrier::ctx_t *barrier_ctx;

    thread_info_t(const brgemm_inner_product_bwd_weights_t *self,
            const exec_ctx_t &ctx, int ithr)
        : scratchpad(ctx.get_scratchpad_grantor()), ithr(ithr) {

        constexpr bool is_amx = (isa == avx512_core_bf16_amx_bf16);

        src = CTX_IN_MEM(const char *, DNNL_ARG_SRC);
        diff_dst = CTX_IN_MEM(const char *, DNNL_ARG_DIFF_DST);
        diff_weights = CTX_OUT_MEM(char *, DNNL_ARG_DIFF_WEIGHTS);
        diff_bias = CTX_OUT_MEM(char *, DNNL_ARG_DIFF_BIAS);
        const auto &jbgp = self->pd()->jbgp_;

        buffer_c = (jbgp.use_buffer)
                ? scratchpad.template get<char>(key_brgemm_primitive_buffer)
                : nullptr;

        buffer_bias = (jbgp.with_bias
                              && (jbgp.bia_dt == data_type::bf16
                                      || jbgp.nthr_mb > 1))
                ? scratchpad.template get<char>(key_iprod_bias_bf16_convert_wsp)
                : nullptr;

        buffer_a = scratchpad.template get<char>(key_brgemm_primitive_buffer_a);
        buffer_b = jbgp.use_buffer_b
                ? scratchpad.template get<char>(key_brgemm_primitive_buffer_b)
                : nullptr;

        wsp_tile_base = is_amx
                ? ctx.get_scratchpad_grantor().template get<char>(
                        key_conv_amx_tile_buffer)
                : nullptr;

        nthr = jbgp.nthr;
        nthr_ic_c = jbgp.nthr_ic_b;
        nthr_oc_c = jbgp.nthr_oc_b;
        nthr_os_c = jbgp.nthr_mb;

        ithr_ic_c = ithr % nthr_ic_c;
        ithr_oc_c = ithr / nthr_ic_c % nthr_oc_c;
        ithr_os_c = ithr / nthr_ic_c / nthr_oc_c;

        int oc_chunks = utils::div_up(jbgp.nb_oc, jbgp.nb_oc_blocking);
        int ic_chunks = utils::div_up(jbgp.nb_ic, jbgp.nb_ic_blocking);
        int os_chunks = utils::div_up(jbgp.nb_os, jbgp.nb_os_blocking);

        /* reduction dimension */
        balance211(os_chunks, nthr_os_c, ithr_os_c, os_c_start, os_c_end);
        os_c_work = os_c_end - os_c_start;

        balance211(oc_chunks, nthr_oc_c, ithr_oc_c, oc_c_start, oc_c_end);
        oc_c_work = oc_c_end - oc_c_start;

        balance211(ic_chunks, nthr_ic_c, ithr_ic_c, ic_c_start, ic_c_end);
        ic_c_work = ic_c_end - ic_c_start;

        if (dnnl_thr_syncable())
            barrier_ctx = scratchpad.template get<simple_barrier::ctx_t>(
                    key_conv_wei_bia_reduction_bctx);
    }
};

template <cpu_isa_t isa>
void brgemm_inner_product_bwd_weights_t<isa>::transform_matrix_a_chunk(
        char *tr_src, const char *src, int trans_batch, int current_m,
        int current_k) const {
    auto ctx = jit_brgemm_trans_src_t::ctx_t();
    ctx.src = (void *)src;
    ctx.tr_src = (void *)tr_src;
    ctx.current_gemm_batch = trans_batch;
    ctx.current_M = current_m;
    ctx.current_K = current_k;
    (*trans_A_kernel_)(&ctx);
}

template <cpu_isa_t isa>
void brgemm_inner_product_bwd_weights_t<isa>::transform_matrix_b_chunk(
        char *tr_diff_dst, const char *diff_dst, int trans_batch,
        int current_col_size, int current_row_size) const {
    auto ctx = jit_brgemm_trans_to_vnni_t::ctx_t();
    ctx.src = (void *)diff_dst;
    ctx.tr_src = (void *)tr_diff_dst;
    ctx.current_gemm_batch = trans_batch;
    ctx.current_col_size = current_col_size;
    ctx.current_row_size = current_row_size;
    (*trans_B_kernel_)(&ctx);
}

template <cpu_isa_t isa>
void brgemm_inner_product_bwd_weights_t<isa>::transpose_matrix_c_chunk(
        const thread_info_t *ti, const int ocb, const int icb, int oc_size,
        int ic_size, bool is_reduction) const {
    const auto &jbgp = pd()->jbgp_;

    if (isa == avx512_core_bf16_amx_bf16) {
        auto p = jit_amx_ip_trans_diff_wei::ctx_t();

        const dim_t ext_nb_ic = div_up(jbgp.ic, ext_ic_block_);
        dim_t icb_shift = (icb * (jbgp.ic_block / ext_ic_block_))
                * ext_ic_block_ * ext_oc_block_;

        dim_t ocb_shift = (ocb * (jbgp.oc_block / ext_oc_block_)) * ext_nb_ic
                * ext_ic_block_ * ext_oc_block_;
        dim_t out_offset = ocb_shift + icb_shift;

        p.src = get_wei_acc_ptr(ti, ocb, icb, 0);
        p.dst = (void *)(ti->diff_weights
                + types::data_type_size(jbgp.wei_dt) * out_offset);

        p.last_ic_block = (jbgp.ic <= ext_ic_block_
                                  || (jbgp.nb_ic > 1 && icb == jbgp.nb_ic - 1))
                ? 1
                : 0;
        p.last_oc_block = (jbgp.oc <= ext_oc_block_
                                  || (jbgp.nb_oc > 1 && ocb == jbgp.nb_oc - 1))
                ? 1
                : 0;
        (*diff_wei_trans_kernel_)(&p);
    } else {
        auto ctx = jit_brgemm_trans_to_vnni_t::ctx_t();
        ctx.src = (void *)(get_wei_acc_ptr(ti, ocb, icb, 0));

        ctx.tr_src = (void *)(ti->diff_weights
                + types::data_type_size(jbgp.wei_dt)
                        * get_wei_offset(ocb, icb));

        ctx.current_gemm_batch = 1;
        ctx.current_col_size = oc_size;
        ctx.current_row_size = ic_size;
        (*trans_C_kernel_)(&ctx);
    }
}

template <cpu_isa_t isa>
dim_t brgemm_inner_product_bwd_weights_t<isa>::get_wei_offset(
        int ocb, int icb) const {
    if (isa == avx512_core_bf16_amx_bf16) {
        const auto &jbgp = pd()->jbgp_;
        const dim_t offset
                = jbgp.kd * jbgp.kh * jbgp.kw * jbgp.ic_block * jbgp.oc_block;
        return (ocb * jbgp.nb_ic + icb) * offset;
    } else {
        const memory_desc_wrapper diff_weights_d(pd()->diff_weights_md(0));
        return diff_weights_d.blk_off(ocb, icb);
    }
}

template <cpu_isa_t isa>
char *brgemm_inner_product_bwd_weights_t<isa>::get_wei_acc_ptr(
        const thread_info_t *ti, int ocb, int icb,
        int reduction_buf_idx) const {
    constexpr bool is_amx_bf16 = (isa == avx512_core_bf16_amx_bf16);

    const auto &jbgp = pd()->jbgp_;
    const int reduction_buf_start_idx = jbgp.wei_dt == f32;
    // reduction_buf_idx argument allows manually set up required reduction
    // buffer index, required for reduction and transform diff_weights parts.
    // It has value -1 by default. If reduction_buf_idx < 0 then ti->ithr_os_c
    // is used for calculation of the current reduction index.
    const int buf_idx = reduction_buf_idx >= 0
            ? reduction_buf_idx
            : (ti->ithr_os_c - reduction_buf_start_idx);
    const size_t acc_dt_size = types::data_type_size(jbgp.acc_dt);

    if ((jbgp.nthr_mb > 1 && buf_idx < 0)
            || (jbgp.wei_dt == jbgp.acc_dt && reduction_buf_idx < 0
                    && ti->ithr_os_c == 0)) {
        MAYBE_UNUSED(reduction_buf_idx);
        const int icb_scale = (!is_amx_bf16 || jbgp.wei_dt == jbgp.acc_dt)
                ? jbgp.ic_block / jbgp.simd_w
                : 1;
        const memory_desc_wrapper diff_weights_d(pd()->diff_weights_md(0));
        return (char *)ti->diff_weights
                + get_blk_off(
                        diff_weights_d, jbgp.wei_dt, ocb, icb * icb_scale);
    }

    if (!jbgp.use_buffer) return nullptr;

    const int ocb_l = ocb % jbgp.nb_oc_blocking;
    const int icb_l = icb % jbgp.nb_ic_blocking;

    if (jbgp.nthr_mb > 1 || jbgp.harness == harness_mb_reduction) {
        const size_t icc = icb / jbgp.nb_ic_blocking;
        const size_t occ = ocb / jbgp.nb_oc_blocking;
        const size_t num_ic_chunks = div_up(jbgp.nb_ic, jbgp.nb_ic_blocking);
        const size_t num_oc_chunks = div_up(jbgp.nb_oc, jbgp.nb_oc_blocking);
        const size_t block_size = acc_dt_size * jbgp.ic_block * jbgp.oc_block;
        const size_t chunk_size
                = block_size * jbgp.nb_ic_blocking * jbgp.nb_oc_blocking;
        const size_t reduction_buf_shift
                = num_ic_chunks * num_oc_chunks * chunk_size * buf_idx;
        return ti->buffer_c + reduction_buf_shift
                + (occ * num_ic_chunks + icc) * chunk_size
                + (ocb_l * jbgp.nb_ic_blocking + icb_l) * block_size;
    } else if (jbgp.nthr_mb == 1) {
        MAYBE_UNUSED(reduction_buf_idx);
        const size_t blk_size = acc_dt_size * jbgp.ic_block * jbgp.oc_block;
        const size_t buf_size_per_thread
                = blk_size * jbgp.nb_ic_blocking * jbgp.nb_oc_blocking;
        const size_t offset_within_thread_buf
                = blk_size * (jbgp.nb_ic_blocking * ocb_l + icb_l);
        const size_t offset
                = ti->ithr * buf_size_per_thread + offset_within_thread_buf;
        return ti->buffer_c + offset;
    }

    assert(!"unsupported case");
    return nullptr;
};

template <cpu_isa_t isa>
void brgemm_inner_product_bwd_weights_t<isa>::compute_diff_weights_and_bias(
        const thread_info_t *ti) const {
    auto diff_dst = const_cast<char *>(ti->diff_dst);
    auto diff_bias = ti->diff_bias;

    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());

    const auto &jbgp = pd()->jbgp_;

    const size_t bia_dt_size
            = jbgp.with_bias ? types::data_type_size(jbgp.bia_dt) : 0;
    const size_t acc_dt_size = types::data_type_size(jbgp.acc_dt);

    const int oc_chunk_sz = jbgp.oc_block * jbgp.nb_oc_blocking;

    brgemm_batch_element_t *addr_batch_global
            = ti->scratchpad.template get<brgemm_batch_element_t>(
                    key_brgemm_primitive_batch);

    constexpr bool is_amx_bf16 = (isa == avx512_core_bf16_amx_bf16);
    char *a_buffer_global = ti->buffer_a;
    char *b_buffer_global = ti->buffer_b;
    char *wsp_tile_global = (is_amx_bf16) ? ti->wsp_tile_base : nullptr;
    int os_chunks = utils::div_up(jbgp.nb_os, jbgp.nb_os_blocking);

    const auto get_bia_acc_ptr = [&](int oc) {
        const int reduction_buf_start_idx = jbgp.bia_dt == f32;
        if (jbgp.bia_dt == data_type::bf16
                || (jbgp.nthr_mb > 1
                        && ti->ithr_os_c >= reduction_buf_start_idx)) {
            return ti->buffer_bias
                    + acc_dt_size * (ti->ithr_os_c - reduction_buf_start_idx)
                    * jbgp.oc
                    + acc_dt_size * oc;
        } else {
            return ti->diff_bias + bia_dt_size * oc;
        }
    };

    const auto ker = [&](const int osc, const int icb, const int ocb) {
        dim_t os_chunks_per_thr = utils::div_up(os_chunks, jbgp.nthr_mb);
        int ic_chunks = utils::div_up(jbgp.nb_ic, jbgp.nb_ic_blocking);
        dim_t ic_chunks_per_thr = utils::div_up(ic_chunks, jbgp.nthr_ic_b);

        int osc_l_idx = osc - ti->os_c_start;
        int icb_l_idx = icb - ti->ic_c_start * jbgp.nb_ic_blocking;
        int ocb_l_idx = ocb - ti->oc_c_start * jbgp.nb_oc_blocking;
        dim_t a_buf_idx = osc_l_idx * ic_chunks_per_thr * jbgp.nb_ic_blocking
                + icb_l_idx;
        int b_buf_idx = osc_l_idx;

        constexpr int tile_size = 1024;

        brgemm_batch_element_t *addr_batch
                = addr_batch_global + ti->ithr * jbgp.adjusted_batch_size;
        const int size_A = jbgp.LDA * jbgp.M;
        const int size_B = jbgp.LDB * rnd_up(jbgp.K, 2);
        char *a_buffer = a_buffer_global
                + types::data_type_size(jbgp.src_dt)
                        * ((ti->ithr * os_chunks_per_thr * ic_chunks_per_thr
                                           * jbgp.nb_ic_blocking
                                   + a_buf_idx)
                                * jbgp.gemm_batch_size * jbgp.os_block
                                * jbgp.ic_block);
        char *b_buffer = b_buffer_global ? b_buffer_global
                        + types::data_type_size(jbgp.dst_dt)
                                * ((ti->ithr * os_chunks_per_thr + b_buf_idx)
                                                * jbgp.gemm_batch_size
                                                * jbgp.os_block * jbgp.LDB
                                        + (ocb_l_idx % jbgp.nb_oc_blocking)
                                                * jbgp.oc_block)
                                         : nullptr;

        char *wsp_tile = is_amx_bf16 ? wsp_tile_global + ti->ithr * tile_size
                                     : nullptr;
        int ic = icb * jbgp.ic_block;
        int oc = ocb * jbgp.oc_block;
        int n = osc * jbgp.nb_os_blocking * jbgp.os_block;

        bool kernel_init = (osc == ti->os_c_start);

        bool is_os_tail = jbgp.mb - n < jbgp.os_block * jbgp.nb_os_blocking;
        bool is_ic_tail = jbgp.ic - ic < jbgp.ic_block;
        bool is_oc_tail = jbgp.oc - oc < jbgp.oc_block;
        const int oc_chunk_tail = jbgp.oc % oc_chunk_sz;
        const bool is_last_oc_chunk = jbgp.oc - oc < oc_chunk_sz;
        const int curr_oc_chunk_sz = oc_chunk_tail > 0 && is_last_oc_chunk
                ? oc_chunk_tail
                : oc_chunk_sz;

        const bool transform_weights_to_vnni = jbgp.wei_dt == bf16
                && (jbgp.nthr_mb == 1 || os_chunks == 1)
                && osc == (os_chunks - 1);

        auto nb_os_b = is_os_tail ? (jbgp.mb - n) / jbgp.os_block
                                  : jbgp.nb_os_blocking;

        const int brg_ker_idx
                = brgemm_inner_product_utils::get_brg_kernel_index(
                        jbgp, kernel_init, is_ic_tail, is_oc_tail, false);
        auto brg_kernel = brg_kernels_[brg_ker_idx].get();

        if (kernel_init && (is_ic_tail || is_oc_tail))
            utils::array_set(get_wei_acc_ptr(ti, ocb, icb), 0,
                    types::data_type_size(jbgp.acc_dt) * jbgp.ic_block
                            * jbgp.oc_block);
        if (nb_os_b > 0 && brg_kernel != nullptr) {
            if (is_amx_bf16)
                amx_tile_configure(&brg_kernel_palettes_[brg_ker_idx][0]);
            if (jbgp.use_buffer_a && ocb_l_idx == 0) {
                const memory_desc_wrapper src_d(pd()->src_md());
                auto src_ptr = ti->src
                        + types::data_type_size(jbgp.src_dt)
                                * src_d.blk_off(n, ic);
                transform_matrix_a_chunk(a_buffer, src_ptr, nb_os_b,
                        is_ic_tail ? jbgp.ic % jbgp.ic_block : jbgp.ic_block,
                        jbgp.os_block);
            }

            if (jbgp.use_buffer_b && icb_l_idx == 0
                    && ocb_l_idx % jbgp.nb_oc_blocking == 0) {
                auto diff_dst_ptr = diff_dst
                        + types::data_type_size(jbgp.dst_dt)
                                * diff_dst_d.blk_off(n, oc);
                transform_matrix_b_chunk(b_buffer, diff_dst_ptr, nb_os_b,
                        curr_oc_chunk_sz, jbgp.os_block);
            }

            for (int os_block = 0; os_block < nb_os_b; os_block++) {
                auto a_ptr = a_buffer
                        + types::data_type_size(jbgp.src_dt) * os_block
                                * size_A;
                addr_batch[os_block].ptr.A = a_ptr;
                auto diff_dst_ptr = diff_dst
                        + types::data_type_size(jbgp.dst_dt)
                                * diff_dst_d.blk_off(
                                        n + os_block * jbgp.os_block, oc);
                if (jbgp.use_buffer_b) {
                    auto b_ptr = b_buffer
                            + types::data_type_size(jbgp.dst_dt) * os_block
                                    * size_B;
                    addr_batch[os_block].ptr.B = b_ptr;
                } else {
                    addr_batch[os_block].ptr.B = diff_dst_ptr;
                }
                if (jbgp.with_bias && icb == 0) {
                    brgemm_kernel_diff_bias_t p;
                    auto bias_ptr = diff_bias + bia_dt_size * oc;
                    p.ptr_diff_dst = (void *)addr_batch[os_block].ptr.B;
                    p.ptr_diff_bias_acc = (void *)get_bia_acc_ptr(oc);
                    p.ptr_diff_bias = (void *)bias_ptr;
                    bool is_first = kernel_init && os_block == 0;
                    bool is_last = (jbgp.nthr_mb == 1 || os_chunks == 1)
                            && osc == os_chunks - 1 && os_block == nb_os_b - 1
                            && !is_os_tail;
                    p.flags = 0 | (is_first ? FLAG_REDUCE_FIRST : 0)
                            | (is_last ? FLAG_REDUCE_LAST : 0);

                    (*kernels_db_[false][is_oc_tail])(&p);
                }
            }
            brgemm_kernel_execute(brg_kernel, nb_os_b, addr_batch,
                    (void *)get_wei_acc_ptr(ti, ocb, icb), wsp_tile);
        }

        if (is_os_tail) {
            int os_block = nb_os_b;
            auto a_ptr = a_buffer
                    + types::data_type_size(jbgp.src_dt) * os_block
                            * jbgp.ic_block * jbgp.os_block;
            if (jbgp.use_buffer_a && ocb_l_idx == 0) {
                const memory_desc_wrapper src_d(pd()->src_md());
                auto src_ptr = ti->src
                        + types::data_type_size(jbgp.src_dt)
                                * src_d.blk_off(
                                        n + os_block * jbgp.os_block, ic);
                transform_matrix_a_chunk(a_ptr, src_ptr, 1,
                        is_ic_tail ? jbgp.ic % jbgp.ic_block : jbgp.ic_block,
                        jbgp.mb % jbgp.os_block);
            }

            addr_batch[0].ptr.A = a_ptr;
            auto diff_dst_ptr = diff_dst
                    + types::data_type_size(jbgp.dst_dt)
                            * diff_dst_d.blk_off(
                                    n + os_block * jbgp.os_block, oc);
            if (jbgp.use_buffer_b) {
                auto b_ptr = b_buffer
                        + types::data_type_size(jbgp.dst_dt) * os_block
                                * jbgp.os_block * jbgp.LDB;
                if (icb_l_idx == 0 && ocb_l_idx % jbgp.nb_oc_blocking == 0)
                    transform_matrix_b_chunk(b_ptr, diff_dst_ptr, 1,
                            curr_oc_chunk_sz, jbgp.mb % jbgp.os_block);
                addr_batch[0].ptr.B = b_ptr;
            } else {
                addr_batch[0].ptr.B = diff_dst_ptr;
            }

            if (jbgp.with_bias && icb == 0) {
                brgemm_kernel_diff_bias_t p;
                auto bias_ptr = diff_bias + bia_dt_size * oc;
                p.ptr_diff_dst = (void *)addr_batch[0].ptr.B;
                p.ptr_diff_bias_acc = (void *)get_bia_acc_ptr(oc);
                p.ptr_diff_bias = (void *)bias_ptr;
                bool is_first = kernel_init && os_block == 0;
                bool is_last = (jbgp.nthr_mb == 1 || os_chunks == 1)
                        && osc == os_chunks - 1;
                p.flags = 0 | (is_first ? FLAG_REDUCE_FIRST : 0)
                        | (is_last ? FLAG_REDUCE_LAST : 0);

                (*kernels_db_[true][is_oc_tail])(&p);
            }

            auto use_init_ker = (kernel_init && nb_os_b == 0);
            const int brg_ker_idx_os_tail
                    = brgemm_inner_product_utils::get_brg_kernel_index(
                            jbgp, use_init_ker, is_ic_tail, is_oc_tail, true);
            auto brg_kernel_os_tail = brg_kernels_[brg_ker_idx_os_tail].get();
            if (brg_kernel_os_tail != nullptr) {
                if (is_amx_bf16)
                    amx_tile_configure(
                            &brg_kernel_palettes_[brg_ker_idx_os_tail][0]);
                brgemm_kernel_execute(brg_kernel_os_tail, 1, addr_batch,
                        (void *)get_wei_acc_ptr(ti, ocb, icb), wsp_tile);
            }
        }

        if (transform_weights_to_vnni) {
            transpose_matrix_c_chunk(ti, ocb, icb,
                    is_oc_tail ? jbgp.oc % jbgp.oc_block : jbgp.oc_block,
                    is_ic_tail ? jbgp.ic % jbgp.ic_block : jbgp.ic_block);
        }
    };

    const auto occ_work = (ti->oc_c_end - ti->oc_c_start);
    const auto icc_work = (ti->ic_c_end - ti->ic_c_start);
    const auto osc_work = (ti->os_c_end - ti->os_c_start);

    auto loop_idx = 0;
    const auto loop_end = occ_work * icc_work * osc_work;

    int occ_idx = 0, icc_idx = 0, osc_idx = 0;
    //TODO: Introduce loop order enum for brgemm-based implementations
    const bool osc_loop_outermost = jbgp.harness == harness_mb_reduction;
    if (osc_loop_outermost)
        nd_iterator_init(loop_idx, osc_idx, osc_work, occ_idx, occ_work,
                icc_idx, icc_work);
    else
        nd_iterator_init(loop_idx, occ_idx, occ_work, icc_idx, icc_work,
                osc_idx, osc_work);

    while (loop_idx < loop_end) {
        const int occ = ti->oc_c_start + occ_idx;
        const int icc = ti->ic_c_start + icc_idx;
        const int osc = ti->os_c_start + osc_idx;

        const int ocb_work = nstl::min(
                jbgp.nb_oc_blocking, jbgp.nb_oc - occ * jbgp.nb_oc_blocking);
        const int icb_work = nstl::min(
                jbgp.nb_ic_blocking, jbgp.nb_ic - icc * jbgp.nb_ic_blocking);

        for_(int ocb = 0; ocb < ocb_work; ocb++)
        for (int icb = 0; icb < icb_work; icb++) {
            ker(osc, icc * jbgp.nb_ic_blocking + icb,
                    occ * jbgp.nb_oc_blocking + ocb);
        }

        ++loop_idx;
        if (osc_loop_outermost)
            nd_iterator_step(
                    osc_idx, osc_work, occ_idx, occ_work, icc_idx, icc_work);
        else
            nd_iterator_step(
                    occ_idx, occ_work, icc_idx, icc_work, osc_idx, osc_work);
    }
    if (is_amx_bf16) amx_tile_release();
}

template <cpu_isa_t isa>
void brgemm_inner_product_bwd_weights_t<
        isa>::reduce_and_convert_diff_weights_and_bias(const thread_info_t *ti)
        const {
    const auto &jbgp = pd()->jbgp_;

    if (dnnl_thr_syncable() && jbgp.nthr > 1)
        simple_barrier::barrier(ti->barrier_ctx, jbgp.nthr);
    if (ti->nthr_os_c == 1) return;

    const bool is_bf16_out = jbgp.wei_dt == data_type::bf16;
    const int icb_scale = (!is_bf16_out) ? jbgp.ic_block / jbgp.simd_w : 1;

    const int icb_work = ti->ic_c_work * jbgp.nb_ic_blocking;
    const int ocb_work = ti->oc_c_work * jbgp.nb_oc_blocking;
    const int work = ocb_work * icb_work;

    int os_chunks = utils::div_up(jbgp.nb_os, jbgp.nb_os_blocking);
    int reduce_buffers = nstl::min(ti->nthr_os_c, os_chunks);
    int reduce_buf_idx_start = is_bf16_out;
    int reduce_buf_idx_end = reduce_buffers - !is_bf16_out;

    int start = 0, end = 0;
    balance211(work, ti->nthr_os_c, ti->ithr_os_c, start, end);
    if (start == end) return;

    int icb_l = 0, ocb_l = 0;
    const int acc_size = jbgp.ic_block * jbgp.oc_block;
    for (int ir = reduce_buf_idx_start; ir < reduce_buf_idx_end; ++ir) {
        int counter = start;
        nd_iterator_init(start, ocb_l, ocb_work, icb_l, icb_work);
        while (counter < end) {
            const int ocb = ti->oc_c_start * jbgp.nb_oc_blocking + ocb_l;
            const int icb = ti->ic_c_start * jbgp.nb_ic_blocking + icb_l;
            char *wei_to_reduce = get_wei_acc_ptr(ti, ocb, icb, ir);
            const memory_desc_wrapper diff_weights_d(pd()->diff_weights_md(0));
            char *wei_reduced = is_bf16_out ? get_wei_acc_ptr(ti, ocb, icb, 0)
                                            : ti->diff_weights
                            + get_blk_off(diff_weights_d, jbgp.wei_dt, ocb,
                                    icb * icb_scale);
            acc_ker_->accumulate(
                    (float *)(wei_reduced), (float *)(wei_to_reduce), acc_size);
            if (is_bf16_out && ir + 1 == reduce_buf_idx_end) {
                transpose_matrix_c_chunk(ti, ocb, icb * icb_scale,
                        jbgp.oc_block, jbgp.ic_block, true);
            }
            ++counter;
            nd_iterator_step(ocb_l, ocb_work, icb_l, icb_work);
        }
    }

    if (jbgp.with_bias && ti->ithr_ic_c == 0 && ti->ic_c_work > 0
            && ti->ithr_os_c == 0 && ti->os_c_work > 0 && ti->oc_c_work > 0) {
        const bool is_bf16_bias = jbgp.bia_dt == data_type::bf16;
        float *bias_reduced = is_bf16_bias ? (float *)ti->buffer_bias
                                           : (float *)ti->diff_bias;
        int reduce_buf_idx_start = is_bf16_bias;
        int reduce_buf_idx_end = reduce_buffers - 1;
        int oc_chunk_size = jbgp.nb_oc_blocking * jbgp.oc_block;
        int oc = ti->oc_c_start * oc_chunk_size;
        int acc_size = nstl::min(ti->oc_c_work * oc_chunk_size, jbgp.oc - oc);

        int ir = reduce_buf_idx_start;
        for (; ir < reduce_buf_idx_end; ++ir) {
            float *bias_to_reduce = (float *)ti->buffer_bias + ir * jbgp.oc;
            acc_ker_->accumulate(
                    &bias_reduced[oc], &bias_to_reduce[oc], acc_size);
        }

        if (is_bf16_bias) {
            float *bias_to_reduce = (float *)ti->buffer_bias + ir * jbgp.oc;
            add_floats_and_cvt_to_bfloat16((bfloat16_t *)(ti->diff_bias) + oc,
                    &bias_reduced[oc], &bias_to_reduce[oc], acc_size);
        }
    }
}

template <cpu_isa_t isa>
void brgemm_inner_product_bwd_weights_t<isa>::execute_backward_weights(
        const exec_ctx_t &ctx) const {
    const auto &jbgp = pd()->jbgp_;

    if (dnnl_thr_syncable() && jbgp.nthr > 1) {
        auto scratchpad = ctx.get_scratchpad_grantor();
        simple_barrier::ctx_init(scratchpad.template get<simple_barrier::ctx_t>(
                key_conv_wei_bia_reduction_bctx));
    }

    parallel(jbgp.nthr, [&](const int ithr, const int nthr) {
        thread_info_t thread_info(this, ctx, ithr);
        compute_diff_weights_and_bias(&thread_info);

        if (dnnl_thr_syncable()) {
            reduce_and_convert_diff_weights_and_bias(&thread_info);
        }
    });

    if (!dnnl_thr_syncable()) {
        parallel(jbgp.nthr, [&](const int ithr, const int nthr) {
            thread_info_t thread_info(this, ctx, ithr);
            reduce_and_convert_diff_weights_and_bias(&thread_info);
        });
    }
}

template struct brgemm_inner_product_bwd_weights_t<avx512_core_bf16_amx_bf16>;
template struct brgemm_inner_product_bwd_weights_t<avx512_core_bf16>;
template struct brgemm_inner_product_bwd_weights_t<avx512_core>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
