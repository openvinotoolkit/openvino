/*******************************************************************************
* Copyright 2017-2021 Intel Corporation
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

#include <atomic>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/binary_injector_utils.hpp"
#include "cpu/cpu_primitive.hpp"
#include "cpu/gemm/gemm.hpp"
#include "cpu/gemm_x8s8s32x_conv_zp_src_pad_comp.hpp"
#include "cpu/gemm_x8s8s32x_convolution.hpp"
#include "cpu/ref_io_helper.hpp"
#include "cpu/simple_q10n.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using namespace dnnl::impl::utils;
using namespace dnnl::impl::memory_tracking::names;

const int32_t *mul_zp_src_comp_from_wei_by_zp_src(const int zp_comp_size,
        int32_t *zp_src_comp_scratch_dst,
        const int32_t *const zp_src_comp_from_wei, const int32_t zp_src) {
    static constexpr int cache_line_size
            = platform::get_cache_line_size() / sizeof(int);
    const auto res = std::div(zp_comp_size, cache_line_size);

    if (res.quot) {
        parallel_nd(res.quot, [&](size_t shift_factor) {
            const auto shift = shift_factor * cache_line_size;
            const int32_t *__restrict const src = zp_src_comp_from_wei + shift;
            int32_t *__restrict dst = zp_src_comp_scratch_dst + shift;

            PRAGMA_OMP_SIMD()
            for (int i = 0; i < cache_line_size; ++i) {
                dst[i] = src[i] * zp_src;
            }
        });
    }

    if (res.rem) {
        const auto shift = res.quot * cache_line_size;
        const int32_t *__restrict const src = zp_src_comp_from_wei + shift;
        int32_t *__restrict dst = zp_src_comp_scratch_dst + shift;

        PRAGMA_OMP_SIMD()
        for (int i = 0; i < res.rem; ++i) {
            dst[i] = src[i] * zp_src;
        }
    }

    return zp_src_comp_scratch_dst;
}

static zero_point_call_params_t prepare_zp_params(const conv_gemm_conf_t &jcp,
        const memory_tracking::grantor_t &scratchpad, const int8_t *weights,
        const memory_desc_wrapper &weights_md, bool with_groups,
        const int32_t *zp_src, const int32_t *zp_dst) {

    int32_t *zp_src_comp_pad = nullptr;
    const int32_t *zp_src_comp = nullptr;

    if (jcp.zp.src_exists) {
        const int32_t *zp_src_comp_from_wei = get_src_zp_comp_from_wei(
                weights, weights_md, jcp.signed_input, jcp.ngroups, jcp.oc);
        int32_t *zp_src_comp_scratch
                = scratchpad.get<int32_t>(key_conv_gemm_zp_src_comp);
        static constexpr auto cache_line_size
                = platform::get_cache_line_size() / sizeof(int);
        const auto zp_comp_size = jcp.oc * jcp.ngroups;

        if (jcp.zp.src_is_common) {
            zp_src_comp = mul_zp_src_comp_from_wei_by_zp_src(zp_comp_size,
                    zp_src_comp_scratch, zp_src_comp_from_wei, *zp_src);
        } else
            zp_src_comp = zp_src_comp_from_wei;

        if (jit_gemm_convolution_utils::padding_exists(jcp)) {
            const auto shift = jcp.zp.src_is_common
                    ? utils::rnd_up(zp_comp_size, cache_line_size)
                    : 0;
            zp_src_comp_pad = zp_src_comp_scratch + shift;
            compute_zp_src_comp_pad(jcp, zp_src_comp_pad, zp_src, weights,
                    weights_md, with_groups);
        }
    }

    return {zp_src, zp_dst, zp_src_comp, zp_src_comp_pad};
}

template <data_type_t src_type, data_type_t dst_type>
status_t _gemm_x8s8s32x_convolution_fwd_t<src_type, dst_type>::execute_forward(
        const exec_ctx_t &ctx) const {
    const conv_gemm_conf_t &jcp = this->pd()->jcp_;
    auto src_base = CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC);
    auto wei_base = CTX_IN_MEM(const wei_data_t *, DNNL_ARG_WEIGHTS);
    auto bia_base = CTX_IN_MEM(const char *, DNNL_ARG_BIAS);
    auto dst_base = CTX_OUT_MEM(dst_data_t *, DNNL_ARG_DST);
    DEFINE_ZERO_POINTS_BUFFER(zp_src, DNNL_ARG_SRC);
    DEFINE_ZERO_POINTS_BUFFER(zp_dst, DNNL_ARG_DST);
    const auto post_ops_binary_rhs_arg_vec
            = binary_injector_utils::prepare_binary_args(
                    this->pd()->attr()->post_ops_, ctx);

    DEFINE_INPUT_ZERO_POINTS_BUFFER(input_zp_base, jcp);
    DEFINE_OUTPUT_COMPENSATION_BUFFER(output_compensation_base, jcp);

    auto MB = CTX_IN_BATCH(DNNL_ARG_SRC);

    auto scratchpad = ctx.get_scratchpad_grantor();

    assert(IMPLICATION(jcp.ow_block != jcp.ow, jcp.oh_block == 1));

    const zero_point_call_params_t zp = prepare_zp_params(jcp, scratchpad,
            wei_base, memory_desc_wrapper(pd()->weights_md(0)),
            this->pd()->with_groups(), zp_src, zp_dst);

    std::atomic<status_t> st(status::success);

    parallel(jcp.nthr, [&](const int ithr, const int nthr) {
        status_t st_thr = execute_forward_thr(ithr, nthr, src_base, wei_base,
                bia_base, dst_base, zp, scratchpad,
                post_ops_binary_rhs_arg_vec.data(), ctx, MB,
                input_zp_base, output_compensation_base);

        if (st_thr != status::success) st = st_thr;
    });

    return st;
}

static const int32_t *get_wei_comp(
        const int8_t *weights, const memory_desc_wrapper &weights_md) {
    const size_t comp_off
            = weights_md.size() - weights_md.additional_buffer_size();
    return reinterpret_cast<const int32_t *>(&weights[comp_off]);
}

template <data_type_t src_type, data_type_t dst_type>
status_t
_gemm_x8s8s32x_convolution_fwd_t<src_type, dst_type>::execute_forward_thr(
        const int ithr, const int nthr, const src_data_t *src_base,
        const wei_data_t *wei_base, const char *bia_base, dst_data_t *dst_base,
        const zero_point_call_params_t &zp,
        const memory_tracking::grantor_t &scratchpad,
        const void *post_ops_binary_rhs_arg_vec, const exec_ctx_t &ctx, int MB,
        const uint8_t *input_zp_base, const int32_t *output_compensation_base) const {

    const conv_gemm_conf_t &jcp = this->pd()->jcp_;

    const auto src_md = memory_desc_wrapper(pd()->src_md());
    const size_t src_mb_stride = src_md.blk_off(1);
    const size_t src_g_stride = src_md.blk_off(0, 1) * jcp.ic;

    const auto wei_md = memory_desc_wrapper(pd()->weights_md(0));
    const size_t wei_g_stride = pd()->with_groups() ? wei_md.blk_off(1) : 0;

    const auto dst_md = memory_desc_wrapper(pd()->dst_md());
    const size_t dst_mb_stride = dst_md.blk_off(1);
    const size_t dst_g_stride = dst_md.blk_off(0, 1) * jcp.oc;

    const float *scales = pd()->attr()->output_scales_.scales_;

    const auto &post_ops = pd()->attr()->post_ops_;
    const bool do_sum = post_ops.contain(primitive_kind::sum, 0);
    const float sum_scale = do_sum ? post_ops.entry_[0].sum.scale : 0;

    uint8_t *__restrict col = scratchpad.get<uint8_t>(key_conv_gemm_col)
            + (ptrdiff_t)ithr * jcp.im2col_sz;
    src_data_t *__restrict imtr = scratchpad.get<src_data_t>(key_conv_gemm_imtr)
            + (ptrdiff_t)ithr * jcp.is * jcp.ic;
    acc_data_t *__restrict acc
            = scratchpad.get<acc_data_t>(key_conv_int_dat_in_acc_dt)
            + (ptrdiff_t)ithr * jcp.oh_block * jcp.ow_block * jcp.oc;

    const int32_t *_wei_comp
            = jcp.signed_input ? get_wei_comp(wei_base, wei_md) :
              jcp.with_input_zp ? output_compensation_base : nullptr;

    const bool should_apply_zp_src_comp_pad_jit_pp = false;
    const bool should_apply_zp_src_comp_outside_pp = false;

    dim_t g {0}, n {0}, ohb {0}, owb {0};
    dim_t start = 0, end = 0;

    const bool is_problem_3d = pd()->ndims() == 5;
    assert(IMPLICATION(is_problem_3d,
            jcp.oh_block == jcp.oh && jcp.ow_block == jcp.ow
                    && jcp.ic_block == jcp.ic));

    const dim_t nb_oh = div_up(jcp.oh, jcp.oh_block);
    const dim_t nb_ow = div_up(jcp.ow, jcp.ow_block);
    const dim_t work_amount = jcp.ngroups * MB * nb_oh * nb_ow;
    balance211(work_amount, nthr, ithr, start, end);
    nd_iterator_init(start, n, MB, g, jcp.ngroups, ohb, nb_oh, owb, nb_ow);
    const uint8_t shift = jcp.signed_input ? 128 : 0;
    parallel_nd_legacy(jcp.im2col_sz, [&](ptrdiff_t i) { col[i] = shift; });

    status_t st = status::success;

    for (dim_t iwork = start; iwork < end; ++iwork) {
        const int oh = ohb * jcp.oh_block;
        const int ow = owb * jcp.ow_block;
        const src_data_t *__restrict src
                = src_base + n * src_mb_stride + g * src_g_stride;
        const wei_data_t *__restrict wei = wei_base + g * wei_g_stride;
        const int32_t *__restrict wei_comp
                = _wei_comp ? _wei_comp + g * jcp.oc : nullptr;
        const int h_step = nstl::min(jcp.oh_block, jcp.oh - oh);
        const int w_step = nstl::min(jcp.ow_block, jcp.ow - ow);
        if (jcp.im2col_sz && is_problem_3d)
            jit_gemm_convolution_utils::transpose_dt<src_data_t>(
                    jcp, src, imtr);

        for (int od = 0; od < jcp.od; od++) {
            dst_data_t *__restrict dst = dst_base + n * dst_mb_stride
                    + g * dst_g_stride
                    + ((od * jcp.oh + oh) * jcp.ow + ow) * jcp.dst_os_stride;

            const uint8_t *__restrict input_zp = nullptr;
            if (jcp.with_input_zp)
                input_zp = input_zp_base + g * jcp.ic;

            if (jcp.im2col_sz) {
                if (is_problem_3d)
                    jit_gemm_convolution_utils::im2col_dt_3d<src_data_t,
                            uint8_t>(jcp, imtr, col, od, input_zp);
                else
                    jit_gemm_convolution_utils::im2col_dt<src_data_t, uint8_t>(
                            jcp, src, imtr, col, oh, h_step, ow, w_step, input_zp);
            }

            const dim_t M = jcp.oc;
            const dim_t K = jcp.ks * jcp.ic;
            const dim_t N = h_step * w_step;
            const dim_t LDA = M * jcp.ngroups;
            const dim_t LDB = jcp.im2col_sz ? N : K * jcp.ngroups;
            const char *BT = jcp.im2col_sz ? "T" : "N";
            const int8_t off_a = 0;
            const uint8_t off_b = 0;
            const int32_t off_c = 0;
            const float onef = 1.f, zerof = 0.f;
            const src_data_t *__restrict src_od
                    = src + od * jcp.oh * jcp.ow * jcp.ngroups * jcp.ic;
            st = gemm_s8x8s32("N", BT, (jcp.signed_input || jcp.with_input_zp) ? "C" : "F", &M, &N, &K,
                    &onef, wei, &LDA, &off_a,
                    jcp.im2col_sz ? col : (uint8_t *)src_od, &LDB, &off_b,
                    &zerof, acc, &M, (jcp.signed_input || jcp.with_input_zp) ? wei_comp : &off_c);

            if (st != status::success) return st;

            const auto wei_adj_scale
                    = (wei_md.extra().flags & memory_extra_flags::scale_adjust)
                    ? wei_md.extra().scale_adjust
                    : 1.f;

            if (should_apply_zp_src_comp_outside_pp)
                apply_zp_src_comp_pad(jcp, g, od, oh, ow, h_step, w_step, acc,
                        zp.src_pad_comp);

            const single_gemm_conv_chunk_desc_t chunk_desc
                    = should_apply_zp_src_comp_pad_jit_pp
                    ? single_gemm_conv_chunk_desc_t {od, 1, oh, h_step, ow,
                            w_step}
                    : single_gemm_conv_chunk_desc_t {};

            parallel(0, [&](int ithr, int nthr) {
                dim_t _start, _end;
                balance211(N * jcp.oc, nthr, ithr, _start, _end);

                (*pp_ker_)(dst, acc, bia_base, scales, sum_scale,
                        1.f / wei_adj_scale, g, _start, _end, zp,
                        post_ops_binary_rhs_arg_vec, dst_base, ctx,
                        *pd()->dst_md(), chunk_desc);
            });
        }
        nd_iterator_step(n, MB, g, jcp.ngroups, ohb, nb_oh, owb, nb_ow);
    }

    return st;
}

template <data_type_t dst_type>
status_t _gemm_u8s8s32x_convolution_bwd_data_t<dst_type>::execute_backward_data(
        const exec_ctx_t &ctx) const {
    auto diff_dst_base = CTX_IN_MEM(const diff_dst_data_t *, DNNL_ARG_DIFF_DST);
    auto wei_base = CTX_IN_MEM(const wei_data_t *, DNNL_ARG_WEIGHTS);
    auto bia_base = CTX_IN_MEM(const char *, DNNL_ARG_BIAS);
    auto diff_src_base = CTX_OUT_MEM(diff_src_data_t *, DNNL_ARG_DIFF_SRC);

    auto MB = CTX_IN_BATCH(DNNL_ARG_DIFF_DST);

    auto scratchpad = ctx.get_scratchpad_grantor();

    const conv_gemm_conf_t &jcp = this->pd()->jcp_;

    std::atomic<status_t> st(status::success);

    parallel(jcp.nthr, [&](const int ithr, const int nthr) {
        status_t st_thr = execute_backward_data_thr(ithr, nthr, diff_dst_base,
                wei_base, bia_base, diff_src_base, scratchpad, MB);

        if (st_thr != status::success) st = st_thr;
    });

    return st;
}

template <data_type_t dst_type>
status_t
_gemm_u8s8s32x_convolution_bwd_data_t<dst_type>::execute_backward_data_thr(
        const int ithr, const int nthr, const diff_dst_data_t *diff_dst_base,
        const wei_data_t *wei_base, const char *bia_base,
        diff_src_data_t *diff_src_base,
        const memory_tracking::grantor_t &scratchpad, int MB) const {
    const conv_gemm_conf_t &jcp = this->pd()->jcp_;

    const auto diff_dst_md = memory_desc_wrapper(pd()->diff_dst_md());
    const size_t diff_dst_mb_stride = diff_dst_md.blk_off(1);
    const size_t diff_dst_g_stride = diff_dst_md.blk_off(0, 1) * jcp.oc;

    const auto wei_md = memory_desc_wrapper(pd()->weights_md(0));
    const size_t wei_g_stride = pd()->with_groups() ? wei_md.blk_off(1) : 0;

    const auto diff_src_md = memory_desc_wrapper(pd()->diff_src_md());
    const size_t diff_src_mb_stride = diff_src_md.blk_off(1);
    const size_t diff_src_g_stride = diff_src_md.blk_off(0, 1) * jcp.ic;
    const size_t diff_src_os_stride
            = diff_src_md.blocking_desc().strides[pd()->ndims() - 1];

    /* scale_idx_mult = 1 for per_oc scales and 0, otherwise */
    const int scale_idx_mult = pd()->attr()->output_scales_.mask_ == (1 << 1);
    const float *__restrict scales = pd()->attr()->output_scales_.scales_;
    const dim_t work_amount = jcp.ngroups * MB;

    acc_data_t *__restrict col = scratchpad.get<acc_data_t>(key_conv_gemm_col)
            + (ptrdiff_t)ithr * jcp.im2col_sz;
    acc_data_t *__restrict acc
            = scratchpad.get<acc_data_t>(key_conv_int_dat_in_acc_dt)
            + (ptrdiff_t)ithr * jcp.is * jcp.id * jcp.ic;

    dim_t n {0}, g {0};
    dim_t start = 0, end = 0;

    balance211(work_amount, nthr, ithr, start, end);
    nd_iterator_init(start, n, MB, g, jcp.ngroups);

    for (dim_t iwork = start; iwork < end; ++iwork) {
        const diff_dst_data_t *__restrict diff_dst = diff_dst_base
                + n * diff_dst_mb_stride + g * diff_dst_g_stride;
        const wei_data_t *__restrict wei = wei_base + g * wei_g_stride;
        diff_src_data_t *__restrict diff_src = diff_src_base
                + n * diff_src_mb_stride + g * diff_src_g_stride;

        const dim_t M = jcp.ks * jcp.ic;
        const dim_t N = jcp.os * jcp.od;
        const dim_t K = jcp.oc;
        const int8_t off_a = 0;
        const diff_dst_data_t off_b = 0;
        const int32_t off_c = 0;
        const float onef = 1.0, zerof = 0.0;
        const dim_t LD = K * jcp.ngroups;

        status_t st = gemm_s8x8s32("T", "N", "F", &M, &N, &K, &onef, wei, &LD,
                &off_a, diff_dst, &LD, &off_b, &zerof,
                jcp.im2col_sz ? col : acc, &M, &off_c);

        if (st != status::success) return st;

        if (jcp.im2col_sz)
            jit_gemm_convolution_utils::col2im_dt<int32_t>(jcp, col, acc);

        // TODO: the code below is not tested and broken anyway.
        parallel_nd(jcp.is * jcp.id, [&](dim_t is) {
            diff_src_data_t *__restrict diff_src_loc
                    = diff_src + is * diff_src_os_stride;
            const acc_data_t *__restrict acc_loc = acc + is * jcp.ic;
            const float *__restrict scales_loc
                    = scales + g * jcp.ic * scale_idx_mult;
            for (int ic = 0; ic < jcp.ic; ic++) {
                acc_data_t d = acc_loc[ic];
                if (jcp.with_bias) {
                    const float b = io::load_float_value(
                            pd()->desc()->bias_desc.data_type, bia_base,
                            g * jcp.ic + ic);
                    d += b;
                }
                d *= scales_loc[ic * scale_idx_mult];
                diff_src_loc[ic] = qz_a1b0<acc_data_t, diff_src_data_t>()(d);
            }
        });
        nd_iterator_step(n, MB, g, jcp.ngroups);
    }

    return status::success;
}

using namespace data_type;

template struct _gemm_x8s8s32x_convolution_fwd_t<u8, f32>;
template struct _gemm_x8s8s32x_convolution_fwd_t<u8, s32>;
template struct _gemm_x8s8s32x_convolution_fwd_t<u8, s8>;
template struct _gemm_x8s8s32x_convolution_fwd_t<u8, u8>;

template struct _gemm_x8s8s32x_convolution_fwd_t<s8, f32>;
template struct _gemm_x8s8s32x_convolution_fwd_t<s8, s32>;
template struct _gemm_x8s8s32x_convolution_fwd_t<s8, s8>;
template struct _gemm_x8s8s32x_convolution_fwd_t<s8, u8>;

template struct _gemm_u8s8s32x_convolution_bwd_data_t<f32>;
template struct _gemm_u8s8s32x_convolution_bwd_data_t<s32>;
template struct _gemm_u8s8s32x_convolution_bwd_data_t<s8>;
template struct _gemm_u8s8s32x_convolution_bwd_data_t<u8>;
} // namespace cpu
} // namespace impl
} // namespace dnnl
