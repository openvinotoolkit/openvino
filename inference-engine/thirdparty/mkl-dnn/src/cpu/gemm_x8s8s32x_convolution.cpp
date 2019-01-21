/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

#include "mkldnn_types.h"

#include "c_types_map.hpp"
#include "utils.hpp"
#include "type_helpers.hpp"
#include "mkldnn_thread.hpp"
#include "math_utils.hpp"

#include "simple_q10n.hpp"

#include "gemm_x8s8s32x_convolution.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::utils;
using namespace mkldnn::impl::math;

template <bool with_relu, data_type_t src_type, data_type_t dst_type>
void _gemm_x8s8s32x_convolution_fwd_t<with_relu, src_type,
        dst_type>::execute_forward() {
    auto src_base = reinterpret_cast<const src_data_t *>(this->input_memory(0));
    auto wei_base = reinterpret_cast<const wei_data_t *>(this->input_memory(1));
    auto bia_base = reinterpret_cast<const char *>(this->input_memory(2));
    auto dst_base = reinterpret_cast<dst_data_t *>(this->memory());

    jit_gemm_conv_conf_t &jcp = this->conf_.jcp_;

    char *scratchpad = (char *)this->scratchpad_->get();
    uint8_t *col = (uint8_t *)scratchpad;
    parallel_nd(jcp.im2col_sz * jcp.nthr, [&](ptrdiff_t i) {
        col[i] = jcp.signed_input ? (uint8_t)128 : (uint8_t)0;
    });

    parallel(jcp.nthr, [&](const int ithr, const int nthr) {
        execute_forward_thr(ithr, nthr, src_base, wei_base, bia_base,
                dst_base, scratchpad);
    });
}

template <bool with_relu, data_type_t src_type, data_type_t dst_type>
void _gemm_x8s8s32x_convolution_fwd_t<with_relu, src_type,
        dst_type>::execute_forward_thr(const int ithr, const int nthr,
        const src_data_t *src_base, const wei_data_t *wei_base,
        const char *bia_base, dst_data_t *dst_base, char *scratchpad) {
#if USE_MKL_IGEMM
    jit_gemm_conv_conf_t &jcp = this->conf_.jcp_;

    const auto src_md = memory_desc_wrapper(conf_.src_pd());
    const size_t src_mb_stride = src_md.blk_off(1);
    const size_t src_g_stride = src_md.blk_off(0, 1) * jcp.ic;

    const auto wei_md = memory_desc_wrapper(conf_.weights_pd(0));
    const size_t wei_g_stride = conf_.with_groups() ? wei_md.blk_off(1) : 0;

    const auto dst_md = memory_desc_wrapper(conf_.dst_pd());
    const size_t dst_mb_stride = dst_md.blk_off(1);
    const size_t dst_g_stride = dst_md.blk_off(0, 1) * jcp.oc;
    const size_t dst_os_stride = dst_md.blk_off(0, 0, 0, 1);

    auto get_bias = [=, &bia_base](size_t off) -> acc_data_t {
#       define CASE(dt) case dt: return (acc_data_t)\
        (*((const prec_traits<dt>::type *)bia_base + off))
        switch (conf_.cdesc()->bias_desc.data_type) {
        CASE(data_type::s8);
        CASE(data_type::u8);
        CASE(data_type::s32);
        CASE(data_type::f32);
        default: assert(!"unimplemented");
        }
#       undef CASE
        return 0;
    };

    /* scale_idx_mult = 1 for per_oc scales and 0, otherwise */
    const int scale_idx_mult = conf_.attr()->output_scales_.mask_ == (1 << 1);
    const float *scales = conf_.attr()->output_scales_.scales_;

    const auto rmode = conf_.attr()->round_mode_;

    const bool use_fast_path = true
        && scale_idx_mult == 0
        && jcp.ngroups == 1
        && !jcp.with_bias;
    const float fast_path_alpha = scales[0] / jcp.wei_adj_scale;

    const auto &post_ops = conf_.attr()->post_ops_;
    const bool do_sum = post_ops.contain(primitive_kind::sum, 0);
    const float sum_scale = do_sum ? post_ops.entry_[0].sum.scale : 0;

    float nslope = jcp.with_relu ? jcp.relu_negative_slope : 0;
    int entry_idx = -1;
    for (int idx = 0; idx < post_ops.len_; ++idx) {
        const auto &e = post_ops.entry_[idx];
        if (e.is_relu(true, false)) {
            entry_idx = idx;
            nslope = e.eltwise.alpha;
            break;
        }
    }
    const bool do_relu = jcp.with_relu || (entry_idx >= 0);

    uint8_t *_col = (uint8_t *)scratchpad;
    ptrdiff_t offset = (ptrdiff_t)jcp.im2col_sz * sizeof(uint8_t) * jcp.nthr;
    acc_data_t *_acc = (acc_data_t *)(scratchpad + offset);

    uint8_t *col = _col + (ptrdiff_t)ithr * jcp.im2col_sz;
    acc_data_t *acc = _acc + (ptrdiff_t)ithr * jcp.os * jcp.oc;

    offset = (ptrdiff_t)jcp.ngroups * jcp.ks * jcp.ic * jcp.oc;
    const int32_t *_wei_comp = (const int32_t *)(wei_base + offset);

    int n{0}, g{0};
    size_t start = 0, end = 0;

    const size_t work_amount = jcp.ngroups * jcp.mb;
    balance211(work_amount, nthr, ithr, start, end);
    nd_iterator_init(start, n, jcp.mb, g, jcp.ngroups);

    for (size_t iwork = start; iwork < end; ++iwork) {
        const src_data_t *src = src_base + n * src_mb_stride
            + g * src_g_stride;
        const wei_data_t *wei = wei_base + g * wei_g_stride;
        dst_data_t *dst = dst_base + n * dst_mb_stride + g * dst_g_stride;
        const int32_t *wei_comp = _wei_comp + g * jcp.oc;

        if (jcp.im2col_sz)
            jit_gemm_convolution_utils::im2col_u8<src_data_t>(jcp, src, col);

        const int M = jcp.oc;
        const int K = jcp.ks * jcp.ic;
        const int N = jcp.os;
        const CBLAS_OFFSET offsetc
                = jcp.signed_input ? CblasColOffset : CblasFixOffset;
        const int8_t off_a = 0, off_b = 0;
        const int32_t off_c = 0;

        cblas_gemm_s8u8s32(CblasColMajor, CblasNoTrans, CblasNoTrans, offsetc,
                M, N, K, 1.0f, wei, M * jcp.ngroups, off_a,
                jcp.im2col_sz ? col : (uint8_t *)src, K, off_b, 0.0f, acc, M,
                jcp.signed_input ? wei_comp : &off_c);

        if (use_fast_path) {
            auto body = [&](int o) {
                float d = fast_path_alpha * acc[o] + sum_scale * dst[o];
                if (do_relu && d < 0) d *= nslope;
                dst[o] = qz_a1b0<float, dst_data_t>()(d, rmode);
            };

#           if MKLDNN_THR == MKLDNN_THR_OMP && _OPENMP >= 201307
#           pragma omp parallel for simd
            for (int o = 0; o < jcp.os * jcp.oc; ++o) body(o);
#           else
            parallel_nd(jcp.os * jcp.oc, body);
#           endif
        } else {
            parallel_nd(jcp.os, jcp.oc, [&](const int os, const int oc) {
                const size_t acc_off = os * jcp.oc + oc;
                float d = (float)acc[acc_off];
                if (jcp.signed_input)
                    d /= jcp.wei_adj_scale;

                if (jcp.with_bias)
                    d += get_bias(g * jcp.oc + oc);

                d *= scales[(g * jcp.oc + oc) * scale_idx_mult];

                const size_t dst_off = os * dst_os_stride + oc;
                if (do_sum) d += sum_scale * dst[dst_off];
                if (do_relu && d < 0) d *= nslope;
                dst[dst_off] = qz_a1b0<float, dst_data_t>()(d, rmode);
            });
        }
        nd_iterator_step(n, jcp.mb, g, jcp.ngroups);
    }
#endif
}

template <data_type_t dst_type>
void _gemm_u8s8s32x_convolution_bwd_data_t<dst_type>::execute_backward_data() {
    auto diff_dst_base = reinterpret_cast<const diff_dst_data_t *>
            (this->input_memory(0));
    auto wei_base = reinterpret_cast<const wei_data_t *>(this->input_memory(1));
    auto bia_base = reinterpret_cast<const char *>(this->input_memory(2));
    auto diff_src_base = reinterpret_cast<diff_src_data_t *>(this->memory());

    jit_gemm_conv_conf_t &jcp = this->conf_.jcp_;
    char *scratchpad = (char *)this->scratchpad_->get();

    parallel(jcp.nthr, [&](const int ithr, const int nthr) {
        execute_backward_data_thr(ithr, nthr, diff_dst_base, wei_base,
                bia_base, diff_src_base, scratchpad);
    });
}

template <data_type_t dst_type>
void _gemm_u8s8s32x_convolution_bwd_data_t<dst_type>
::execute_backward_data_thr(const int ithr, const int nthr,
        const diff_dst_data_t *diff_dst_base, const wei_data_t *wei_base,
        const char *bia_base, diff_src_data_t *diff_src_base, char *scratchpad)
{
#if USE_MKL_IGEMM
    jit_gemm_conv_conf_t &jcp = this->conf_.jcp_;

    const auto diff_dst_md = memory_desc_wrapper(conf_.diff_dst_pd());
    const size_t diff_dst_mb_stride = diff_dst_md.blk_off(1);
    const size_t diff_dst_g_stride = diff_dst_md.blk_off(0, 1) * jcp.oc;

    const auto wei_md = memory_desc_wrapper(conf_.weights_pd(0));
    const size_t wei_g_stride = conf_.with_groups() ? wei_md.blk_off(1) : 0;

    const auto diff_src_md = memory_desc_wrapper(conf_.diff_src_pd());
    const size_t diff_src_mb_stride = diff_src_md.blk_off(1);
    const size_t diff_src_g_stride = diff_src_md.blk_off(0, 1) * jcp.ic;
    const size_t diff_src_os_stride = diff_src_md.blk_off(0, 0, 0, 1);

    auto get_bias = [=, &bia_base](size_t off) -> acc_data_t {
#       define CASE(dt) case dt: return (acc_data_t)\
        (*((const prec_traits<dt>::type *)bia_base + off))
        switch (conf_.desc()->bias_desc.data_type) {
        CASE(data_type::s8);
        CASE(data_type::u8);
        CASE(data_type::s32);
        CASE(data_type::f32);
        default: assert(!"unimplemented");
        }
#       undef CASE
        return 0;
    };

    /* scale_idx_mult = 1 for per_oc scales and 0, otherwise */
    const int scale_idx_mult = conf_.attr()->output_scales_.mask_ == (1 << 1);
    const float *scales = conf_.attr()->output_scales_.scales_;
    const auto rmode = conf_.attr()->round_mode_;
    const size_t work_amount = jcp.ngroups * jcp.mb;

    acc_data_t *_col = (acc_data_t *)scratchpad;
    ptrdiff_t offset = (ptrdiff_t)jcp.im2col_sz
                                    * sizeof(acc_data_t) * jcp.nthr;
    acc_data_t *_acc = (acc_data_t *)(scratchpad + offset);

    acc_data_t *col = _col + (ptrdiff_t)ithr * jcp.im2col_sz;
    acc_data_t *acc = _acc + (ptrdiff_t)ithr * jcp.is * jcp.ic;

    int n{0}, g{0};
    size_t start = 0, end = 0;

    balance211(work_amount, nthr, ithr, start, end);
    nd_iterator_init(start, n, jcp.mb, g, jcp.ngroups);

    for (size_t iwork = start; iwork < end; ++iwork) {
        const diff_dst_data_t *diff_dst = diff_dst_base
            + n * diff_dst_mb_stride + g * diff_dst_g_stride;
        const wei_data_t *wei = wei_base + g * wei_g_stride;
        diff_src_data_t *diff_src = diff_src_base + n * diff_src_mb_stride
            + g * diff_src_g_stride;

        const int M = jcp.ks * jcp.ic;
        const int N = jcp.os;
        const int K = jcp.oc;
        const int8_t off_a = 0, off_b = 0;
        const int32_t off_c = 0;

        cblas_gemm_s8u8s32(CblasColMajor, CblasTrans, CblasNoTrans,
                CblasFixOffset, M, N, K, 1., wei, K * jcp.ngroups, off_a,
                diff_dst, K * jcp.ngroups, off_b, 0., jcp.im2col_sz ? col
                : acc, M, &off_c);

        if (jcp.im2col_sz)
            jit_gemm_convolution_utils::col2im_s32(jcp, col, acc);

        parallel_nd(jcp.is, jcp.ic, [&](int is, int ic) {
            float d = (float)acc[is * jcp.ic + ic];
            if (jcp.with_bias)
                d += get_bias(g * jcp.ic + ic);
            d *= scales[(g * jcp.ic + ic) * scale_idx_mult];
            const size_t diff_src_off = is * diff_src_os_stride + ic;
            diff_src[diff_src_off] =
                qz_a1b0<float, diff_src_data_t>()(d, rmode);
        });
        nd_iterator_step(n, jcp.mb, g, jcp.ngroups);
    }
#endif
}

using namespace data_type;

template struct _gemm_x8s8s32x_convolution_fwd_t<true, u8, f32>;
template struct _gemm_x8s8s32x_convolution_fwd_t<true, u8, s32>;
template struct _gemm_x8s8s32x_convolution_fwd_t<true, u8, s8>;
template struct _gemm_x8s8s32x_convolution_fwd_t<true, u8, u8>;
template struct _gemm_x8s8s32x_convolution_fwd_t<false, u8, f32>;
template struct _gemm_x8s8s32x_convolution_fwd_t<false, u8, s32>;
template struct _gemm_x8s8s32x_convolution_fwd_t<false, u8, s8>;
template struct _gemm_x8s8s32x_convolution_fwd_t<false, u8, u8>;

template struct _gemm_x8s8s32x_convolution_fwd_t<true, s8, f32>;
template struct _gemm_x8s8s32x_convolution_fwd_t<true, s8, s32>;
template struct _gemm_x8s8s32x_convolution_fwd_t<true, s8, s8>;
template struct _gemm_x8s8s32x_convolution_fwd_t<true, s8, u8>;
template struct _gemm_x8s8s32x_convolution_fwd_t<false, s8, f32>;
template struct _gemm_x8s8s32x_convolution_fwd_t<false, s8, s32>;
template struct _gemm_x8s8s32x_convolution_fwd_t<false, s8, s8>;
template struct _gemm_x8s8s32x_convolution_fwd_t<false, s8, u8>;

template struct _gemm_u8s8s32x_convolution_bwd_data_t<f32>;
template struct _gemm_u8s8s32x_convolution_bwd_data_t<s32>;
template struct _gemm_u8s8s32x_convolution_bwd_data_t<s8>;
template struct _gemm_u8s8s32x_convolution_bwd_data_t<u8>;
}
}
}
