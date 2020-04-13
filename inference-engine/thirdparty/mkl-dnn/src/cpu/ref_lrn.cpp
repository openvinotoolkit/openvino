/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
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

#include <assert.h>
#include <math.h>

#include "c_types_map.hpp"
#include "mkldnn_thread.hpp"
#include "type_helpers.hpp"

#include "bfloat16_utils.hpp"
#include "ref_lrn.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace bf16_cvt_utils;

typedef float acc_data_t;
static inline float fast_negative_powf(float omega, float beta) {
    float Y;
/*
 * Y = omega^(-3/4) =
 * = 1.0f / sqrtf(omega) * sqrtf(1.0f / sqrtf(omega))
 * = sqrtf(1.0f / sqrtf(omega)) * 1.0f / sqrtf(omega)
 * = sqrtf(1.0f / sqrtf(omega)) / sqrtf(omega)
 * = sqrtf(1.0f / sqrtf(omega) / omega)
 * = sqrtf(1.0f / (sqrtf(omega) * omega))
 */
    if (beta == 0.75f) {
        Y = sqrtf(1.0f / (sqrtf(omega) * omega));
    } else {
        Y = 1.0f / powf(omega, beta);
    }
    return Y;
};

template <impl::data_type_t data_type>
template <mkldnn_memory_format_t fmt>
void ref_lrn_fwd_t<data_type>::execute_forward() const {
    using namespace alg_kind;
    using namespace memory_format;

    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto dst = reinterpret_cast<data_t*>(this->memory(0));
    auto ws = reinterpret_cast<data_t*>(this->memory(1));

    const memory_desc_wrapper data_d(pd()->src_pd());
    const memory_desc_wrapper ws_d(pd()->workspace_pd());
    MAYBE_UNUSED(ws_d);

    const int C = pd()->C();
    const int H = pd()->H();
    const int W = pd()->W();
    const size_t stride_mb = data_d.blocking_desc().strides[0][0];
    const bool across_channels = pd()->desc()->alg_kind == lrn_across_channels;
    constexpr int blksize = fmt == nChw16c ? 16 : 8;

    auto data_off = [&](int mb, int c, int h, int w) -> size_t {
        switch (fmt) {
        case nChw16c:
        case nChw8c: return mb * stride_mb + c / blksize * H * W * blksize
                     + h * W * blksize + w * blksize + c % blksize;
        case nchw: return mb * stride_mb + c * H * W + h * W + w;
        case nhwc: return mb * stride_mb + h * W * C + w * C + c;
        default: return data_d.off(mb, c, h, w);
        }
    };

    auto ker = [=](data_t *d, int mb, int oc, int oh, int ow) {
        const acc_data_t alpha = static_cast<acc_data_t>(pd()->desc()->lrn_alpha);
        const acc_data_t beta = static_cast<acc_data_t>(pd()->desc()->lrn_beta);
        const acc_data_t k = static_cast<acc_data_t>(pd()->desc()->lrn_k);

        const int size = pd()->desc()->local_size;
        const int half_size = (size - 1) / 2;

        acc_data_t sum = 0;
        if (across_channels) {
            const int c_st = nstl::max(oc - half_size + 0, 0);
            const int c_en = nstl::min(oc + half_size + 1, C);

            for (int c = c_st; c < c_en; ++c) {
                acc_data_t s = (data_type == data_type::bf16)
                        ? cvt_bfloat16_to_float(static_cast<mkldnn_bfloat16_t>(
                                  src[data_off(mb, c, oh, ow)]))
                        : src[data_off(mb, c, oh, ow)];
                sum += s * s;
            }
        } else {
            int h_st = nstl::max(oh - half_size + 0, 0);
            int h_en = nstl::min(oh + half_size + 1, H);
            int w_st = nstl::max(ow - half_size + 0, 0);
            int w_en = nstl::min(ow + half_size + 1, W);
            for (int h = h_st; h < h_en; ++h) {
                for (int w = w_st; w < w_en; ++w) {
                    acc_data_t s = (data_type == data_type::bf16)
                            ? cvt_bfloat16_to_float(
                                      static_cast<mkldnn_bfloat16_t>(
                                              src[data_off(mb, oc, h, w)]))
                            : src[data_off(mb, oc, h, w)];
                    sum += s * s;
                }
            }
        }
        const int summands = across_channels ? size : size * size;
        sum = k + alpha * sum / summands;
        size_t off = data_off(mb, oc, oh, ow);
        if (ws) {
            if (data_type == data_type::bf16)
                ws[off] = cvt_float_to_bfloat16(sum);
            else
                ws[off] = static_cast<data_t>(sum);
        }

        if (data_type == data_type::bf16) {
            const acc_data_t s = cvt_bfloat16_to_float(
                    static_cast<mkldnn_bfloat16_t>(src[off]));
            d[0] = cvt_float_to_bfloat16(s * fast_negative_powf(sum, beta));
        } else
            d[0] = static_cast<data_t>(
                    src[off] * fast_negative_powf(sum, beta));
    };

    const int MB = pd()->MB();
    if (fmt == nChw16c || fmt == nChw8c) {
        parallel_nd(MB, utils::div_up(C, blksize), H, W,
            [&](int mb, int c_blk, int h, int w) {
            int c = c_blk * blksize;
            const size_t off = mb * stride_mb + c * H * W
                + (h * W + w) * blksize;
            PRAGMA_OMP_SIMD()
            for (int cc = 0; cc < nstl::min(blksize, C - c); ++cc)
                ker(&dst[off + cc], mb, c + cc, h, w);
        });
    } else if (fmt == nhwc) {
        parallel_nd(MB, H, W, C,
            [&](int mb, int h, int w, int c) {
            const size_t off = mb * stride_mb + h * W * C + w * C + c;
            ker(&dst[off], mb, c, h, w);
        });
    } else {
        parallel_nd(MB, C, H, W,
            [&](int mb, int c, int h, int w) {
            const size_t off = data_off(mb, c, h, w);
            ker(&dst[off], mb, c, h, w);
        });
    }
}

template <impl::data_type_t data_type>
template <mkldnn_memory_format_t fmt>
void ref_lrn_bwd_t<data_type>::execute_backward() const {
    using namespace alg_kind;
    using namespace memory_format;

    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto diff_src = reinterpret_cast<data_t*>(this->memory(0));

    const memory_desc_wrapper data_d(pd()->src_pd());
    const memory_desc_wrapper diff_data_d(pd()->diff_dst_pd());
    MAYBE_UNUSED(diff_data_d);

    const int MB = pd()->MB();
    const int C = pd()->C();
    const int H = pd()->H();
    const int W = pd()->W();
    const size_t stride_mb = data_d.blocking_desc().strides[0][0];
    constexpr int blksize = fmt == nChw16c ? 16 : 8;

    const acc_data_t alpha = static_cast<acc_data_t>(pd()->desc()->lrn_alpha);
    const acc_data_t beta = static_cast<acc_data_t>(pd()->desc()->lrn_beta);
    const acc_data_t k = static_cast<acc_data_t>(pd()->desc()->lrn_k);
    const int kernel_size = pd()->desc()->local_size;
    const int half_ksize = (kernel_size - 1) / 2;

    auto data_off = [&](int mb, int c, int h, int w) -> size_t {
        switch (fmt) {
        case nChw16c:
        case nChw8c: return mb * stride_mb + c/blksize * H * W * blksize
                     + h * W * blksize + w * blksize + c%blksize;
        case nchw: return mb * stride_mb + c * H * W + h * W + w;
        case nhwc: return mb * stride_mb + h * W * C + w * C + c;
        default: return data_d.off(mb, c, h, w);
        }
    };

    auto ker = [=](data_t *d, int mb, int oc, int oh, int ow) {
        const int c_st = nstl::max(oc - half_ksize + 0, 0);
        const int c_en = nstl::min(oc + half_ksize + 1, C);

        acc_data_t A = 0, B = 0, omega_mid = 0;
        for (int c = c_st; c < c_en; c++) {
            acc_data_t sum = 0.0;
            const int i_st = nstl::max(c - half_ksize, 0);
            const int i_en = nstl::min(c + kernel_size - half_ksize, C);

            for (int i = i_st; i < i_en; ++i) {
                acc_data_t value = (data_type == data_type::bf16)
                        ? cvt_bfloat16_to_float(static_cast<mkldnn_bfloat16_t>(
                                          src[data_off(mb, i, oh, ow)]))
                        : src[data_off(mb, i, oh, ow)];
                sum += value * value;
            }
            const acc_data_t omega
                    = static_cast<acc_data_t>(k + sum * alpha / kernel_size);
            if (c == oc)
                omega_mid = omega;
            acc_data_t sv = (data_type == data_type::bf16)
                    ? cvt_bfloat16_to_float(static_cast<mkldnn_bfloat16_t>(
                                      src[data_off(mb, c, oh, ow)]))
                    : src[data_off(mb, c, oh, ow)];
            acc_data_t t = sv * fast_negative_powf(omega, beta);
            acc_data_t ddv = (data_type == data_type::bf16)
                    ? cvt_bfloat16_to_float(static_cast<mkldnn_bfloat16_t>(
                                      diff_dst[data_off(mb, c, oh, ow)]))
                    : diff_dst[data_off(mb, c, oh, ow)];
            B += 1.0f / omega * t * ddv;
        }

        const size_t off = data_off(mb, oc, oh, ow);
        A = fast_negative_powf(omega_mid, beta) * ((data_type == data_type::bf16)
                ? cvt_bfloat16_to_float(
                          static_cast<mkldnn_bfloat16_t>(diff_dst[off]))
                : diff_dst[off]);
        B *= (data_type == data_type::bf16)
                ? cvt_bfloat16_to_float(static_cast<mkldnn_bfloat16_t>(src[off]))
                : src[off];
        B *= (2.0f * alpha * beta) / kernel_size;
        *d = (data_type == data_type::bf16)
                ? cvt_float_to_bfloat16(A - B)
                : static_cast<data_t>(A - B); // final cast down to data_t
    };

    if (fmt == nChw16c || fmt == nChw8c) {
        parallel_nd(MB, utils::div_up(C, blksize), H, W,
            [&](int mb, int c_blk, int h, int w) {
            int c = c_blk * blksize;
            const size_t off = mb * stride_mb + c * H * W +
                (h * W + w) * blksize;
            PRAGMA_OMP_SIMD()
            for (int cc = 0; cc < nstl::min(blksize, C - c); ++cc)
                ker(&diff_src[off + cc], mb, c + cc, h, w);
        });
    } else if (fmt == nhwc) {
        parallel_nd(MB, H, W, C,
            [&](int mb, int h, int w, int c) {
            const size_t off = mb * stride_mb + h * W * C + w * C + c;
            ker(&diff_src[off], mb, c, h, w);
        });
    } else {
        parallel_nd(MB, C, H, W,
            [&](int mb, int c, int h, int w) {
            const size_t off = data_off(mb, c, h, w);
            ker(&diff_src[off], mb, c, h, w);
        });
    }
}

template void ref_lrn_fwd_t<data_type::f32>::execute_forward<memory_format::nChw16c>() const;
template void ref_lrn_fwd_t<data_type::f32>::execute_forward<memory_format::nChw8c>() const;
template void ref_lrn_fwd_t<data_type::f32>::execute_forward<memory_format::nchw>() const;
template void ref_lrn_fwd_t<data_type::f32>::execute_forward<memory_format::nhwc>() const;
template void ref_lrn_fwd_t<data_type::f32>::execute_forward<memory_format::any>() const;
template void ref_lrn_bwd_t<data_type::f32>::execute_backward<memory_format::nChw16c>() const;
template void ref_lrn_bwd_t<data_type::f32>::execute_backward<memory_format::nChw8c>() const;
template void ref_lrn_bwd_t<data_type::f32>::execute_backward<memory_format::nchw>() const;
template void ref_lrn_bwd_t<data_type::f32>::execute_backward<memory_format::nhwc>() const;
template void ref_lrn_bwd_t<data_type::f32>::execute_backward<memory_format::any>() const;

template void
ref_lrn_fwd_t<data_type::bf16>::execute_forward<memory_format::nChw16c>() const;
template void
ref_lrn_fwd_t<data_type::bf16>::execute_forward<memory_format::nChw8c>() const;
template void
ref_lrn_fwd_t<data_type::bf16>::execute_forward<memory_format::nchw>() const;
template void
ref_lrn_fwd_t<data_type::bf16>::execute_forward<memory_format::nhwc>() const;
template void
ref_lrn_fwd_t<data_type::bf16>::execute_forward<memory_format::any>() const;
template void
ref_lrn_bwd_t<data_type::bf16>::execute_backward<memory_format::nChw16c>() const;
template void
ref_lrn_bwd_t<data_type::bf16>::execute_backward<memory_format::nChw8c>() const;
template void
ref_lrn_bwd_t<data_type::bf16>::execute_backward<memory_format::nchw>() const;
template void
ref_lrn_bwd_t<data_type::bf16>::execute_backward<memory_format::nhwc>() const;
template void
ref_lrn_bwd_t<data_type::bf16>::execute_backward<memory_format::any>() const;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
