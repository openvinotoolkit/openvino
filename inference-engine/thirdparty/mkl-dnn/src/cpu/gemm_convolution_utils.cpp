/*******************************************************************************
* Copyright 2016-2019 Intel Corporation
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
#include "type_helpers.hpp"
#include "mkldnn_thread.hpp"
#include "utils.hpp"
#include "cpu_isa_traits.hpp"

#include "gemm_convolution_utils.hpp"
#include "jit_generator.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;
using namespace prop_kind;
using namespace data_type;

namespace jit_gemm_convolution_utils {

template <typename data_type_t>
void im2col_3d(const jit_gemm_conv_conf_t &jcp, const data_type_t *im,
        data_type_t *col, int od)
{
    const size_t OHW = jcp.oh * jcp.ow;
    const size_t im_step = jcp.ih * jcp.iw * jcp.id;
    const size_t col_step = jcp.ks * OHW;

    parallel_nd(jcp.ic, [&](int ic) {
        const data_type_t *__restrict im_loc = im + ic * im_step;
        data_type_t *__restrict col_loc = col + ic * col_step;
        int id = od * jcp.stride_d - jcp.f_pad;
        for (int kd = 0; kd < jcp.kd; ++kd) {
            data_type_t *__restrict col_ = col_loc + kd * jcp.kh * jcp.kw * OHW;
            if (id < 0 || id >= jcp.id) {
                int ih_ = -jcp.t_pad;
                for (int kh = 0; kh < jcp.kh; ++kh) {
                    int ih = ih_;
                    for (int oh = 0; oh < jcp.oh; ++oh) {
                        if (ih < 0 || ih >= jcp.ih) {
                            ih += jcp.stride_h;
                            continue;
                        }
                        int iw_ = -jcp.l_pad;
                        for (int kw = 0; kw < jcp.kw; ++kw) {
                            int iw = iw_;
                            for (int ow = 0; ow < jcp.ow; ++ow) {
                                if (iw < 0 || iw >= jcp.iw) {
                                    iw += jcp.stride_w;
                                    continue;
                                }

                                const size_t col_idx = kw * OHW + oh * jcp.ow
                                    + ow;

                                col_[col_idx] = 0;
                                iw += jcp.stride_w;
                            }
                            iw_ += (1 + jcp.dilate_w);
                        }
                        ih += jcp.stride_h;
                    }
                    ih_ += (1 + jcp.dilate_h);
                    col_ += jcp.kw * OHW;
                }
            } else {
                const data_type_t *__restrict im_ =
                    im_loc + id * jcp.ih * jcp.iw;
                int ih_ = -jcp.t_pad;
                for (int kh = 0; kh < jcp.kh; ++kh) {
                    int ih = ih_;
                    for (int oh = 0; oh < jcp.oh; ++oh) {
                        if (ih < 0 || ih >= jcp.ih) {
                            ih += jcp.stride_h;
                            continue;
                        }
                        int iw_ = -jcp.l_pad;
                        for (int kw = 0; kw < jcp.kw; ++kw) {
                            int iw = iw_;
                            for (int ow = 0; ow < jcp.ow; ++ow) {
                                if (iw < 0 || iw >= jcp.iw) {
                                    iw += jcp.stride_w;
                                    continue;
                                }

                                const size_t col_idx = kw * OHW + oh * jcp.ow
                                    + ow;
                                const size_t im_idx = ih * jcp.iw + iw;

                                col_[col_idx] = im_[im_idx];
                                iw += jcp.stride_w;
                            }
                            iw_ += (1 + jcp.dilate_w);
                        }
                        ih += jcp.stride_h;
                    }
                    ih_ += (1 + jcp.dilate_h);
                    col_ += jcp.kw * OHW;
                }
            }
            id += (1 + jcp.dilate_d);
        }
    });
}

template
void im2col_3d(const jit_gemm_conv_conf_t &jcp, const float *im, float *col,
        int od);

template
void im2col_3d(const jit_gemm_conv_conf_t &jcp, const mkldnn_bfloat16_t *im,
         mkldnn_bfloat16_t *col, int od);

inline int saturate(int low, int upper, int value) {
    return nstl::max(low, nstl::min(upper, value));
}

/* col[ic][kh][kw][oh][ow] <-- im2col(im[ic][ih][iw]) */
template <typename data_type_t>
void im2col(const jit_gemm_conv_conf_t &jcp,
        const data_type_t *__restrict im, data_type_t *__restrict col, int ss, int sb,
        int cs, int cb) {
    const size_t im_step = jcp.is;
    const size_t col_step = jcp.ks * sb;
    const int dh = 1 + jcp.dilate_h;
    const int dw = 1 + jcp.dilate_w;
    const int sh = jcp.stride_h;
    const int sw = jcp.stride_w;
    const int tp = jcp.t_pad;
    const int lp = jcp.l_pad;
    const int first_oh = ss / jcp.ow;
    const int last_oh = (ss + sb - 1) / jcp.ow;
    const int oh_begin = first_oh;
    const int oh_end = last_oh + 1;
    const int first_ow = ss % jcp.ow;
    const int last_ow = (ss + sb - 1) % jcp.ow;

    if (jcp.outer_threading) {
        if (sw == 1) {
            // Generated code is more optimized for stride_w == 1
            // because innermost loop is by width
            for (int ic = 0; ic < cb; ic++) {
                const data_type_t *__restrict im_ic = im + (ic + cs) * im_step;
                for (int kh = 0; kh < jcp.kh; kh++) {
                    for (int kw = 0; kw < jcp.kw; kw++) {
                        data_type_t *__restrict col_k
                                = col + ic * col_step + (kh * jcp.kw + kw) * sb;
                        for (int oh = oh_begin; oh < oh_end; oh++) {
                            const int ih = oh * sh - tp + kh * dh;
                            const data_type_t *__restrict im_
                                    = im_ic + ih * jcp.iw - lp + kw * dw;
                            const int ow_begin
                                    = (oh == first_oh) ? first_ow : 0;
                            const int ow_end
                                    = (oh == last_oh) ? (last_ow + 1) : jcp.ow;
                            data_type_t *__restrict col_ = col_k + oh * jcp.ow - ss;
                            if (ih < 0 || ih >= jcp.ih)
                                for (int ow = ow_begin; ow < ow_end; ow++)
                                    col_[ow] = (data_type_t)0;
                            else {
                                for (int ow = ow_begin; ow < ow_end; ++ow) {
                                    const int iw = ow;
                                    if (iw < lp - kw * dw || iw >= jcp.iw + lp - kw * dw)
                                        col_[ow] = (data_type_t)0;
                                    else
                                        col_[ow] = im_[iw];
                                }
                            }
                        }
                    }
                }
            }
        } else {
            for (int ic = 0; ic < cb; ic++) {
                const data_type_t *__restrict im_ = im + (ic + cs) * im_step;
                for (int kh = 0; kh < jcp.kh; kh++) {
                    for (int kw = 0; kw < jcp.kw; kw++) {
                        data_type_t *__restrict col_k
                                = col + ic * col_step + (kh * jcp.kw + kw) * sb;
                        for (int oh = oh_begin; oh < oh_end; oh++) {
                            const int ih = oh * sh - tp + kh * dh;
                            const int ow_begin
                                    = (oh == first_oh) ? first_ow : 0;
                            const int ow_end
                                    = (oh == last_oh) ? (last_ow + 1) : jcp.ow;
                            data_type_t *__restrict col_oh = col_k + oh * jcp.ow - ss;
                            if (ih < 0 || ih >= jcp.ih)
                                for (int ow = ow_begin; ow < ow_end; ow++)
                                    col_oh[ow] = (data_type_t)0;
                            else
                                for (int ow = ow_begin; ow < ow_end; ow++) {
                                    const int iw = ow * sw - lp + kw * dw;
                                    if (iw < 0 || iw >= jcp.iw)
                                        col_oh[ow] = (data_type_t)0;
                                    else {
                                        const ptrdiff_t im_idx
                                                = ih * jcp.iw + iw;
                                        col_oh[ow] = im_[im_idx];
                                    }
                                }
                        }
                    }
                }
            }
        }
    } else {
        // TODO: optimize threading if jcp.ic*jcp.kh*jcp.kw*oh_range is small
        // comparing to number of threads
        const int oh_range = oh_end - oh_begin;
        // Generated code is more optimized for stride_w == 1
        // because innermost loop is by width
        if (sw == 1)
            parallel_nd(cb, jcp.kh, jcp.kw, oh_range,
                    [&](int ic, int kh, int kw, int ohr) {
                        const int oh = ohr + oh_begin;
                        const int ih = oh * sh - tp + kh * dh;
                        const int ow_start = (oh == first_oh) ? first_ow : 0;
                        const int ow_end
                                = (oh == last_oh) ? (last_ow + 1) : jcp.ow;
                        data_type_t *__restrict col_oh = col + ic * col_step
                                + (kh * jcp.kw + kw) * sb + oh * jcp.ow - ss;
                        const data_type_t *__restrict im_ = im + (ic + cs) * im_step + ih * jcp.iw;
                        const int iw_shift = kw * dw - lp;
                        if (ih < 0 || ih >= jcp.ih)
                            for (int ow = ow_start; ow < ow_end; ow++)
                                col_oh[ow] = (data_type_t)0;
                        else
                            for (int ow = ow_start; ow < ow_end; ow++) {
                                const int iw = ow + iw_shift;
                                if (iw < 0 || iw >= jcp.iw)
                                    col_oh[ow] = (data_type_t)0;
                                else
                                    col_oh[ow] = im_[iw];
                            }
                    });
        else
            parallel_nd(cb, jcp.kh, jcp.kw, oh_range,
                    [&](int ic, int kh, int kw, int ohr) {
                        const int oh = ohr + oh_begin;
                        const int ih = oh * sh - tp + kh * dh;
                        const int ow_start = (oh == first_oh) ? first_ow : 0;
                        const int ow_end
                                = (oh == last_oh) ? (last_ow + 1) : jcp.ow;
                        data_type_t *__restrict col_oh = col + ic * col_step
                                + (kh * jcp.kw + kw) * sb + oh * jcp.ow - ss;
                        const data_type_t *__restrict im_ = im + (ic + cs) * im_step;
                        if (ih < 0 || ih >= jcp.ih)
                            for (int ow = ow_start; ow < ow_end; ow++)
                                col_oh[ow] = (data_type_t)0;
                        else
                            for (int ow = ow_start; ow < ow_end; ow++) {
                                const int iw = ow * sw - lp + kw * dw;
                                if (iw < 0 || iw >= jcp.iw)
                                    col_oh[ow] = (data_type_t)0;
                                else {
                                    const ptrdiff_t im_idx = ih * jcp.iw + iw;
                                    col_oh[ow] = im_[im_idx];
                                }
                            }
                    });
    }
}

template
void im2col(const jit_gemm_conv_conf_t &jcp, const float *__restrict im,
       float *__restrict col, int hs, int hb, int ws, int wb);

template
void im2col(const jit_gemm_conv_conf_t &jcp,
       const mkldnn_bfloat16_t *__restrict im,
       mkldnn_bfloat16_t *__restrict col, int hs, int hb, int ws, int wb);

template <typename T, bool with_input_zp, bool with_weights_zp>
void im2col_u8_compute(const jit_gemm_conv_conf_t &jcp, const T *__restrict im,
        T *__restrict imtr, uint8_t *__restrict col, int hs, int hb, int ws,
        int wb, const uint8_t *__restrict input_zp, int32_t *__restrict weights_zp_compensation) {
   uint8_t shift = jcp.signed_input ? 128 : 0;

    const int dh = 1 + jcp.dilate_h;
    const int dw = 1 + jcp.dilate_w;
    const int sh = jcp.stride_h;
    const int sw = jcp.stride_w;
    const int im_iw_stride = jcp.ic * jcp.ngroups;
    const int im_ih_stride = jcp.iw * im_iw_stride;
    const int tp = jcp.t_pad;
    const int lp = jcp.l_pad;

    if (with_weights_zp) {
        for (int oh = 0; oh < hb; oh++) {
            utils::array_set(weights_zp_compensation + oh * wb, 0, wb);
        }
    }

    if (jcp.im2col_sz) {
        if (jcp.outer_threading && sh == 1 && sw == 1 && dh == 1 && dw == 1) {
            /* im[ih][iw][ic] --> imtr[ic][ih][iw] --> col[kh][kw][ic][oh][ow] */
            const int hp = hs - tp;
            const int wp = ws - lp;
            const int ih_start = saturate(0, jcp.ih, hp);
            const int ih_end = saturate(0, jcp.ih, hp + hb + jcp.kh);
            const int iw_start = saturate(0, jcp.iw, wp);
            const int iw_end = saturate(0, jcp.iw, wp + wb + jcp.kw);

            const int ihb = ih_end - ih_start;
            const int iwb = iw_end - iw_start;

            const int imtr_ic_stride = ihb * iwb;
            const ptrdiff_t imtr_idx_shift = ih_start * iwb + iw_start;
            for (int ic = 0; ic < jcp.ic; ic++) {
                const ptrdiff_t imtr_idx_ic = ic * imtr_ic_stride - imtr_idx_shift;
                for (int ih = ih_start; ih < ih_end; ih++) {
                    const ptrdiff_t im_idx_ih = ic + ih * im_ih_stride;
                    const ptrdiff_t imtr_idx_ih = imtr_idx_ic + ih * iwb;
                    for (int iw = iw_start; iw < iw_end; iw++)
                        imtr[imtr_idx_ih + iw] = im[im_idx_ih + iw * im_iw_stride];
                }
            }

            const int col_ic_str = hb * wb;
            const int col_kw_stride = jcp.ic * col_ic_str;
            const int col_kh_stride = jcp.kw * col_kw_stride;

            const int oh_init = ih_start - hp;
            const int ow_init = iw_start - wp;

            for (int kh = 0; kh < jcp.kh; kh++) {
                const ptrdiff_t col_idx_kh = kh * col_kh_stride;
                const int oh_kh = oh_init - kh;
                const int oh_start = saturate(0, hb, oh_kh);
                const int oh_end = saturate(0, hb, oh_kh + ihb);
                for (int kw = 0; kw < jcp.kw; kw++) {
                    const ptrdiff_t col_idx_kw
                            = col_idx_kh + kw * jcp.ic * col_ic_str;
                    const int ow_kw = ow_init - kw;
                    const int imtr_shift = oh_kh * iwb + ow_kw;
                    const int ow_start = saturate(0, wb, ow_kw);
                    const int ow_end = saturate(0, wb, ow_kw + iwb);
                    for (int ic = 0; ic < jcp.ic; ic++) {
                        uint8_t izp = with_input_zp ? input_zp[ic] : (uint8_t) 0;
                        const ptrdiff_t col_idx_ic = col_idx_kw + ic * col_ic_str;
                        const int imtr_idx_ic = ic * imtr_ic_stride - imtr_shift;
                        for (int oh = 0; oh < oh_start; oh++) {
                            const ptrdiff_t col_idx_oh = col_idx_ic + oh * wb;
                            for (int ow = 0; ow < wb; ++ow) {
                                if (with_input_zp)
                                    col[col_idx_oh + ow] = izp;
                                else
                                    col[col_idx_oh + ow] = shift;

                                if (with_weights_zp)
                                    weights_zp_compensation[oh * wb + ow] += izp;
                            }
                        }
                        for (int oh = oh_start; oh < oh_end; oh++) {
                            const ptrdiff_t col_idx_oh = col_idx_ic + oh * wb;
                            const ptrdiff_t imtr_idx_oh = imtr_idx_ic + oh * iwb;
                            for (int ow = 0; ow < ow_start; ++ow) {
                                if (with_input_zp)
                                    col[col_idx_oh + ow] = izp;
                                else
                                    col[col_idx_oh + ow] = shift;

                                if (with_weights_zp)
                                    weights_zp_compensation[oh * wb + ow] += izp;
                            }
                            for (int ow = ow_start; ow < ow_end; ++ow) {
                                if (with_input_zp)
                                    col[col_idx_oh + ow] = imtr[imtr_idx_oh + ow];
                                else
                                    col[col_idx_oh + ow] = imtr[imtr_idx_oh + ow] + shift;

                                if (with_weights_zp)
                                    weights_zp_compensation[oh * wb + ow] += imtr[imtr_idx_oh + ow];
                            }
                            for (int ow = ow_end; ow < wb; ++ow) {
                                if (with_input_zp)
                                    col[col_idx_oh + ow] = izp;
                                else
                                    col[col_idx_oh + ow] = shift;

                                if (with_weights_zp)
                                    weights_zp_compensation[oh * wb + ow] += izp;
                            }
                        }
                        for (int oh = oh_end; oh < hb; oh++) {
                            const ptrdiff_t col_idx_oh = col_idx_ic + oh * wb;
                            for (int ow = 0; ow < wb; ++ow) {
                                if (with_input_zp)
                                    col[col_idx_oh + ow] = izp;
                                else
                                    col[col_idx_oh + ow] = shift;

                                if (with_weights_zp)
                                    weights_zp_compensation[oh * wb + ow] += izp;
                            }
                        }
                    }
                }
            }
        } else {
            if (with_weights_zp) {
                parallel_nd(hb, [&](int oh) {
                    for (int kh = 0; kh < jcp.kh; kh++) {
                        for (int kw = 0; kw < jcp.kw; kw++) {
                            for (int ic = 0; ic < jcp.ic; ic++) {
                                uint8_t izp = with_input_zp ? input_zp[ic] : (uint8_t) 0;
                                const int hp = tp - kh * dh;
                                const int ih = (oh + hs) * sh - hp;
                                const ptrdiff_t col_idx_base = (((kh * jcp.kw + kw) * jcp.ic + ic) * hb + oh) * wb;
                                if (ih < 0 || ih >= jcp.ih) {
                                    for (int ow = 0; ow < wb; ow++) {
                                        if (jcp.with_input_zp)
                                            col[col_idx_base + ow] = izp;
                                        else
                                            col[col_idx_base + ow] = shift;

                                        weights_zp_compensation[oh * wb + ow] += izp;
                                    }
                                } else {
                                    const int wp = lp - kw * dw;
                                    const int ow_start = saturate(0, wb, div_up(wp, sw) - ws);
                                    const int ow_end = saturate(0, wb, div_up(jcp.iw + wp, sw) - ws);
                                    for (int ow = 0; ow < ow_start; ow++) {
                                        if (jcp.with_input_zp)
                                            col[col_idx_base + ow] = izp;
                                        else
                                            col[col_idx_base + ow] = shift;

                                        weights_zp_compensation[oh * wb + ow] += izp;
                                    }

                                    const int iw_base = ws * sw - wp;
                                    const ptrdiff_t im_idx_base = ih * im_ih_stride + ic;
                                    for (int ow = ow_start; ow < ow_end; ow++) {
                                        const int iw = iw_base + ow * sw;
                                        const ptrdiff_t im_idx = im_idx_base + iw * im_iw_stride;
                                        if (jcp.with_input_zp)
                                            col[col_idx_base + ow] = im[im_idx];
                                        else
                                            col[col_idx_base + ow] = im[im_idx] + shift;

                                        weights_zp_compensation[oh * wb + ow] += im[im_idx];
                                    }
                                    for (int ow = ow_end; ow < wb; ow++) {
                                        if (jcp.with_input_zp)
                                            col[col_idx_base + ow] = izp;
                                        else
                                            col[col_idx_base + ow] = shift;

                                        weights_zp_compensation[oh * wb + ow] += izp;
                                    }
                                }
                            }
                        }
                    }
                });
            } else {
                parallel_nd(jcp.kh, jcp.kw, jcp.ic, hb,
                [&](int kh, int kw, int ic, int oh) {
                    uint8_t izp = with_input_zp ? input_zp[ic] : (uint8_t) 0;
                    const int hp = tp - kh * dh;
                    const int ih = (oh + hs) * sh - hp;
                    const ptrdiff_t col_idx_base = (((kh * jcp.kw + kw) * jcp.ic + ic) * hb + oh) * wb;
                    if (ih < 0 || ih >= jcp.ih) {
                        for (int ow = 0; ow < wb; ow++) {
                            if (jcp.with_input_zp)
                                col[col_idx_base + ow] = izp;
                            else
                                col[col_idx_base + ow] = shift;
                        }
                    } else {
                        const int wp = lp - kw * dw;
                        const int ow_start = saturate(0, wb, div_up(wp, sw) - ws);
                        const int ow_end = saturate(0, wb, div_up(jcp.iw + wp, sw) - ws);
                        for (int ow = 0; ow < ow_start; ow++) {
                            if (jcp.with_input_zp)
                                col[col_idx_base + ow] = izp;
                            else
                                col[col_idx_base + ow] = shift;
                        }

                        const int iw_base = ws * sw - wp;
                        const ptrdiff_t im_idx_base = ih * im_ih_stride + ic;
                        for (int ow = ow_start; ow < ow_end; ow++) {
                            const int iw = iw_base + ow * sw;
                            const ptrdiff_t im_idx = im_idx_base + iw * im_iw_stride;
                            if (jcp.with_input_zp)
                                col[col_idx_base + ow] = im[im_idx];
                            else
                                col[col_idx_base + ow] = im[im_idx] + shift;
                        }
                        for (int ow = ow_end; ow < wb; ow++) {
                            if (jcp.with_input_zp)
                                col[col_idx_base + ow] = izp;
                            else
                                col[col_idx_base + ow] = shift;
                        }
                    }
                });
            }
        }
    } else if (with_weights_zp) {
        parallel_nd(hb, [&](int oh) {
            for (int ic = 0; ic < jcp.ic; ic++) {
                uint8_t izp = with_input_zp ? input_zp[ic] : (uint8_t) 0;
                const int hp = tp;
                const int ih = (oh + hs) * sh - hp;
                if (ih < 0 || ih >= jcp.ih) {
                    for (int ow = 0; ow < wb; ow++) {
                        weights_zp_compensation[oh * wb + ow] += izp;
                    }
                } else {
                    const int wp = lp;
                    const int ow_start = saturate(0, wb, div_up(wp, sw) - ws);
                    const int ow_end = saturate(0, wb, div_up(jcp.iw + wp, sw) - ws);
                    for (int ow = 0; ow < ow_start; ow++) {
                        weights_zp_compensation[oh * wb + ow] += izp;
                    }

                    const int iw_base = ws * sw - wp;
                    const ptrdiff_t im_idx_base = ih * im_ih_stride + ic;
                    for (int ow = ow_start; ow < ow_end; ow++) {
                        const int iw = iw_base + ow * sw;
                        const ptrdiff_t im_idx = im_idx_base + iw * im_iw_stride;
                        weights_zp_compensation[oh * wb + ow] += im[im_idx];
                    }
                    for (int ow = ow_end; ow < wb; ow++) {
                        weights_zp_compensation[oh * wb + ow] += izp;
                    }
                }
            }
        });
    }
}

template void im2col_u8_compute<int8_t, false, false>(const jit_gemm_conv_conf_t &jcp,
        const int8_t *__restrict im, int8_t *__restrict imtr,
        uint8_t *__restrict col, int hs, int hb, int ws, int wb, const uint8_t *__restrict input_zp, int32_t *__restrict weights_zp_compensation);
template void im2col_u8_compute<int8_t, true, false>(const jit_gemm_conv_conf_t &jcp,
        const int8_t *__restrict im, int8_t *__restrict imtr,
        uint8_t *__restrict col, int hs, int hb, int ws, int wb, const uint8_t *__restrict input_zp, int32_t *__restrict weights_zp_compensation);
template void im2col_u8_compute<int8_t, false, true>(const jit_gemm_conv_conf_t &jcp,
        const int8_t *__restrict im, int8_t *__restrict imtr,
        uint8_t *__restrict col, int hs, int hb, int ws, int wb, const uint8_t *__restrict input_zp, int32_t *__restrict weights_zp_compensation);
template void im2col_u8_compute<int8_t, true, true>(const jit_gemm_conv_conf_t &jcp,
        const int8_t *__restrict im, int8_t *__restrict imtr,
        uint8_t *__restrict col, int hs, int hb, int ws, int wb, const uint8_t *__restrict input_zp, int32_t *__restrict weights_zp_compensation);

template void im2col_u8_compute<uint8_t, false, false>(const jit_gemm_conv_conf_t &jcp,
        const uint8_t *__restrict im, uint8_t *__restrict imtr,
        uint8_t *__restrict col, int hs, int hb, int ws, int wb, const uint8_t *__restrict input_zp, int32_t *__restrict weights_zp_compensation);
template void im2col_u8_compute<uint8_t, true, false>(const jit_gemm_conv_conf_t &jcp,
        const uint8_t *__restrict im, uint8_t *__restrict imtr,
        uint8_t *__restrict col, int hs, int hb, int ws, int wb, const uint8_t *__restrict input_zp, int32_t *__restrict weights_zp_compensation);
template void im2col_u8_compute<uint8_t, false, true>(const jit_gemm_conv_conf_t &jcp,
        const uint8_t *__restrict im, uint8_t *__restrict imtr,
        uint8_t *__restrict col, int hs, int hb, int ws, int wb, const uint8_t *__restrict input_zp, int32_t *__restrict weights_zp_compensation);
template void im2col_u8_compute<uint8_t, true, true>(const jit_gemm_conv_conf_t &jcp,
        const uint8_t *__restrict im, uint8_t *__restrict imtr,
        uint8_t *__restrict col, int hs, int hb, int ws, int wb, const uint8_t *__restrict input_zp, int32_t *__restrict weights_zp_compensation);

/* col[kh][kw][ic][oh][ow] <-- im2col_u8(im[ih][iw][ic]) */
template <typename T>
void im2col_u8(const jit_gemm_conv_conf_t &jcp, const T *__restrict im,
        T *__restrict imtr, uint8_t *__restrict col, int hs, int hb, int ws,
        int wb, const uint8_t *__restrict input_zp, int32_t *__restrict weights_zp_compensation) {

    if (!jcp.with_input_zp && !jcp.with_weights_zp)
        im2col_u8_compute<T, false, false>(jcp, im, imtr, col, hs, hb, ws, wb, input_zp, weights_zp_compensation);
    else if (jcp.with_input_zp && !jcp.with_weights_zp)
        im2col_u8_compute<T, true, false>(jcp, im, imtr, col, hs, hb, ws, wb, input_zp, weights_zp_compensation);
    else if (!jcp.with_input_zp && jcp.with_weights_zp)
        im2col_u8_compute<T, false, true>(jcp, im, imtr, col, hs, hb, ws, wb, input_zp, weights_zp_compensation);
    else
        im2col_u8_compute<T, true, true>(jcp, im, imtr, col, hs, hb, ws, wb, input_zp, weights_zp_compensation);
}

template void im2col_u8<int8_t>(const jit_gemm_conv_conf_t &jcp,
        const int8_t *__restrict im, int8_t *__restrict imtr,
        uint8_t *__restrict col, int hs, int hb, int ws, int wb, const uint8_t *__restrict input_zp, int32_t *__restrict weights_zp_compensation);
template void im2col_u8<uint8_t>(const jit_gemm_conv_conf_t &jcp,
        const uint8_t *__restrict im, uint8_t *__restrict imtr,
        uint8_t *__restrict col, int hs, int hb, int ws, int wb, const uint8_t *__restrict input_zp, int32_t *__restrict weights_zp_compensation);

/* imtr[ic][od][oh][ow] <-- im[id][ih][iw][ic]*/
template <typename T>
void transpose_u8(const jit_gemm_conv_conf_t &jcp, const T *__restrict im,
        T *__restrict imtr, const uint8_t *__restrict input_zp) {
    const int ic_stride = jcp.id * jcp.ih * jcp.iw;
    const int IC = jcp.ngroups * jcp.ic;
    const int IHW = jcp.ih * jcp.iw;
    constexpr int ic_block = 64;
    const int nb_ic = jcp.ic / ic_block;
    const int ic_blocked = nb_ic * ic_block;
    parallel_nd(jcp.id, jcp.ih, [&](int id, int ih) {
        const T *__restrict im_h = im + id * IHW * IC + ih * jcp.iw * IC;
        T *__restrict imtr_h = imtr + id * IHW + ih * jcp.iw;
        for (int iw = 0; iw < jcp.iw; iw++) {
            const T *__restrict im_w = im_h + iw * IC;
            T *__restrict imtr_w = imtr_h + iw;
            for (int icb = 0; icb < nb_ic; icb++) {
                const T *__restrict im_icb = im_w + icb * ic_block;
                T *__restrict imtr_icb = imtr_w + icb * ic_block * ic_stride;
                PRAGMA_OMP_SIMD()
                for (int ic = 0; ic < ic_block; ic++) {
                    uint8_t shift = jcp.signed_input ? 128 :
                                    jcp.with_input_zp ? input_zp[ic + icb * ic_block] : 0;

                    imtr_icb[ic * ic_stride] = im_icb[ic] + shift;
                }
            }
            for (int ic = ic_blocked; ic < jcp.ic; ic++) {
                uint8_t shift = jcp.signed_input ? 128 :
                                jcp.with_input_zp ? input_zp[ic] : 0;

                imtr_w[ic * ic_stride] = im_w[ic] + shift;
            }
        }
    });
}

template void transpose_u8(const jit_gemm_conv_conf_t &jcp,
        const int8_t *__restrict im, int8_t *__restrict imtr, const uint8_t *__restrict input_zp);
template void transpose_u8(const jit_gemm_conv_conf_t &jcp,
        const uint8_t *__restrict im, uint8_t *__restrict imtr, const uint8_t *__restrict input_zp);

/* col[kd][kh][kw][g][ic][od][oh][ow] <-- im2col_u8_3d(im[id][ih][iw][g][ic]) */
template <typename T>
void im2col_u8_3d(const jit_gemm_conv_conf_t &jcp, const T *__restrict imtr,
                  uint8_t *__restrict col, int od, const uint8_t *__restrict input_zp, int32_t *__restrict weights_zp_compensation) {
    const int dd = 1 + jcp.dilate_d;
    const int dh = 1 + jcp.dilate_h;
    const int dw = 1 + jcp.dilate_w;
    const int sd = jcp.stride_d;
    const int sh = jcp.stride_h;
    const int sw = jcp.stride_w;
    const int fp = jcp.f_pad;
    const int tp = jcp.t_pad;
    const int lp = jcp.l_pad;
    const int col_ic_s = jcp.oh * jcp.ow;
    const int col_kw_s = jcp.ic * col_ic_s;
    const int col_kh_s = jcp.kw * col_kw_s;
    const int col_kd_s = jcp.kh * col_kh_s;
    const int IHW = jcp.ih * jcp.iw;
    const int OHW = jcp.oh * jcp.ow;

    if (sd == 1 && sh == 1 && sw == 1 && dd == 1 && dh == 1 && dw == 1)
        parallel_nd(jcp.kd, jcp.kh, jcp.kw, jcp.ic,
                [&](int kd, int kh, int kw, int ic) {
                    uint8_t shift = jcp.signed_input ? 128 :
                                    jcp.with_input_zp ? input_zp[ic] : 0;

                    uint8_t *__restrict col_loc = col + kd * col_kd_s
                            + kh * col_kh_s + kw * col_kw_s + ic * col_ic_s;
                    const int id = od - fp + kd;
                    if (id < 0 || id >= jcp.id) {
                        for (ptrdiff_t i = 0; i < OHW; i++)
                            col_loc[i] = shift;
                        return;
                    }
                    const T *__restrict imtr_loc
                            = imtr + (ic * jcp.id + id) * IHW;
                    const int oh_start = saturate(0, jcp.oh, tp - kh);
                    const int oh_end = saturate(0, jcp.oh, jcp.ih + tp - kh);
                    const int ow_start = saturate(0, jcp.ow, lp - kw);
                    const int ow_end = saturate(0, jcp.ow, jcp.iw + lp - kw);
                    for (int oh = oh_start, ih = oh_start - tp + kh;
                            oh < oh_end; oh++, ih++) {
                        uint8_t *__restrict col_h = col_loc + oh * jcp.ow;
                        const T *__restrict imtr_h = imtr_loc + ih * jcp.iw;
                        for (int ow = ow_start, iw = ow_start - lp + kw;
                                ow < ow_end; ow++, iw++) {
                            col_h[ow] = imtr_h[iw];
                        }
                    }
                });
    else if (sd == 2 && sh == 2 && sw == 2 && dd == 1 && dh == 1 && dw == 1)
        parallel_nd(jcp.kd, jcp.kh, jcp.kw, jcp.ic,
                [&](int kd, int kh, int kw, int ic) {
                    uint8_t shift = jcp.signed_input ? 128 :
                                    jcp.with_input_zp ? input_zp[ic] : 0;

                    uint8_t *__restrict col_loc = col + kd * col_kd_s
                            + kh * col_kh_s + kw * col_kw_s + ic * col_ic_s;
                    const int id = od * 2 - fp + kd;
                    if (id < 0 || id >= jcp.id) {
                        for (ptrdiff_t i = 0; i < OHW; i++)
                            col_loc[i] = shift;
                        return;
                    }
                    const T *__restrict imtr_loc
                            = imtr + (ic * jcp.id + id) * IHW;
                    const int oh_start
                            = saturate(0, jcp.oh, div_up(tp - kh, 2));
                    const int oh_end
                            = saturate(0, jcp.oh, div_up(jcp.ih + tp - kh, 2));
                    const int ow_start
                            = saturate(0, jcp.ow, div_up(lp - kw, 2));
                    const int ow_end
                            = saturate(0, jcp.ow, div_up(jcp.iw + lp - kw, 2));
                    for (int oh = oh_start, ih = oh_start * 2 - tp + kh;
                            oh < oh_end; ++oh, ih += 2) {
                        uint8_t *__restrict col_h = col_loc + oh * jcp.ow;
                        const T *__restrict imtr_h = imtr_loc + ih * jcp.iw;
                        for (int ow = ow_start, iw = ow_start * 2 - lp + kw;
                                ow < ow_end; ++ow, iw += 2) {
                            col_h[ow] = imtr_h[iw];
                        }
                    }
                });
    else
        parallel_nd(jcp.kd, jcp.kh, jcp.kw, jcp.ic,
                [&](int kd, int kh, int kw, int ic) {
                    uint8_t shift = jcp.signed_input ? 128 :
                                    jcp.with_input_zp ? input_zp[ic] : 0;

                    uint8_t *__restrict col_loc = col + kd * col_kd_s
                            + kh * col_kh_s + kw * col_kw_s + ic * col_ic_s;
                    const int id = od * sd - fp + kd * dd;
                    if (id < 0 || id >= jcp.id) {
                        for (ptrdiff_t i = 0; i < OHW; i++)
                            col_loc[i] = shift;
                        return;
                    }
                    const T *__restrict imtr_loc
                            = imtr + (ic * jcp.id + id) * IHW;
                    const int oh_start
                            = saturate(0, jcp.oh, div_up(tp - kh * dh, sh));
                    const int oh_end = saturate(
                            0, jcp.oh, div_up(jcp.ih + tp - kh * dh, sh));
                    const int ow_start
                            = saturate(0, jcp.ow, div_up(lp - kw * dw, sw));
                    const int ow_end = saturate(
                            0, jcp.ow, div_up(jcp.iw + lp - kw * dw, sw));
                    for (int oh = oh_start, ih = oh_start * sh - tp + kh * dh;
                            oh < oh_end; ++oh, ih += sh) {
                        uint8_t *__restrict col_h = col_loc + oh * jcp.ow;
                        const T *__restrict imtr_h = imtr_loc + ih * jcp.iw;
                        for (int ow = ow_start,
                                 iw = ow_start * sw - lp + kw * dw;
                                ow < ow_end; ++ow, iw += sw) {
                            col_h[ow] = imtr_h[iw];
                        }
                    }
                });
}

template void im2col_u8_3d<int8_t>(const jit_gemm_conv_conf_t &jcp, const int8_t *__restrict im,
                                   uint8_t *__restrict col, int od, const uint8_t *__restrict input_zp, int32_t *__restrict weights_zp_compensation);

template void im2col_u8_3d<uint8_t>(const jit_gemm_conv_conf_t &jcp, const uint8_t *__restrict im,
                                    uint8_t *__restrict col, int od, const uint8_t *__restrict input_zp, int32_t *__restrict weights_zp_compensation);

/* im[ih][iw][ic] <-- col2im_s32(col[oh][ow][kh][kw][ic]) */
void col2im_s32(const jit_gemm_conv_conf_t &jcp, const int32_t *__restrict col,
        int32_t *__restrict im)
{
    parallel(0, (size_t)mkldnn_get_max_threads(), [&](const int ithr, const int nthr) {
        int h_nthr = nstl::min(jcp.ih, nthr);
        int w_nthr = nstl::min(jcp.iw, nthr / h_nthr);
        int h_ithr = 1, h_s = 0, h_e = 0, w_ithr = 1, w_s = 0, w_e = 0;
        if (ithr < h_nthr * w_nthr) {
            h_ithr = ithr / w_nthr;
            w_ithr = ithr % w_nthr;
            balance211(jcp.ih, h_nthr, h_ithr, h_s, h_e);
            balance211(jcp.iw, w_nthr, w_ithr, w_s, w_e);
        } else {
            h_ithr = w_ithr = -ithr;
            h_s = h_e = w_s = w_e = -1;
        }

        for (int ih = h_s; ih < h_e; ++ih) {
            for (int iw = w_s; iw < w_e; ++iw) {
                PRAGMA_OMP_SIMD()
                for (int ic = 0; ic < jcp.ic; ++ic) {
                    im[(ih * jcp.iw + iw) * jcp.ic + ic] = 0;
                }
            }
        }

        // TODO: reduce region: [0.. oh] --> [h_s * sh .. h_e * sh]
        for (int oh = 0; oh < jcp.oh; ++oh) {
            for (int ow = 0; ow < jcp.ow; ++ow) {
                for (int kh = 0; kh < jcp.kh; ++kh) {
                    const int ih = oh * jcp.stride_h
                        - jcp.t_pad + kh * (1 + jcp.dilate_h);
                    if (ih < h_s || ih >= h_e) continue;

                    for (int kw = 0; kw < jcp.kw; ++kw) {
                        const int iw = ow * jcp.stride_w
                            - jcp.l_pad + kw * (1 + jcp.dilate_w);
                        if (iw < w_s || iw >= w_e) continue;

                        const size_t col_idx = (((oh * jcp.ow + ow) * jcp.kh
                                + kh) * jcp.kw + kw) * jcp.ic;
                        const size_t im_idx
                            = (ih * jcp.iw + iw) * jcp.ic;
                        PRAGMA_OMP_SIMD()
                        for (int ic = 0; ic < jcp.ic; ++ic) {
                            im[im_idx + ic] += col[col_idx + ic];
                        }
                    }
                }
            }
        }
    });
}

void col2im_3d(const jit_gemm_conv_conf_t &jcp, const float *col, float *im,
        int od)
{
    parallel_nd(jcp.ic, [&](int ic) {
        const float *__restrict col_ = col + (size_t)ic * jcp.ks * jcp.os;
        float *__restrict im_ic = im + (size_t)ic * jcp.ih * jcp.iw * jcp.id;

        int id = od * jcp.stride_d - jcp.f_pad;
        for (int kd = 0; kd < jcp.kd; ++kd) {
            if (id < 0 || id >= jcp.id) {
                col_ += jcp.kh * jcp.kw * jcp.os;
                id += (1 + jcp.dilate_d);
                continue;
            }

            float *__restrict im_ = im_ic + id * jcp.ih * jcp.iw;

            for (int oh = 0; oh < jcp.oh; ++oh) {
            for (int kh = 0; kh < jcp.kh; ++kh) {
                const int ih = oh * jcp.stride_h - jcp.t_pad
                    + kh * (1 + jcp.dilate_h);
                if (ih < 0 || ih >= jcp.ih) continue;

                for (int ow = 0; ow < jcp.ow; ++ow) {
                for (int kw = 0; kw < jcp.kw; ++kw) {
                    const int iw = ow * jcp.stride_w - jcp.l_pad
                        + kw * (1 + jcp.dilate_w);
                    if (iw < 0 || iw >= jcp.iw) continue;

                    const size_t col_idx =
                        ((kh * jcp.kw + kw) * jcp.oh + oh) * jcp.ow + ow;
                    const size_t im_idx = ih*jcp.iw + iw;
                    im_[im_idx] += col_[col_idx];
                }}
            }}

            col_ += jcp.kh * jcp.kw * jcp.os;
            id += (1 + jcp.dilate_d);
        }
    });
}

void col2im(const jit_gemm_conv_conf_t &jcp, const float *col, float *im) {
    const size_t col_step = jcp.ks * jcp.os;
    const size_t im_step = jcp.ih * jcp.iw;
    const int iS = jcp.ih * jcp.iw;

    parallel_nd(jcp.ic, [&](int ic) {
        float *__restrict im_ = im + ic * im_step;
        const float *__restrict col_ = col + ic * col_step;
        PRAGMA_OMP_SIMD()
        for (int is = 0; is < iS; ++is) im_[is] = 0.;

        for (int kh = 0; kh < jcp.kh; ++kh) {
        for (int oh = 0; oh < jcp.oh; ++oh) {
            const int ih =
                    oh * jcp.stride_h - jcp.t_pad + kh * (1 + jcp.dilate_h);
            if (ih < 0 || ih >= jcp.ih) continue;

            for (int kw = 0; kw < jcp.kw; ++kw) {
            for (int ow = 0; ow < jcp.ow; ++ow) {
                const int iw =
                        ow * jcp.stride_w - jcp.l_pad + kw * (1 + jcp.dilate_w);
                if (iw < 0 || iw >= jcp.iw) continue;

                const size_t col_idx = ((kh*jcp.kw + kw)*jcp.oh+oh)*jcp.ow+ow;
                const size_t im_idx = ih*jcp.iw + iw;
                im_[im_idx] += col_[col_idx];
            }
            }
        }
        }
    });
}

status_t init_conf(jit_gemm_conv_conf_t &jcp,
        memory_tracking::registrar_t &scratchpad, const convolution_desc_t &cd,
        const memory_desc_wrapper &src_d, const memory_desc_wrapper &weights_d,
        const memory_desc_wrapper &dst_d, const primitive_attr_t &attr, int max_threads) {
    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    const int ndims = src_d.ndims();
    const int is_1d = ndims == 3;
    const int is_3d = ndims == 5;

    jcp.prop_kind = cd.prop_kind;

    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];

    jcp.oc = dst_d.dims()[1] / jcp.ngroups;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;
    jcp.id = is_3d ? src_d.dims()[2] : 1;
    jcp.ih = is_1d ? 1 : src_d.dims()[ndims - 2];
    jcp.iw = src_d.dims()[ndims - 1];
    jcp.od = is_3d ? dst_d.dims()[2] : 1;
    jcp.oh = is_1d ? 1 : dst_d.dims()[ndims - 2];
    jcp.ow = dst_d.dims()[ndims - 1];

    jcp.kd = is_3d ? weights_d.dims()[with_groups + 2] : 1;
    jcp.kh = is_1d ? 1 : weights_d.dims()[with_groups + ndims - 2];
    jcp.kw = weights_d.dims()[with_groups + ndims - 1];

    jcp.f_pad = is_3d ? cd.padding[0][0] : 0;
    jcp.t_pad = is_1d ? 0 : cd.padding[0][ndims - 4];
    jcp.l_pad = cd.padding[0][ndims - 3];

    jcp.stride_d = is_3d ? cd.strides[0] : 1;
    jcp.stride_h = is_1d ? 1 : cd.strides[ndims - 4];
    jcp.stride_w = cd.strides[ndims - 3];

    jcp.dilate_d = is_3d ? cd.dilates[0] : 0;
    jcp.dilate_h = is_1d ? 0 : cd.dilates[ndims - 4];
    jcp.dilate_w = cd.dilates[ndims - 3];

    jcp.src_fmt = src_d.format();
    jcp.with_bias = cd.bias_desc.format != memory_format::undef
        || cd.diff_bias_desc.format != memory_format::undef;

    jcp.is = jcp.ih * jcp.iw;
    jcp.os = jcp.oh * jcp.ow;
    jcp.ks = jcp.kh * jcp.kw * jcp.kd;

    jcp.signed_input = src_d.data_type() == data_type::s8;
    jcp.wei_adj_scale =
        !jcp.signed_input || mayiuse(avx512_core_vnni) ? 1.f : 0.5f;

    jcp.outer_threading = false;

    jcp.with_input_zp = !attr.input_zero_points_.has_default_values();
    if (jcp.with_input_zp) {
        if (attr.input_zero_points_.count_ != 1 && attr.input_zero_points_.count_ != jcp.ic * jcp.ngroups)
            return status::unimplemented;

        if (attr.output_compensations_.count_ != jcp.oc * jcp.ngroups)
            return status::unimplemented;
    }

    jcp.with_weights_zp = !attr.weights_zero_points_.has_default_values();
    if (jcp.with_weights_zp) {
        if (attr.weights_zero_points_.count_ != 1 && attr.weights_zero_points_.count_ != jcp.oc * jcp.ngroups)
            return status::unimplemented;
    }

    bool is_int8_conv = utils::one_of(src_d.data_type(), s32, s8, u8)
        && weights_d.data_type() == s8;

    const bool is_bwd_d = jcp.prop_kind == backward_data;
    const bool is_bwd_w = jcp.prop_kind == backward_weights;
    const bool is_fwd = !is_bwd_d && !is_bwd_w;

    bool is_bf16_conv = false
        || (is_fwd && utils::everyone_is(bf16,
                src_d.data_type(), weights_d.data_type()))
        || (is_bwd_d && utils::everyone_is(bf16,
                dst_d.data_type(), weights_d.data_type()))
        || (is_bwd_w && utils::everyone_is(bf16,
                src_d.data_type(), dst_d.data_type()));
    if (is_bf16_conv && !mayiuse(avx512_core))
        return status::unimplemented;

    bool is_bf16_to_bf16_conv = is_bf16_conv
        && ((is_fwd && bf16 == dst_d.data_type())
                || (is_bwd_d && bf16 == src_d.data_type())
                || (is_bwd_w && bf16 == weights_d.data_type()));

    const int vlen = mayiuse(avx512_common)
        ? cpu_isa_traits<avx512_common>::vlen
        : mayiuse(avx)
            ? cpu_isa_traits<avx>::vlen
            : mayiuse(sse42) ? cpu_isa_traits<sse42>::vlen : 4;
    const int data_size = (is_int8_conv ? 1 : (is_bf16_conv ? 2 : 4));
    const int simd_w = vlen / data_size;

    jcp.os_block = jcp.os;
    jcp.oc_block = jcp.oc;
    jcp.ic_block = jcp.ic;
    jcp.loop_order = gemm_loop_rlb;
    jcp.nthr_oc = 1;

    jcp.oh_block = is_fwd ? jcp.oh : jcp.ih;
    jcp.ow_block = is_fwd ? jcp.ow : jcp.iw;

    using namespace memory_tracking::names;
    bool is_depthwise = jcp.ic == 1 && jcp.oc == 1 && jcp.ngroups != 1;

    // TODO: maybe mitigate blocking restriction
    const int L2 = get_cache_size(2, true)
          / data_size;
    const int gemm_thrld = 64 * 1024;

    if (is_int8_conv) {
        if (is_fwd) {
            jcp.im2col_sz
                    = !everyone_is(true, jcp.ow == jcp.iw, jcp.oh == jcp.ih,
                              jcp.od == jcp.id, jcp.stride_w == 1,
                              jcp.stride_h == 1, jcp.stride_d == 1, jcp.ks == 1,
                              !jcp.signed_input)
                    ? (ptrdiff_t)jcp.ic * jcp.ks * jcp.os
                    : 0;

            const int wei_size = jcp.oc * jcp.ic * jcp.kh * jcp.kw;
            bool is_blocking_applicable = true && is_fwd && jcp.im2col_sz
                    && !is_3d && jcp.dilate_h == 0
                    && jcp.dilate_w == 0 && !is_depthwise && wei_size < L2 / 2;
            if (is_blocking_applicable) {
                // looking for oh and ow blocking
                int h_block{ jcp.oh_block }, w_block{ jcp.ow_block };
                const int ic = jcp.ic;
                const int oc = jcp.oc;
                const int iw = jcp.iw;
                const int ow = jcp.ow;
                const int oh = jcp.oh;
                const int os = oh * ow;

                // 1. cache requirement
                int row_size = ic * ow * jcp.ks + 2 * (ic * iw + oc * ow);
                // Heuristic rule: gemm needed a lot of memory for internal
                // usage
                row_size *= 5;
                // memory for accumulators
                row_size += oc * ow * sizeof(uint32_t);
                // memory for transposition
                row_size += ic * iw;

                h_block = nstl::max(1, nstl::min(oh, div_up(L2, row_size)));
                if (h_block == 1) {
                    int col_size = ic * jcp.ks + 2 * (ic + oc);
                    if (is_int8_conv) {
                        col_size *= 5;
                        col_size += oc * sizeof(uint32_t);
                        col_size += ic;
                    }
                    w_block = nstl::max(1, nstl::min(ow, div_up(L2, col_size)));
                }

                // 2. threading requirement
                if (h_block != oh)
                    h_block = nstl::max(1, rnd_dn(h_block, 4));
                if (w_block != ow)
                    w_block = nstl::max(1, rnd_dn(w_block, simd_w));

                float thr_eff = 0.f;
                float thr_eff_treshold = 0.9f;
                if (w_block == ow) {
                    do {
                        int nb_h = div_up(oh, h_block);
                        size_t work = jcp.ngroups * jcp.mb * jcp.od * nb_h;
                        float disb = (float)oh / rnd_up(oh, h_block);
                        thr_eff = (float)work / rnd_up(work, max_threads);
                        thr_eff = (thr_eff + disb) / 2.f;
                        if (thr_eff >= thr_eff_treshold)
                            break;
                        h_block = rnd_dn(h_block - 4, 4);
                    } while (h_block > 0);
                }
                if (thr_eff
                        < thr_eff_treshold) // we didn't find suitable h_block
                {
                    h_block = 1;
                    int nb_h = oh;
                    do {
                        int nb_w = div_up(ow, w_block);
                        size_t work_amount = jcp.ngroups * jcp.mb * nb_h * nb_w;
                        float disb = (float)ow / rnd_up(ow, w_block);
                        thr_eff = (float)work_amount
                                / rnd_up(work_amount, max_threads);
                        thr_eff = (thr_eff + disb) / 2.f;
                        if (thr_eff > thr_eff_treshold)
                            break;
                        w_block = rnd_dn(w_block - simd_w, simd_w);
                    } while (w_block > 0);
                }
                h_block = nstl::max(1, h_block);
                w_block = nstl::max(1, w_block);
                const size_t inner_work
                        = div_up(os, simd_w) * div_up(oc, simd_w);
                const float inner_thr_eff
                        = (float)inner_work / rnd_up(inner_work, max_threads);
                if (thr_eff >= inner_thr_eff / 2 && h_block > 0
                        && w_block > 0) {
                    jcp.oh_block = h_block;
                    jcp.ow_block = w_block;
                    jcp.outer_threading = true;
                }
                // updating jcp.im2col_sz
                if (jcp.oh_block != 1)
                    jcp.ow_block = ow;
                jcp.im2col_sz
                        = (ptrdiff_t)ic * jcp.ks * jcp.oh_block * jcp.ow_block;
            }
            //  For threading selection in bwd_d we do:
            //  1. Rough estimation of efficiency for inner and outer threading.
            //  2. Gemm size estimation in assumption that it does not work
            //  so effectively for small sizes.
            //  64K - this is heuristic gemm size per thread threshold.
            const int gemm_thrld = 64 * 1024;
            if (!jcp.outer_threading && !is_3d) {
                bool is_depthwise = jcp.ic == 1 && jcp.oc == 1
                   && jcp.ngroups != 1;
                const size_t outer_work = jcp.ngroups * jcp.mb;
                const float outer_thr_eff
                        = (float)outer_work / rnd_up(outer_work, max_threads);
                const size_t inner_work
                        = div_up(jcp.is, simd_w) * div_up(jcp.ic, simd_w);
                const float inner_thr_eff
                        = (float)inner_work / rnd_up(inner_work, max_threads);
                jcp.outer_threading
                        = (is_depthwise
                                  || (jcp.is / max_threads < 64 && jcp.mb != 1))
                        && (outer_thr_eff / inner_thr_eff >= 1.f
                                   || (jcp.os * jcp.ic * jcp.oc) / max_threads
                                           < gemm_thrld);
            }
            jcp.nthr = jcp.outer_threading ? max_threads : 1;
            scratchpad.book(key_conv_gemm_col,
                    sizeof(int8_t) * jcp.nthr * jcp.im2col_sz);
            scratchpad.book(key_conv_int_dat_in_acc_dt,
                sizeof(int32_t) * jcp.nthr * jcp.oh_block
                    * jcp.ow_block * jcp.oc);
            scratchpad.book(key_conv_gemm_imtr,
                    sizeof(int8_t) * jcp.nthr * jcp.id * jcp.is * jcp.ic);

            if (jcp.with_input_zp || jcp.with_weights_zp)
                scratchpad.book(key_conv_padded_compensation, sizeof(int32_t) * jcp.ngroups * jcp.oc);
            if (jcp.with_weights_zp)
                scratchpad.book(key_weights_zp_compensation, sizeof(int32_t) * jcp.nthr * jcp.oh * jcp.ow);
        } else if (is_bwd_d) {
            jcp.im2col_sz
                    = !everyone_is(true, jcp.ow == jcp.iw, jcp.oh == jcp.ih,
                              jcp.od == jcp.id, jcp.stride_w == 1,
                              jcp.stride_h == 1, jcp.stride_d == 1, jcp.ks == 1,
                              !jcp.signed_input)
                    ? (ptrdiff_t)jcp.ic * jcp.ks * jcp.os * jcp.od
                    : 0;

            bool is_depthwise = jcp.ic == 1 && jcp.oc == 1 && jcp.ngroups != 1;
            const size_t outer_work = jcp.ngroups * jcp.mb;
            const float outer_thr_eff
                    = (float)outer_work / rnd_up(outer_work, max_threads);
            const size_t inner_work
                    = div_up(jcp.is, simd_w) * div_up(jcp.ic, simd_w);
            const float inner_thr_eff
                    = (float)inner_work / rnd_up(inner_work, max_threads);
            jcp.outer_threading = !is_3d
                    && (is_depthwise
                            || (jcp.is / max_threads < 64 && jcp.mb != 1))
                    && (outer_thr_eff / inner_thr_eff >= 1.f
                            || (jcp.is * jcp.ic * jcp.oc) / max_threads
                                    < gemm_thrld);

            jcp.nthr = jcp.outer_threading ? max_threads : 1;
            scratchpad.book(key_conv_gemm_col,
                    sizeof(int32_t) * jcp.nthr * jcp.im2col_sz);
            scratchpad.book(key_conv_int_dat_in_acc_dt,
                    sizeof(int32_t) * jcp.nthr * jcp.is * jcp.id * jcp.ic);
        } else if (is_bwd_w) {
            assert(!"unimplemented prop_kind");
            return status::unimplemented;
        }
    } else {
        jcp.im2col_sz = !everyone_is(true, jcp.ow == jcp.iw, jcp.oh == jcp.ih,
                        jcp.od == jcp.id, jcp.stride_w == 1,
                        jcp.stride_h == 1, jcp.stride_d == 1,
                        jcp.ks == 1, !jcp.signed_input)
        ? (ptrdiff_t)jcp.ic * jcp.ks * jcp.os
        : 0;
        if (is_fwd) {
            const int sh = jcp.stride_h;
            const int sw = jcp.stride_w;
            const int spatial = jcp.mb * jcp.ngroups * jcp.od * jcp.os;
            int K = jcp.ic * jcp.ks;

            // There is some heuristics in the definition of
            // inner/outer threading cross point due to the nature of the
            // gemm implementation which we cannot control
            bool is_blocking_applicable = true
                && !is_bf16_conv // TODO: apply blocking to bf16
                && !is_3d
                && (!jcp.im2col_sz
                    // spatial is small
                    || spatial >= max_threads * simd_w
                    // inner threading work is greater then outer threading work
                    || jcp.os < jcp.mb * jcp.ngroups * jcp.od
                    // im2col is big
                    || (sw == 1 && K <= 0.05 * jcp.oc))
                // heuristic condition
                && (jcp.im2col_sz
                    || (jcp.ic / jcp.oc < 42
                        && jcp.ic * jcp.oc * jcp.is < 1024));

            if (is_blocking_applicable) {
                const int min_oc_block = 8;
                const int min_os_block = simd_w;
                const int non_cache_access = 20;
                const float strided_im2col_k = 8;
                const float thr_disb_k = 8;
                const float thr_mem_eff_k{ 1 }, oc_disb_k{ 1 }, os_disb_k{ 1 },
                        ic_disb_k{ 1 }, reg_osb_disb_k{ 1 }, gemm_eff_k{ 0.5 },
                        gemm_calc_eff_k{ 1 };
                const float k_sum = thr_disb_k + oc_disb_k + os_disb_k
                        + ic_disb_k + reg_osb_disb_k + thr_mem_eff_k
                        + gemm_eff_k + gemm_calc_eff_k;

                auto calc_max_icb = [=](int nthr_oc, int ocb, int osb,
                                            int oc_per_thr, int os_per_thr) {
                    const int block_out_size = ocb * osb;
                    // TODO: need more precise calculation if stride more than
                    // kernel size
                    const int inp_row_size = sh * sw * osb;
                    int max_icb = 1;
                    if (jcp.im2col_sz) {
                        const int col_row_size = jcp.ks * osb;
                        if (osb >= os_per_thr) { // one pass by os
                            const int wei_col_size = jcp.ks * ocb;
                            max_icb = L2 / (inp_row_size + col_row_size);
                            if (ocb < oc_per_thr) {
                                max_icb = nstl::min(max_icb,
                                        (L2 - block_out_size)
                                                / (col_row_size
                                                          + wei_col_size));
                            }
                        } else {
                            const int wei_col_size = jcp.ks * oc_per_thr;
                            max_icb = (L2 - block_out_size)
                                    / (inp_row_size + col_row_size
                                              + wei_col_size);
                        }
                    } else {
                        if (osb >= os_per_thr)
                            max_icb = L2 / inp_row_size;
                        else {
                            const int wei_col_size = jcp.ks * oc_per_thr;
                            max_icb = L2 / (inp_row_size + wei_col_size);
                        }
                    }
                    if (max_icb < jcp.ic) {
                        if (jcp.im2col_sz) {
                            const int col_row_size = jcp.ks * osb;
                            const int wei_col_size = jcp.ks * oc_per_thr;
                            max_icb = (L2 - block_out_size)
                                    / (inp_row_size + col_row_size
                                              + wei_col_size);
                        }
                    }
                    return max_icb;
                };

                auto est_eff = [=](int nthr_oc, int ocb, int osb, int &icb,
                                       int max_oc_per_thr, int max_os_per_thr) {
                    // for given nthr_oc, oc block:
                    // 1. find ic block to fit into cache
                    // 2. estimate efficiency basing on rules and heuristic:
                    // - Minimize im2col cost
                    // - ratio of FMA number to data size
                    // - gemm works better if M divided by 48 and N divided by 8
                    if (osb > max_os_per_thr || ocb > max_oc_per_thr)
                        return 0.f;

                    int sp_start{ 0 }, sp_end{ 0 }, oc_start{ 0 }, oc_end{ 0 };
                    size_t max_thr_size{ 0 }, working_threads{ 0 },
                            min_thr_size{ (size_t)spatial * jcp.oc + 1 },
                            max_y{ 0 }, max_oc{ 0 };

                    for (int i = 0; i < max_threads; i++) {
                        balance2D(max_threads, i, spatial, sp_start, sp_end,
                                jcp.oc, oc_start, oc_end, nthr_oc);
                        const size_t thr_size
                                = (sp_end - sp_start) * (oc_end - oc_start);
                        if (thr_size > 0)
                            working_threads++;
                        if (thr_size > max_thr_size) {
                            max_y = (sp_end - sp_start);
                            max_oc = (oc_end - oc_start);
                            max_thr_size = thr_size;
                        }
                        if (thr_size < min_thr_size)
                            min_thr_size = thr_size;
                    }
                    auto thr_disb = max_thr_size > 0
                            ? (min_thr_size > 0
                                    ? (float)min_thr_size / max_thr_size
                                    : (float)working_threads / max_threads)
                            : 1;

                    const int oc_per_thr = max_oc;
                    const int os_per_thr = max_y;
                    ocb = nstl::min(oc_per_thr, ocb);
                    const int os_max = nstl::min(jcp.os, os_per_thr);
                    osb = nstl::min(os_max, osb);

                    // -- selecting icb ---------------------
                    int max_ic_block = calc_max_icb(
                            nthr_oc, ocb, osb, oc_per_thr, os_per_thr);
                    // if we don't fit into cache then access to memory is
                    // expensive
                    int mem_access_cost
                            = (max_ic_block < 1) ? non_cache_access : 1;
                    max_ic_block = nstl::max(1, max_ic_block);
                    icb = nstl::max(1, jcp.ic / div_up(jcp.ic, max_ic_block));
                    int nb_ic = div_up(jcp.ic, icb);
                    int kb = icb * jcp.ks;
                    int kb_caligned = rnd_up(kb, simd_w);

                    // -- mem efficiency ------------
                    const size_t out_size
                            = oc_per_thr * rnd_up(os_per_thr, simd_w);
                    const size_t out_ops = mem_access_cost * out_size
                            * ((icb == jcp.ic) ? 1 : (2 * nb_ic - 1));
                    const int osb_caligned = rnd_up(osb, simd_w);
                    const size_t inp_size
                            = jcp.ic * rnd_up(os_per_thr * sh * sw, simd_w);
                    size_t inp_ops = 0;
                    size_t col_ops = 0;
                    // TODO: simplify calculations
                    if (jcp.im2col_sz) {
                        inp_ops = mem_access_cost * jcp.ks * inp_size;
                        const float col_tail_koeff = (float)osb_caligned / osb;
                        col_ops = (size_t)(mem_access_cost * 2 * jcp.ks
                                * inp_size * col_tail_koeff);
                        if (sw != 1) // im2col with strides is much slower
                            col_ops = (size_t)(col_ops * strided_im2col_k);
                    } else {
                        inp_ops = mem_access_cost * jcp.ks * inp_size;
                    }
                    // TODO: what about groups?
                    const size_t wei_size = oc_per_thr * rnd_up(K, simd_w);
                    const size_t wei_ops = mem_access_cost * wei_size;
                    // ratio of real FMA to number of memory ops
                    const float thr_mem_eff
                            = (((float)os_per_thr / simd_w) * oc_per_thr * K)
                            / (inp_ops + col_ops + wei_ops + out_ops);

                    auto oc_disb = (float)oc_per_thr / rnd_up(oc_per_thr, ocb);
                    auto os_disb = (float)os_max / rnd_up(os_max, osb);
                    auto ic_disb = (float)jcp.ic / rnd_up(jcp.ic, icb);

                    auto reg_osb_disb = (float)osb / rnd_up(osb, 3 * simd_w);

                    // Heuristics
                    const float gemm_eff = ((float)osb * ocb * kb)
                            / ((float)oc_per_thr * os_per_thr * K);

                    // number of FMA to memory size
                    const float gemm_calc_eff
                            = (((float)osb / simd_w) * ocb * kb)
                            / (osb_caligned * kb + ocb * kb_caligned
                                      + ocb * osb_caligned);

                    const float res_eff = pow(pow(thr_disb, thr_disb_k)
                                    * pow(oc_disb, oc_disb_k)
                                    * pow(os_disb, os_disb_k)
                                    * pow(ic_disb, ic_disb)
                                    * pow(reg_osb_disb, reg_osb_disb_k)
                                    * pow(thr_mem_eff, thr_mem_eff_k)
                                    * pow(gemm_eff, gemm_eff_k)
                                    * pow(gemm_calc_eff, gemm_calc_eff_k),
                            1.f / k_sum);
                    return res_eff;
                };

                /* find the best thread distribution and blocking with highest
                 * efficiency */
                int best_nthr_oc{ 1 }, best_ocb{ jcp.oc }, best_osb{ jcp.os },
                        best_icb{ jcp.ic };
                float best_thr_eff = est_eff(best_nthr_oc, best_ocb, best_osb,
                        best_icb, jcp.oc, jcp.os);

                int icb{ best_icb };
                const int nthr_oc_max = max_threads;
                for (int nthr_oc = 1; nthr_oc <= nthr_oc_max; ++nthr_oc) {
                    const int max_oc_per_thr = div_up(jcp.oc, nthr_oc);
                    const int min_oc_per_thr
                            = nstl::min(min_oc_block, max_oc_per_thr);
                    const int max_os_per_thr = nstl::min(jcp.os,
                            div_up(spatial,
                                    nstl::max(1, max_threads / nthr_oc)));
                    const int min_os_per_thr
                            = nstl::min(min_os_block, max_os_per_thr);
                    for (int ocb = min_oc_per_thr; ocb <= max_oc_per_thr;
                            ocb += nstl::max(1,
                                    nstl::min(min_oc_block,
                                            max_oc_per_thr - ocb))) {
                        for (int osb = min_os_per_thr; osb <= jcp.os;
                                osb += nstl::max(1,
                                        nstl::min(min_os_block,
                                                max_os_per_thr - osb))) {
                            float thr_eff = est_eff(nthr_oc, ocb, osb, icb,
                                    max_oc_per_thr, max_os_per_thr);
                            if (thr_eff > best_thr_eff) {
                                best_thr_eff = thr_eff;
                                best_nthr_oc = nthr_oc;
                                best_ocb = ocb;
                                best_osb = osb;
                                best_icb = icb;
                            }
                        }
                    }
                }

                jcp.outer_threading = true;
                jcp.nthr_oc = best_nthr_oc;
                jcp.oc_block = best_ocb;
                jcp.os_block = best_osb;
                jcp.ic_block = best_icb;
                // TODO: define loop order
                // if im2col then gemm_loop_rlb and gemm_loop_lrb looks
                // preferable otherwise other loop orders are possible
                jcp.loop_order = gemm_loop_rlb;
            } else {
                const size_t outer_work_amount = jcp.ngroups * jcp.mb * jcp.od;
                const float outer_thr_eff = (float)outer_work_amount
                        / rnd_up(outer_work_amount, max_threads);
                const size_t inner_work_amount
                        = div_up(jcp.os, simd_w) * div_up(jcp.oc, simd_w);
                const float inner_thr_eff = (float)inner_work_amount
                        / rnd_up(inner_work_amount, max_threads);
                jcp.outer_threading = jcp.os / max_threads < 512
                        && IMPLICATION(
                                   jcp.od == 1, jcp.mb != 1 || jcp.ngroups > 2)
                        && (outer_thr_eff / inner_thr_eff >= 1.f
                                   || (jcp.os * jcp.ic * jcp.oc) / max_threads
                                           < gemm_thrld);
            }
            if (jcp.im2col_sz)
                jcp.im2col_sz = (ptrdiff_t)jcp.ic_block * jcp.ks * jcp.os_block;
        } else if (is_bwd_d) {
            const size_t outer_work_amount = jcp.ngroups * jcp.mb;
            const float outer_thr_eff = (float)outer_work_amount
                    / rnd_up(outer_work_amount, max_threads);
            const size_t inner_work
                    = div_up(jcp.is, simd_w) * div_up(jcp.ic, simd_w);
            const float inner_thr_eff
                    = (float)inner_work / rnd_up(inner_work, max_threads);
            jcp.outer_threading = (jcp.os / max_threads < 512 || jcp.ks < 64)
                    && (jcp.mb != 1 || jcp.ngroups > 2)
                    && (outer_thr_eff / inner_thr_eff >= 1.f
                               || (jcp.is * jcp.ic * jcp.oc) / max_threads
                                       < gemm_thrld);
        } else if (is_bwd_w)
            jcp.outer_threading = jcp.os / max_threads < 256
                    && (jcp.mb != 1 || jcp.ngroups > 2);

        jcp.nthr = jcp.outer_threading ? max_threads : 1;
        const size_t gemm_col_datatype_size = is_bf16_conv && !is_bwd_d
            ? sizeof(mkldnn_bfloat16_t)
            : sizeof(float);
        scratchpad.book(key_conv_gemm_col,
                gemm_col_datatype_size * jcp.nthr * jcp.im2col_sz);

        const int sizeof_cacheline_float = 16;
        if (is_bwd_w) {
            jcp.need_wei_reduction = mkldnn_thr_syncable()
                    ? jcp.mb != 1 && jcp.nthr != 1
                    : false;
            scratchpad.book(key_conv_wei_reduction,
                    sizeof(float) * jcp.nthr * weights_d.size());

            if (is_bf16_conv && jcp.with_bias) {
                const size_t ws_size = sizeof(float)
                    * max_threads * rnd_up(jcp.ow, sizeof_cacheline_float);
                scratchpad.book(key_conv_dst_bf16_convert_wsp, ws_size);
            }
        }

        if (is_bf16_to_bf16_conv) {
            size_t conv_acc_buffer_size = 0;
            if (is_fwd)
                conv_acc_buffer_size = sizeof(float) * jcp.nthr
                    * rnd_up(jcp.oc * jcp.oh_block * jcp.ow_block,
                          sizeof_cacheline_float);
            else if (is_bwd_d)
                conv_acc_buffer_size = sizeof(float) * jcp.nthr
                    * rnd_up(jcp.ic * jcp.ih * jcp.iw * jcp.id,
                          sizeof_cacheline_float);
            else if (is_bwd_w)
                conv_acc_buffer_size = sizeof(float) * weights_d.size();
            scratchpad.book(key_conv_int_dat_in_acc_dt, conv_acc_buffer_size);
        }
        if ((is_fwd && cd.bias_desc.data_type == data_type::bf16)
            || (is_bwd_w && cd.diff_bias_desc.data_type == data_type::bf16)) {
            scratchpad.book(key_conv_bias_bf16_convert_wsp,
                sizeof(float) * jcp.oc * jcp.ngroups);
        }
    }
    return status::success;
}

void bwd_weights_balance(int ithr, int nthr, int ngroups, int mb, int &ithr_g,
        int &nthr_g, int &ithr_mb, int &nthr_mb) {
    nthr_g = nstl::min(ngroups, nthr);
    nthr_mb = nstl::min(mb, nthr / nthr_g);
    if (ithr / nthr_mb >= ngroups) {
        ithr_g = ithr_mb = -1;
    } else {
        ithr_g = ithr / nthr_mb;
        ithr_mb = ithr % nthr_mb;
    }
}

void bwd_weights_reduction_par(int ithr, int nthr,
        const jit_gemm_conv_conf_t &jcp, const float *weights_reduce_ws,
        float *weights) {
    const size_t weights_g_size = jcp.ic * jcp.oc * jcp.ks;

    size_t weights_start{0}, weights_end{0};
    balance211(weights_g_size, nthr, ithr, weights_start, weights_end);

    for (int i = 0; i < nthr; ++i) {
        const float *ws_i = weights_reduce_ws + i * weights_g_size;
        for (size_t s = weights_start; s < weights_end; ++s)
            weights[s] = (i == 0 ? 0 : weights[s]) + ws_i[s];
    }
}

};

}
}
}
