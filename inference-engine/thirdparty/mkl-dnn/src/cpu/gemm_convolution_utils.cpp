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

void im2col_3d(const jit_gemm_conv_conf_t &jcp, const float *im, float *col,
        int od)
{
    const size_t OHW = jcp.oh * jcp.ow;
    const size_t im_step = jcp.ih * jcp.iw * jcp.id;
    const size_t col_step = jcp.ks * OHW;

    parallel_nd(jcp.ic, [&](int ic) {
        const float *__restrict im_loc = im + ic * im_step;
        float *__restrict col_loc = col + ic * col_step;
        int id = od * jcp.stride_d - jcp.f_pad;
        for (int kd = 0; kd < jcp.kd; ++kd) {
            float *__restrict col_ = col_loc + kd * jcp.kh * jcp.kw * OHW;
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
                const float *__restrict im_ = im_loc + id * jcp.ih * jcp.iw;
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

/* col[ic][kh][kw][oh][ow] <-- im2col(im[ic][ih][iw]) */
void im2col(const jit_gemm_conv_conf_t &jcp, const float *__restrict im,
       float *__restrict col, int hs, int hb, int ws, int wb) {
    const size_t im_step = jcp.is;
    const size_t col_step = jcp.ks * hb * wb;
    if (jcp.stride_w == 1) {
        // Generated code is more optimized for stride_w == 1
        // because innermost loop is by width
        auto ker = [&](int ic, int kh, int kw, int oh) {
            const float *__restrict im_ = im + ic * im_step;
            float *__restrict col_
                = col + ic * col_step + ((kh * jcp.kw + kw) * hb + oh) * wb;

            const int ih = (oh + hs) * jcp.stride_h - jcp.t_pad
                + kh * (1 + jcp.dilate_h);
            if (ih < 0 || ih >= jcp.ih) {
                for (int ow = 0; ow < wb; ++ow)
                    col_[ow] = 0.f;
            } else {
                for (int ow = 0; ow < wb; ++ow) {
                    const int iw = ow + ws - jcp.l_pad + kw * (1 + jcp.dilate_w);
                    if (iw < 0 || iw >= jcp.iw)
                        col_[ow] = 0.f;
                    else {
                        const size_t im_idx = ih * jcp.iw + iw;
                        col_[ow] = im_[im_idx];
                    }
                }
            }
        };

        if (jcp.outer_threading) {
            for (int ic = 0; ic < jcp.ic; ic++)
                for (int kh = 0; kh < jcp.kh; kh++)
                    for (int kw = 0; kw < jcp.kw; kw++)
                        for (int oh = 0; oh < hb; oh++)
                            ker(ic, kh, kw, oh);
        }
        else {
            parallel_nd(jcp.ic, jcp.kh, jcp.kw, hb, ker);
        }
    } else if (jcp.ic == 1) {
        parallel_nd(jcp.kh, hb, [&](int kh, int oh) {
            const int ih = (oh + hs) * jcp.stride_h - jcp.t_pad
                    + kh * (1 + jcp.dilate_h);
            if (ih < 0 || ih >= jcp.ih)
                for (int kw = 0; kw < jcp.kw; ++kw) {
                    for (int ow = 0; ow < wb; ++ow) {
                        const size_t col_idx
                                = ((kh * jcp.kw + kw) * hb + oh) * wb + ow;
                        col[col_idx] = 0;
                    }
                }
            else
                for (int kw = 0; kw < jcp.kw; ++kw) {
                    for (int ow = 0; ow < wb; ++ow) {
                        const int iw = (ow + ws) * jcp.stride_w - jcp.l_pad
                                + kw * (1 + jcp.dilate_w);
                        const size_t col_idx
                                = ((kh * jcp.kw + kw) * hb + oh) * wb + ow;
                        const size_t im_idx = ih * jcp.iw + iw;
                        if (iw < 0 || iw >= jcp.iw)
                            col[col_idx] = 0;
                        else
                            col[col_idx] = im[im_idx];
                    }
                }
        });
    } else {

        parallel_nd(jcp.ic, jcp.kh, jcp.kw, hb,
            [&](int ic, int kh, int kw, int oh) {
            const float *__restrict im_ = im + ic * im_step;
            float *__restrict col_ = col + ic * col_step
                + ((kh * jcp.kw + kw) * hb + oh) * wb;

            const int ih = (oh + hs) * jcp.stride_h - jcp.t_pad
                + kh * (1 + jcp.dilate_h);
            if (ih < 0 || ih >= jcp.ih) {
                for (int ow = 0; ow < wb; ++ow)
                    col_[ow] = 0.f;
            } else {
                for (int ow = 0; ow < wb; ++ow) {
                    const int iw = (ow + ws) * jcp.stride_w - jcp.l_pad
                        + kw * (1 + jcp.dilate_w);
                    const size_t im_idx = ih * jcp.iw + iw;
                    if (iw < 0 || iw >= jcp.iw)
                        col_[ow] = 0.f;
                    else
                        col_[ow] = im_[im_idx];
                }
            }
        });
    }
}

/* col[oh][ow][kh][kw][ic] <-- im2col_u8(im[ih][iw][ic]) */
template <typename T>
void im2col_u8(const jit_gemm_conv_conf_t &jcp, const T *__restrict im,
        uint8_t *__restrict col) {
    uint8_t shift = jcp.signed_input ? 128 : 0;
    const int dh = 1 + jcp.dilate_h;
    const int dw = 1 + jcp.dilate_w;
    const int sh = jcp.stride_h;
    const int sw = jcp.stride_w;
    if (sh == 1 && sw == 1 && jcp.oh > 2 * mkldnn_get_max_threads()) {
        const int ihp = jcp.ih + jcp.t_pad;
        const int iwp = jcp.iw + jcp.l_pad;
        const int col_kw_step = jcp.ic;
        const int col_kh_step = jcp.kw * col_kw_step;
        const int col_ow_step = jcp.kh * col_kh_step;
        const int col_oh_step = jcp.ow * col_ow_step;
        const int im_iw_step = jcp.ngroups * jcp.ic;
        const int im_ih_step = jcp.iw * im_iw_step;

        const int nb_ic = jcp.ic / 4;
        const int ic_blocked = nb_ic * 4;

        parallel_nd(jcp.oh, [&](int oh) {
            const int kh_start = nstl::max(div_up(jcp.t_pad - oh, dh), 0);
            const int kh_end = nstl::min(div_up(ihp - oh, dh), jcp.kh);
            const int ih_start = oh - jcp.t_pad + kh_start * dh;
            const int col_oh_idx = oh * col_oh_step;

            for (int kh = kh_start, ih = ih_start; kh < kh_end; ++kh, ih += dh)
            {
                const int col_kh_idx = col_oh_idx + kh * col_kh_step;
                const int im_kh_idx = ih * im_ih_step;

                for (int kw = 0; kw < jcp.kw; ++kw) {
                    const int ow_start = nstl::max(jcp.l_pad - kw * dw, 0);
                    const int ow_end = nstl::min(iwp - kw * dw, jcp.ow);
                    const int iw_start = ow_start - jcp.l_pad + kw * dw;
                    const int col_kw_idx = col_kh_idx + kw * col_kw_step;

                    const int col_idx_start
                            = col_kw_idx + ow_start * col_ow_step;
                    const int im_idx_start = im_kh_idx + iw_start * im_iw_step;
                    const int col_idx_end = col_kw_idx + ow_end * col_ow_step;

                    // loop by iw and ow
                    if (nb_ic > 0) {
                        for (int col_idx = col_idx_start, im_idx = im_idx_start;
                                col_idx < col_idx_end;
                                col_idx += col_ow_step, im_idx += im_iw_step) {
                            for (int icb = 0; icb < 4 * nb_ic; icb += 4) {
                                PRAGMA_OMP_SIMD()
                                for (int ic = 0; ic < 4; ++ic) {
                                    col[col_idx + icb + ic]
                                            = im[im_idx + icb + ic] + shift;
                                }
                            }
                        }
                    }
                    if (ic_blocked != jcp.ic) {
                        for (int col_idx = col_idx_start, im_idx = im_idx_start;
                                col_idx < col_idx_end;
                                col_idx += col_ow_step, im_idx += im_iw_step) {
                            PRAGMA_OMP_SIMD()
                            for (int ic = ic_blocked; ic < jcp.ic; ++ic) {
                                col[col_idx + ic] = im[im_idx + ic] + shift;
                            }
                        }
                    }
                }
            }
        });
    }
    else {
        const size_t col_kh_step = jcp.kw * jcp.ic;
        const size_t col_ow_step = jcp.kh * col_kh_step;
        const size_t col_oh_step = jcp.ow * col_ow_step;
        const size_t im_ih_step = jcp.iw * jcp.ngroups * jcp.ic;
        const size_t im_iw_step = jcp.ngroups * jcp.ic;
        const int ih_pad = jcp.ih + jcp.t_pad;
        const int iw_pad = jcp.iw + jcp.l_pad;
        parallel_nd(jcp.oh, jcp.ow, [&](int oh, int ow) {
            const int ihs = oh * sh;
            const int ihsp = jcp.t_pad - ihs;
            const int kh_start = nstl::max(div_up(ihsp, dh), 0);
            const int kh_end = nstl::min(div_up(ih_pad - ihs, dh), jcp.kh);
            const int ih_start = kh_start * dh - ihsp;
            const int iws = ow * sw;
            const int iwsp = jcp.l_pad - iws;
            const int kw_start = nstl::max(div_up(iwsp, dw), 0);
            const int kw_end = nstl::min(div_up(iw_pad - iws, dw), jcp.kw);
            const int iw_start = kw_start * dw - iwsp;

            uint8_t *__restrict col_base
                    = col + oh * col_oh_step + ow * col_ow_step;
            for (int kh = kh_start, ih = ih_start; kh < kh_end;
                    ++kh, ih += dh) {
                uint8_t *__restrict col_ = col_base + kh * col_kh_step;
                const T *__restrict im_ = im + ih * im_ih_step;

                for (int kw = kw_start, iw = iw_start; kw < kw_end;
                    ++kw, iw += dw) {

                    const size_t col_idx = kw * jcp.ic;
                    const size_t im_idx = iw * im_iw_step;
                    PRAGMA_OMP_SIMD()
                        for (int ic = 0; ic < jcp.ic; ++ic) {
                            col_[col_idx + ic] = im_[im_idx + ic] + shift;
                        }
                }
            }
        });
    }

}

template void im2col_u8<int8_t>(const jit_gemm_conv_conf_t &jcp,
        const int8_t *__restrict im, uint8_t *__restrict col);
template void im2col_u8<uint8_t>(const jit_gemm_conv_conf_t &jcp,
        const uint8_t *__restrict im, uint8_t *__restrict col);

/* im[ih][iw][ic] <-- col2im_s32(col[oh][ow][kh][kw][ic]) */
void col2im_s32(const jit_gemm_conv_conf_t &jcp, const int32_t *__restrict col,
        int32_t *__restrict im)
{
    parallel(0, [&](const int ithr, const int nthr) {
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

                    const size_t col_idx = ((kh*jcp.kw + kw)*jcp.oh+oh)*jcp.ow+ow;
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
            const int ih = oh * jcp.stride_h - jcp.t_pad + kh * (1 + jcp.dilate_h);
            if (ih < 0 || ih >= jcp.ih) continue;

            for (int kw = 0; kw < jcp.kw; ++kw) {
            for (int ow = 0; ow < jcp.ow; ++ow) {
                const int iw = ow * jcp.stride_w - jcp.l_pad + kw * (1 + jcp.dilate_w);
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
        const memory_desc_wrapper &dst_d, int max_threads) {
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

    jcp.im2col_sz = !everyone_is(true,
            jcp.ow == jcp.iw, jcp.oh == jcp.ih, jcp.od == jcp.id,
            jcp.stride_w == 1, jcp.stride_h == 1, jcp.stride_d == 1,
            jcp.ks == 1, !jcp.signed_input)
        ? (ptrdiff_t)jcp.ic * jcp.ks * jcp.os : 0;

    jcp.outer_threading = false;
    jcp.oh_block = jcp.oh;
    jcp.ow_block = jcp.ow;

    bool is_int8_conv = utils::one_of(src_d.data_type(), s32, s8, u8)
        && weights_d.data_type() == s8;

    const int vlen = mayiuse(avx512_common)
        ? cpu_isa_traits<avx512_common>::vlen
        : mayiuse(avx)
            ? cpu_isa_traits<avx>::vlen
            : mayiuse(sse42) ? cpu_isa_traits<sse42>::vlen : 4;
    const int simd_w = vlen / (is_int8_conv ? 1 : 4);

    const bool is_bwd_d = jcp.prop_kind == backward_data;
    const bool is_bwd_w = jcp.prop_kind == backward_weights;
    const bool is_fwd = !is_bwd_d && !is_bwd_w;

    using namespace memory_tracking::names;
    //  For threading selection we do:
    //  1. Rough estimation of efficiency for inner and outer threading.
    //  2. Gemm size estimation in assumption that it does not work
    //  so effectively for small sizes.
    //  64K - this is heuristic gemm size per thread threshold.
    const int gemm_threshold = 64 * 1024;
    if (is_int8_conv) {
        bool is_depthwise = jcp.ic == 1 && jcp.oc == 1 && jcp.ngroups != 1;

        const int bs = is_fwd ? jcp.os : jcp.is;
        const int ls = is_fwd ? jcp.oc : jcp.ic;
        const size_t outer_work_amount = jcp.ngroups * jcp.mb;
        const float outer_thr_eff = (float)outer_work_amount
                / rnd_up(outer_work_amount, max_threads);
        const size_t inner_work_amount
                = div_up(bs, simd_w) * div_up(ls, simd_w);
        const float inner_thr_eff = (float)inner_work_amount
                / rnd_up(inner_work_amount, max_threads);
        jcp.outer_threading = (is_depthwise
                || (bs  / max_threads < 64 && jcp.mb != 1))
            && (outer_thr_eff / inner_thr_eff >= 1.f
                   || (bs * jcp.ic * jcp.oc) / max_threads < gemm_threshold);
        jcp.nthr = jcp.outer_threading ? max_threads : 1;

        if (is_fwd) {
            scratchpad.book(key_conv_gemm_col,
                    sizeof(int8_t) * jcp.nthr * jcp.im2col_sz);
            scratchpad.book(key_conv_int_dat_in_acc_dt,
                    sizeof(int32_t) * jcp.nthr * jcp.os * jcp.oc);
        } else if (is_bwd_d) {
            scratchpad.book(key_conv_gemm_col,
                    sizeof(int32_t) * jcp.nthr * jcp.im2col_sz);
            scratchpad.book(key_conv_int_dat_in_acc_dt,
                    sizeof(int32_t) * jcp.nthr * jcp.is * jcp.ic);
        } else if (is_bwd_w) {
            assert(!"unimplemented prop_kind");
            return status::unimplemented;
        }
    } else {
        if (is_fwd) {
            const int L2 = get_cache_size(2, true) / sizeof(float);
            const int wei_size = jcp.oc * jcp.ic * jcp.kh * jcp.kw;

            // It makes sense to try blocking for some special cases:
            // when weights size is small and we have to do im2col
            if (wei_size < L2/2 && jcp.im2col_sz && jcp.id == 1 && jcp.od == 1) {
                // looking for oh and ow blocking
                int h_block{ jcp.oh }, w_block{ jcp.ow };
                // 1. cache requirement
                // !!! used memory (assuming strides = 1 and dilate = 0 etc):
                const int row_size = jcp.ic * jcp.kh * jcp.kw * jcp.ow
                    + 2 * jcp.ic * jcp.iw + 2 * jcp.oc * jcp.ow;
                h_block = nstl::max(
                    1, nstl::min(jcp.oh, div_up(L2 - wei_size, row_size)));
                if (h_block == 1) {
                    const int col_size = jcp.ic * jcp.kh * jcp.kw + 2 * jcp.ic
                        + 2 * jcp.oc;
                    w_block = nstl::max(
                        1, nstl::min(jcp.ow, div_up(L2 - wei_size, col_size)));
                }

                // 2. threading requirement
                if (h_block != jcp.oh)
                    h_block = nstl::max(1, rnd_dn(h_block, 4));
                if (w_block != jcp.ow)
                    w_block = nstl::max(1, rnd_dn(w_block, simd_w));

                float thr_eff = 0.f;
                float thr_eff_treshold = 0.9f;
                if (w_block == jcp.ow) {
                    do {
                        int nb_oh = div_up(jcp.oh, h_block);
                        size_t work = jcp.ngroups * jcp.mb * jcp.od * nb_oh;
                        float disb = (float)jcp.oh / rnd_up(jcp.oh, h_block);
                        thr_eff = (float)work
                            / rnd_up(work, max_threads);
                        thr_eff = (thr_eff + disb) / 2.f;
                        if (thr_eff >= thr_eff_treshold)
                            break;
                        h_block = rnd_dn(h_block - 4, 4);
                    } while (h_block > 0);
                }
                if (thr_eff < thr_eff_treshold) // we didn't find suitable h_block
                {
                    h_block = 1;
                    int nb_oh = jcp.oh;
                    do {
                        int nb_ow = div_up(jcp.ow, w_block);
                        size_t work_amount
                            = jcp.ngroups * jcp.mb * jcp.od * nb_oh * nb_ow;
                        float disb = (float)jcp.ow / rnd_up(jcp.ow, w_block);
                        thr_eff = (float)work_amount
                            / rnd_up(work_amount, max_threads);
                        thr_eff = (thr_eff + disb) / 2.f;
                        if (thr_eff > thr_eff_treshold)
                            break;
                        w_block = rnd_dn(w_block - simd_w, simd_w);
                    } while (w_block > 0);
                }
                const size_t inner_work_amount
                    = div_up(jcp.os, simd_w) * div_up(jcp.oc, simd_w);
                const float inner_thr_eff = (float)inner_work_amount
                    / rnd_up(inner_work_amount, max_threads);
                if (thr_eff >= inner_thr_eff / 2 && h_block > 0 && w_block > 0) {
                    jcp.oh_block = h_block;
                    jcp.ow_block = w_block;
                    jcp.outer_threading = true;
                }
                // updating jcp.im2col_sz
                if (jcp.oh_block != 1)
                    jcp.ow_block = jcp.ow;
                jcp.im2col_sz
                    = (ptrdiff_t)jcp.ic * jcp.ks * jcp.oh_block * jcp.ow_block;
            } else {
                const size_t outer_work_amount = jcp.ngroups * jcp.mb * jcp.od;
                const float outer_thr_eff = (float)outer_work_amount
                        / rnd_up(outer_work_amount, max_threads);
                const size_t inner_work_amount
                        = div_up(jcp.os, simd_w) * div_up(jcp.oc, simd_w);
                const float inner_thr_eff = (float)inner_work_amount
                        / rnd_up(inner_work_amount, max_threads);
                jcp.outer_threading = jcp.os / max_threads < 512
                    && IMPLICATION(jcp.od == 1, jcp.mb != 1 || jcp.ngroups > 2)
                    && (outer_thr_eff / inner_thr_eff >= 1.f
                      || (jcp.os * jcp.ic * jcp.oc) / max_threads < gemm_threshold);
            }
        } else if (is_bwd_d) {
            const size_t outer_work_amount = jcp.ngroups * jcp.mb;
            const float outer_thr_eff = (float)outer_work_amount
                / rnd_up(outer_work_amount, max_threads);
            const size_t inner_work_amount
                = div_up(jcp.is, simd_w) * div_up(jcp.ic, simd_w);
            const float inner_thr_eff = (float)inner_work_amount
                / rnd_up(inner_work_amount, max_threads);
            jcp.outer_threading = (jcp.os / max_threads < 512 || jcp.ks < 64)
                && (jcp.mb != 1 || jcp.ngroups > 2)
                && (outer_thr_eff / inner_thr_eff >= 1.f
                  || (jcp.os * jcp.ic * jcp.oc) / max_threads < gemm_threshold);
        } else if (is_bwd_w)
            jcp.outer_threading = jcp.os / max_threads < 256
                && (jcp.mb != 1 || jcp.ngroups > 2);

        jcp.nthr = jcp.outer_threading ? max_threads : 1;

        scratchpad.book(key_conv_gemm_col,
                sizeof(float) * jcp.nthr * jcp.im2col_sz);

        if (is_bwd_w) {
            jcp.need_wei_reduction = mkldnn_thr_syncable()
                ? jcp.mb != 1 && jcp.nthr != 1 : false;

            scratchpad.book(key_conv_wei_reduction,
                    sizeof(float) * jcp.nthr * jcp.ngroups * weights_d.size());
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
