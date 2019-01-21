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

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;
using namespace prop_kind;
using namespace data_type;

namespace jit_gemm_convolution_utils {

void im2col_3d(jit_gemm_conv_conf_t &jcp, const float *im, float *col, int od) {
    const size_t OHW = jcp.oh * jcp.ow;
    const size_t im_step = jcp.ih * jcp.iw * jcp.id;
    const size_t col_step = jcp.ks * OHW;

    parallel_nd(jcp.ic, [&](int ic) {
        const float *im_loc = im + ic * im_step;
        float *col_loc = col + ic * col_step;
        int id = od * jcp.stride_d - jcp.f_pad;
        for (int kd = 0; kd < jcp.kd; ++kd) {
            float *col_ = col_loc + kd * jcp.kh * jcp.kw * OHW;
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
                const float *im_ = im_loc + id * jcp.ih * jcp.iw;
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

void im2col(jit_gemm_conv_conf_t &jcp, const float *im, float *col) {
    if (jcp.ic == 1) {
        parallel_nd(jcp.kh, jcp.oh, [&](int kh, int oh) {
            const int ih = oh * jcp.stride_h - jcp.t_pad + kh * (1 + jcp.dilate_h);
            if (ih < 0 || ih >= jcp.ih) return;

            for (int kw = 0; kw < jcp.kw; ++kw) {
            for (int ow = 0; ow < jcp.ow; ++ow) {
                const int iw = ow * jcp.stride_w - jcp.l_pad + kw * (1 + jcp.dilate_w);
                if (iw < 0 || iw >= jcp.iw) continue;

                const size_t col_idx = ((kh*jcp.kw + kw)*jcp.oh+oh)*jcp.ow+ow;
                const size_t im_idx = ih*jcp.iw + iw;
                col[col_idx] = im[im_idx];
            }}
        });
    } else {
        const size_t im_step = jcp.ih * jcp.iw;
        const size_t col_step = jcp.ks * jcp.os;

        parallel_nd(jcp.ic, [&](int ic) {
            const float *im_ = im + ic * im_step;
            float *col_ = col + ic * col_step;

            for (int kh = 0; kh < jcp.kh; ++kh) {
            for (int oh = 0; oh < jcp.oh; ++oh) {
                const int ih = oh * jcp.stride_h
                               - jcp.t_pad + kh * (1 + jcp.dilate_h);
                if (ih < 0 || ih >= jcp.ih) continue;

                for (int kw = 0; kw < jcp.kw; ++kw) {
                for (int ow = 0; ow < jcp.ow; ++ow) {
                    const int iw = ow * jcp.stride_w
                                   - jcp.l_pad + kw * (1 + jcp.dilate_w);
                    if (iw < 0 || iw >= jcp.iw) continue;

                    const size_t col_idx = ((kh * jcp.kw + kw) * jcp.oh+oh)
                                           * jcp.ow + ow;
                    const size_t im_idx = ih*jcp.iw + iw;
                    col_[col_idx] = im_[im_idx];
                }}
            }}
        });
    }
}

/* col[oh][ow][kh][kw][ic] <-- im2col_u8(im[ih][iw][ic]) */
template <typename T>
void im2col_u8(jit_gemm_conv_conf_t &jcp, const T *im, uint8_t *col) {
    parallel_nd(jcp.oh, jcp.ow, [&](int oh, int ow) {
            for (int kh = 0; kh < jcp.kh; ++kh) {
                const int ih = oh * jcp.stride_h
                    - jcp.t_pad + kh * (1 + jcp.dilate_h);
                if (ih < 0 || ih >= jcp.ih) continue;

                for (int kw = 0; kw < jcp.kw; ++kw) {
                    const int iw = ow * jcp.stride_w
                        - jcp.l_pad + kw * (1 + jcp.dilate_w);
                    if (iw < 0 || iw >= jcp.iw) continue;

                    const size_t col_idx = (((oh * jcp.ow + ow) * jcp.kh + kh)
                            * jcp.kw + kw) * jcp.ic;
                    const size_t im_idx
                        = (ih * jcp.iw + iw) * jcp.ngroups * jcp.ic;
                    PRAGMA_OMP_SIMD()
                    for (int ic = 0; ic < jcp.ic; ++ic) {
                        col[col_idx + ic] = jcp.signed_input
                        ? im[im_idx + ic] + 128
                        : im[im_idx + ic];
                    }
                }
            }
        }
    );
}
template void im2col_u8<int8_t>(
        jit_gemm_conv_conf_t &jcp, const int8_t *im, uint8_t *col);
template void im2col_u8<uint8_t>(
        jit_gemm_conv_conf_t &jcp, const uint8_t *im, uint8_t *col);

/* im[ih][iw][ic] <-- col2im_s32(col[oh][ow][kh][kw][ic]) */
void col2im_s32(jit_gemm_conv_conf_t &jcp, const int32_t *col, int32_t *im) {
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

void col2im_3d(jit_gemm_conv_conf_t &jcp, const float *col, float *im, int od) {
    parallel_nd(jcp.ic, [&](int ic) {
        const float *col_ = col + (size_t)ic * jcp.ks * jcp.os;
        float *im_ic = im + (size_t)ic * jcp.ih * jcp.iw * jcp.id;

        int id = od * jcp.stride_d - jcp.f_pad;
        for (int kd = 0; kd < jcp.kd; ++kd) {
            if (id < 0 || id >= jcp.id) {
                col_ += jcp.kh * jcp.kw * jcp.os;
                id += (1 + jcp.dilate_d);
                continue;
            }

            float *im_ = im_ic + id * jcp.ih * jcp.iw;

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

void col2im(
    jit_gemm_conv_conf_t &jcp, const float *col, float *im) {

    const size_t col_step = jcp.ks * jcp.os;
    const size_t im_step = jcp.ih * jcp.iw;
    const int iS = jcp.ih * jcp.iw;

    parallel_nd(jcp.ic, [&](int ic) {
        float *im_ = im + ic * im_step;
        const float *col_ = col + ic * col_step;
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

void init_conf(
    jit_gemm_conv_conf_t &jcp, const convolution_desc_t &cd,
    const memory_desc_wrapper &src_d, const memory_desc_wrapper &weights_d,
    const memory_desc_wrapper &dst_d, int max_threads,
    bool with_relu, float relu_negative_slope) {

    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    jcp.prop_kind = cd.prop_kind;
    const int ndims = src_d.ndims();
    const int is_1d = ndims == 3;
    const int is_3d = ndims == 5;

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
    jcp.with_bias
        = cd.bias_desc.format != memory_format::undef
        || cd.diff_bias_desc.format != memory_format::undef;
    jcp.with_relu = with_relu;
    jcp.relu_negative_slope = relu_negative_slope;

    jcp.is = jcp.ih * jcp.iw;
    jcp.os = jcp.oh * jcp.ow;
    jcp.ks = jcp.kh * jcp.kw * jcp.kd;

    jcp.signed_input = (src_d.data_type() == data_type::s8);
    jcp.wei_adj_scale = (!jcp.signed_input || mayiuse(avx512_core_vnni))
            ? 1.0f
            : (1.0f / 2.0f);
    jcp.im2col_sz = !everyone_is(true,
            jcp.ow == jcp.iw, jcp.oh == jcp.ih, jcp.od == jcp.id,
            jcp.stride_w == 1, jcp.stride_h == 1, jcp.stride_d == 1,
            jcp.ks == 1, !jcp.signed_input)
        ? (ptrdiff_t)jcp.ic * jcp.ks * jcp.os
        : 0;

    bool do_outer_threading = false;
    bool is_int8_conv
            = (utils::one_of(cd.src_desc.data_type == u8, cd.src_desc.data_type == s8)
                    && cd.weights_desc.data_type == s8);
    if (is_int8_conv) {
        bool is_depthwise =
                utils::everyone_is(1, jcp.ic, jcp.oc) && jcp.ngroups != 1;
        do_outer_threading
                = (is_depthwise || (jcp.os / max_threads < 64 && jcp.mb != 1));
    } else {
        if (utils::one_of(jcp.prop_kind, forward_training, forward_inference))
            do_outer_threading = jcp.os / max_threads < 512
                && IMPLICATION(jcp.od == 1, (jcp.mb != 1 || jcp.ngroups > 2));
        else if (jcp.prop_kind == backward_data)
            do_outer_threading = (jcp.mb != 1 || jcp.ngroups > 2);
        else //(jcp.prop_kind == backward_weights)
            do_outer_threading = jcp.os / max_threads < 256
                       && (jcp.mb != 1 || jcp.ngroups > 2);
    }
    jcp.nthr = do_outer_threading ? max_threads : 1;
    jcp.need_wei_reduction = mkldnn_thr_syncable()
        ? (jcp.mb != 1 && jcp.nthr != 1) : false;
}

status_t prepare_scratchpad(jit_gemm_conv_conf_t &jcp,
                scratchpad_t **scratchpad_, size_t size, const int nthr) {
    if (size > 0) {
        *scratchpad_ = create_scratchpad(nthr * size);
        if (*scratchpad_ == nullptr) return status::out_of_memory;
    } else {
        *scratchpad_ = nullptr;
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

void bwd_weights_reduction_par(int ithr, int nthr, const jit_gemm_conv_conf_t &jcp,
        const float *weights_reduce_ws, float *weights) {
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
