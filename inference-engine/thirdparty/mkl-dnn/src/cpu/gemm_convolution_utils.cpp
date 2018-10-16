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

#include <cstring>
#include "mkldnn_types.h"

#include "c_types_map.hpp"
#include "utils.hpp"
#include "type_helpers.hpp"
#include "gemm_convolution_utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;

namespace jit_gemm_convolution_utils {

void im2col_3d(jit_gemm_conv_conf_t &jcp, const float *im, float *col, int od) {
    const size_t OHW = jcp.oh * jcp.ow;
    const size_t im_step = jcp.ih * jcp.iw * jcp.id;
    const size_t col_step = jcp.ks * OHW;

    #pragma omp parallel for
    for (int ic = 0; ic < jcp.ic; ++ic) {
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
    }
}

void im2col(
    jit_gemm_conv_conf_t &jcp, const float *im, float *col) {
//    const size_t im_step = jcp.ih * jcp.iw;
//    const size_t col_step = jcp.ks * jcp.os;

    auto im2col_1st = [&](const float *im, float *col) {
        const size_t work_amount = jcp.oh * jcp.kh;
        #pragma omp parallel
        {
            const int ithr = omp_get_thread_num();
            const int nthr = omp_get_num_threads();

            size_t start = 0, end = 0;
            int oh = 0, kh = 0;
            balance211(work_amount, nthr, ithr, start, end);
            nd_iterator_init(start, kh, jcp.kh, oh, jcp.oh);

            for (size_t iwork = start; iwork < end; ++iwork)
            {
                const int ih = oh * jcp.stride_h - jcp.t_pad + kh * (1 + jcp.dilate_h);
                if (ih < 0 || ih >= jcp.ih) {
                    nd_iterator_step(kh, jcp.kh, oh, jcp.oh);
                    continue;
                }

                for (int kw = 0; kw < jcp.kw; ++kw) {
                for (int ow = 0; ow < jcp.ow; ++ow) {
                    const int iw = ow * jcp.stride_w - jcp.l_pad + kw * (1 + jcp.dilate_w);
                    if (iw < 0 || iw >= jcp.iw) continue;

                    const size_t col_idx = ((kh*jcp.kw + kw)*jcp.oh+oh)*jcp.ow+ow;
                    const size_t im_idx = ih*jcp.iw + iw;
                    col[col_idx] = im[im_idx];
                }}
                nd_iterator_step(kh, jcp.kh, oh, jcp.oh);
            }
        }
    };

    auto im2col_common = [&](const float *im, float *col) {
        int kernel_h = jcp.kh;
        int kernel_w = jcp.kw;
        int dilate_h = jcp.dilate_h + 1;
        int dilate_w = jcp.dilate_w + 1;
        int stride_h = jcp.stride_h;
        int stride_w = jcp.stride_w;
        int pad_t = jcp.t_pad;
        int pad_l = jcp.l_pad;
        int pad_b = jcp.b_pad;
        int pad_r = jcp.r_pad;
        int ih = jcp.ih;
        int iw = jcp.iw;
        int ic = jcp.ic;

        int dil_kernel_h = (kernel_h - 1) * dilate_h + 1;
        int dil_kernel_w = (kernel_w - 1) * dilate_w + 1;
        int col_h = (ih + pad_t + pad_b - dil_kernel_h) / stride_h + 1;
        int col_w = (iw + pad_l + pad_r - dil_kernel_w) / stride_w + 1;
        int col_c = ic * kernel_h * kernel_w;

        #pragma omp parallel for schedule(static)
        for (int c = 0; c < col_c; ++c) {
            int w_offset = c % kernel_w;
            int h_offset = (c / kernel_w) % kernel_h;
            int im_c = c / kernel_h / kernel_w;

            const int im_h_start = h_offset * dilate_h - pad_t;
            const int im_w_start = w_offset * dilate_w - pad_l;
            for (int h = 0, im_h = im_h_start; h < col_h; ++h, im_h += stride_h) {
                const int col_offset = (c * col_h + h) * col_w;
                const int im_offset = (im_c * ih + im_h) * iw;

                for (int w = 0, im_w = im_w_start; w < col_w; ++w, im_w += stride_w) {
                    bool is_in_bounds = ((unsigned)im_h) < ((unsigned)ih) && ((unsigned)im_w) < ((unsigned)iw);
                    col[col_offset + w] = is_in_bounds ? im[im_offset + im_w] : 0.f;
                }
            }
        }
    };

    if (jcp.ic != 1) {
        im2col_common(im, col);
    } else {
        im2col_1st(im, col);
    }
}

/* col[oh][ow][kh][kw][ic] <-- im2col_u8(im[ih][iw][ic]) */
void im2col_u8(
    jit_gemm_conv_conf_t &jcp, const uint8_t *im, uint8_t *col) {
    int num_thr = (jcp.mb != 1) ? omp_get_max_threads() : 1;
    MAYBE_UNUSED(num_thr);
#   pragma omp parallel for collapse(2) num_threads(num_thr)
    for (int oh = 0; oh < jcp.oh; ++oh) {
        for (int ow = 0; ow < jcp.ow; ++ow) {
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
                        col[col_idx + ic] = im[im_idx + ic];
                    }
                }
            }
        }
    }
}

void col2im_3d(
    jit_gemm_conv_conf_t &jcp, const float *col, float *im, int od) {
    const size_t col_step = jcp.ks * jcp.os;
    const size_t im_step = jcp.ih * jcp.iw * jcp.id;

    int num_thr = (jcp.mb != 1) ? omp_get_max_threads() : 1;
    MAYBE_UNUSED(num_thr);
#pragma omp parallel for  num_threads(num_thr)
    for (int ic = 0; ic < jcp.ic; ++ic) {
        const float *col_ = col;
        int id = od * jcp.stride_d - jcp.f_pad;
        for (int kd = 0; kd < jcp.kd; ++kd) {
        if (id < 0 || id >= jcp.id) {
            col_ += jcp.kh * jcp.kw * jcp.os;
            id += (1 + jcp.dilate_d);
            continue;
        }
        float *im_ = im + id * jcp.ih * jcp.iw;

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
            }
            }
        }
        }
        col_ += jcp.kh * jcp.kw * jcp.os;
        id += (1 + jcp.dilate_d);
        }
        col += col_step;
        im += im_step;
    }
}

void col2im(
        jit_gemm_conv_conf_t &jcp, const float *col, float *im) {
    int kernel_h = jcp.kh;
    int kernel_w = jcp.kw;
    int dilate_h = jcp.dilate_h + 1;
    int dilate_w = jcp.dilate_w + 1;
    int stride_h = jcp.stride_h;
    int stride_w = jcp.stride_w;
    int pad_t = jcp.t_pad;
    int pad_l = jcp.l_pad;
    int pad_b = jcp.b_pad;
    int pad_r = jcp.r_pad;
    int ih = jcp.ih;
    int iw = jcp.iw;
    int ic = jcp.ic;

    int dil_patch_h = (kernel_h - 1) * dilate_h + 1;
    int dil_patch_w = (kernel_w - 1) * dilate_w + 1;
    int col_h = (ih + pad_t + pad_b - dil_patch_h) / stride_h + 1;
    int col_w = (iw + pad_l + pad_r - dil_patch_w) / stride_w + 1;

    memset(im, 0, ih * iw * ic * sizeof(float));

#pragma omp parallel for schedule(static)
    for (int i = 0; i < ic; ++i) {
        for (int inner_idx = 0; inner_idx < kernel_h*kernel_w; ++inner_idx) {
            int c = i*kernel_h*kernel_w + inner_idx;
            int w_offset = c % kernel_w;
            int h_offset = (c / kernel_w) % kernel_h;
            int c_im = c / kernel_h / kernel_w;

            const int im_h_start = h_offset * dilate_h - pad_t;
            const int im_w_start = w_offset * dilate_w - pad_l;
            for (int h = 0, im_h = im_h_start; h < col_h; ++h, im_h += stride_h) {
                const int im_offset = (c_im * ih + im_h) * iw;
                const int col_offset = (c * col_h + h) * col_w;

                for (int w = 0, im_w = im_w_start; w < col_w; ++w, im_w += stride_w) {
                    bool is_in_bound = (((unsigned)im_h) < ((unsigned)ih) && ((unsigned)im_w) < ((unsigned)iw));
                    if (is_in_bound)
                        im[im_offset + im_w] += col[col_offset + w];
                }
            }
        }
    }
}

void init_conf(
    jit_gemm_conv_conf_t &jcp, const convolution_desc_t &cd,
    const memory_desc_wrapper &src_d, const memory_desc_wrapper &weights_d,
    const memory_desc_wrapper &dst_d,
    bool with_relu, float relu_negative_slope) {

    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    jcp.prop_kind = cd.prop_kind;
    const int ndims = src_d.ndims();

    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];

    jcp.oc = dst_d.dims()[1] / jcp.ngroups;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;

    jcp.id = (ndims == 4) ? 1 : src_d.dims()[2];
    jcp.ih = src_d.dims()[ndims - 2];
    jcp.iw = src_d.dims()[ndims - 1];
    jcp.od = (ndims == 4) ? 1 : dst_d.dims()[2];
    jcp.oh = dst_d.dims()[ndims - 2];
    jcp.ow = dst_d.dims()[ndims - 1];

    jcp.kd = (ndims == 4) ? 1 : weights_d.dims()[with_groups + 2];
    jcp.kh = weights_d.dims()[with_groups + ndims - 2];
    jcp.kw = weights_d.dims()[with_groups + ndims - 1];

    jcp.f_pad = (ndims == 4) ? 0 : cd.padding[0][0];
    jcp.t_pad = cd.padding[0][ndims - 4];
    jcp.l_pad = cd.padding[0][ndims - 3];
    jcp.b_pad = cd.padding[1][ndims - 4];
    jcp.r_pad = cd.padding[1][ndims - 3];

    jcp.stride_d = (ndims == 4) ? 1 : cd.strides[0];
    jcp.stride_h = cd.strides[ndims - 4];
    jcp.stride_w = cd.strides[ndims - 3];

    jcp.dilate_d = (ndims == 4) ? 0 : cd.dilates[0];
    jcp.dilate_h = cd.dilates[ndims - 4];
    jcp.dilate_w = cd.dilates[ndims - 3];

    jcp.src_fmt = src_d.format();
    jcp.with_bias
        = cd.bias_desc.format != memory_format::undef
        || cd.diff_bias_desc.format != memory_format::undef;
    jcp.with_relu = with_relu;
    jcp.relu_negative_slope = relu_negative_slope;

    jcp.os = jcp.oh * jcp.ow;
    jcp.ks = jcp.kh * jcp.kw * jcp.kd;
    jcp.need_im2col = !(jcp.oh == jcp.ih && jcp.ow == jcp.iw
        && jcp.od == jcp.id && jcp.ks == 1);
}

template <typename src_t>
status_t prepare_ws_col(jit_gemm_conv_conf_t &jcp, src_t **col, const int nthr) {
    if (!jcp.need_im2col) {
        *col = nullptr;
        return status::success;
    }
    const ptrdiff_t im2col_sz_per_thr = (ptrdiff_t)jcp.os * jcp.ks * jcp.ic;
    const ptrdiff_t im2col_sz = nthr * im2col_sz_per_thr;
    *col = (src_t *)malloc(im2col_sz * sizeof(src_t), 64);
    if (*col == nullptr) return status::out_of_memory;

#   pragma omp parallel for
    for (ptrdiff_t i = 0; i < im2col_sz; ++i)
        (*col)[i] = (src_t)0;

    return status::success;
}

template status_t prepare_ws_col<float>(jit_gemm_conv_conf_t &jcp,
        float **col, const int nthr);
template status_t prepare_ws_col<uint8_t>(jit_gemm_conv_conf_t &jcp,
        uint8_t **col, const int nthr);

status_t prepare_ws_wei_reduction(jit_gemm_conv_conf_t &jcp,
        float **wei_reduction, size_t wei_sz, const int nthr) {
    if (jcp.mb == 1 || nthr == 1)
        return status::success;

    const size_t sz_per_thr = jcp.ngroups * wei_sz; // XXX: why groups?
    *wei_reduction = (float *)malloc(nthr * sz_per_thr, 64);
    if (*wei_reduction == nullptr) return status::out_of_memory;

    return status::success;
}

template <typename acc_t>
status_t prepare_ws_acc(jit_gemm_conv_conf_t &jcp, acc_t **acc, const int nthr) {
    const size_t acc_sz_per_thr = jcp.os * jcp.oc;
    const size_t acc_sz = nthr * acc_sz_per_thr;

    *acc = (int32_t *)malloc(acc_sz * sizeof(acc_t), 64);
    if (*acc == nullptr) return status::out_of_memory;
    return status::success;
}

template status_t prepare_ws_acc<int32_t>(jit_gemm_conv_conf_t &jcp,
        int32_t **acc, const int nthr);

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
