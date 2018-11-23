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

#include <cstring>
#include "mkldnn_types.h"

#include "c_types_map.hpp"
#include "jit_sse42_convolution.hpp"
#include "mkldnn_thread.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;

template <bool with_relu>
void _jit_sse42_convolution_fwd_t<with_relu>::execute_forward() {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto bias = reinterpret_cast<const data_t *>(this->input_memory(2));
    auto dst = reinterpret_cast<data_t *>(this->memory());

    const memory_desc_wrapper src_d(conf_.src_pd());
    const memory_desc_wrapper dst_d(conf_.dst_pd());
    const memory_desc_wrapper weights_d(conf_.weights_pd(0));
    const memory_desc_wrapper bias_d(conf_.weights_pd(1));

    const auto &jcp = kernel_->jcp;
    int MB = conf_.MB();

    int ocb_work = div_up(jcp.nb_oc, jcp.nb_oc_blocking);
    const size_t work_amount = MB * jcp.ngroups * ocb_work * jcp.oh;

    if (conf_.want_padded_bias()) {
        for (int oc = 0; oc < jcp.oc_without_padding; ++oc)
            padded_bias_[oc] = bias[oc];
        bias = padded_bias_;
    }

    parallel(0, [&](const int ithr, const int nthr) {
        size_t start{ 0 }, end{ 0 };
        balance211(work_amount, nthr, ithr, start, end);

        int icbb = 0;
        while (icbb < jcp.nb_ic) {
            int icb_step = jcp.nb_ic_blocking;
            int icb_step_rem = jcp.nb_ic - icbb;
            if (icb_step_rem < jcp.nb_ic_blocking_max)
                icb_step = icb_step_rem;

            size_t n{0}, g{0}, ocbb{0}, oh{0};
            nd_iterator_init(start, n, jcp.mb, g, jcp.ngroups, ocbb, ocb_work,
                             oh, jcp.oh);
            for (size_t iwork = start; iwork < end; ++iwork) {
                int ocb = ocbb * jcp.nb_oc_blocking;
                int ocb_num = jcp.nb_oc_blocking;

                for (int icb = icbb; icb < icbb + icb_step; ++icb) {
                    jit_conv_call_s par_conv = {};

                    const int ij = oh * jcp.stride_h;
                    const int i_t_overflow = nstl::max(0, jcp.t_pad - ij);
                    const int i_b_overflow = nstl::max(jcp.ih, ij
                        + (jcp.kh-1) * (jcp.dilate_h+1) - jcp.t_pad+1) - jcp.ih;

                    const size_t _oc = g * jcp.nb_oc + ocb;
                    const size_t _ic = g * jcp.nb_ic + icb;

                    const int ih = nstl::max(ij - jcp.t_pad
                        + div_up(i_t_overflow,
                                 (jcp.dilate_h+1)) * (jcp.dilate_h + 1), 0);
                    par_conv.src = &src[src_d.blk_off(n,
                        jcp.ic == 3 ? 0 : _ic, ih, 0)];

                    par_conv.dst = &dst[dst_d.blk_off(n, _oc, oh, 0)];

                    const int wh = div_up(i_t_overflow, (jcp.dilate_h + 1));
                    par_conv.filt = &weights[conf_.with_groups()
                                        ? weights_d.blk_off(g, ocb,
                                            jcp.ic == 3 ? 0 : icb, wh, 0)
                                        : weights_d.blk_off(ocb,
                                            jcp.ic == 3 ? 0 : icb, wh, 0)];

                    if (icb == 0) {
                        if (bias)
                            par_conv.bias =
                                    &bias[bias_d.blk_off(_oc * jcp.oc_block)];
                        par_conv.flags |= FLAG_IC_FIRST;
                    }

                    if (icb + 1 == jcp.nb_ic) {
                        par_conv.flags |= FLAG_IC_LAST;
                    }

                    par_conv.oc_blocks =
                            nstl::min(ocb + ocb_num, jcp.nb_oc) - ocb;

                    par_conv.kw_padding = 0;
                    const int kh_padding = jcp.kh
                        - div_up(i_t_overflow, (jcp.dilate_h + 1))
                        - div_up(i_b_overflow, (jcp.dilate_h + 1));
                    par_conv.kh_padding = nstl::max(0, kh_padding);

                    par_conv.oc_off = _oc * jcp.oc_block * sizeof(float);

                    kernel_->jit_ker(&par_conv);
                }
                nd_iterator_step(n, MB, g, jcp.ngroups, ocbb, ocb_work,
                                 oh, jcp.oh);
            }
            icbb += icb_step;
        }
    });
}

template <bool with_relu>
void _jit_sse42_convolution_fwd_t<with_relu>::execute_forward_fusing() {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto bias = reinterpret_cast<const data_t *>(this->input_memory(2));
    auto dst = reinterpret_cast<data_t *>(this->memory());

    const memory_desc_wrapper src_d(conf_.src_pd());
    const memory_desc_wrapper weights_d(conf_.weights_pd(0));
    const memory_desc_wrapper bias_d(conf_.weights_pd(1));

    const auto &jcp = kernel_->jcp;
    const auto &jcp_dw = kernel_dw_->jcp;
    int MB = conf_.MB();

    auto dw_bias = jcp.dw_conv_biases;

    int ocb_work = div_up(jcp.nb_oc, jcp.nb_oc_blocking);
    const size_t work_amount = MB * jcp.ngroups * ocb_work * jcp.oh;

    auto ker = [&](const int ithr, const int nthr) {
        auto compute_row_gen = [&](float* ws_p, int n, int g, int ocb, int ocb_num, int oh, int num_rows) {
            for (int h = 0; h < num_rows; h++) {
                if ((oh + h) < 0 || (oh + h) >= jcp.oh) {
                    for (int chb = ocb; chb < ocb + ocb_num; chb++) {
                        memset(ws_p + (((oh + h) + 1) % jcp.dw_conv_ker_h) * jcp.ow * jcp.oc_block +
                               (chb - ocb) * jcp.dw_conv_ker_h * jcp.ow * jcp.oc_block, 0, jcp.ow * jcp.oc_block * sizeof(float));
                    }
                } else {
                    for (int icb = 0; icb < jcp.nb_ic; ++icb) {
                        auto par_conv = jit_conv_call_s();

                        const int ij = (oh + h) * jcp.stride_h;
                        const int i_t_overflow = nstl::max(0, jcp.t_pad - ij);
                        const int i_b_overflow = nstl::max(jcp.ih, ij
                                                                   + (jcp.kh - 1) * (jcp.dilate_h + 1) - jcp.t_pad +
                                                                   1) - jcp.ih;

                        const size_t _oc = g * jcp.nb_oc + ocb;
                        const size_t _ic = g * jcp.nb_ic + icb;

                        const int ih = nstl::max(ij - jcp.t_pad
                                                 + div_up(i_t_overflow,
                                                          (jcp.dilate_h + 1)) * (jcp.dilate_h + 1), 0);
                        par_conv.src = &src[src_d.blk_off(n,
                                                          jcp.ic == 3 ? 0 : _ic, ih, 0)];

                        par_conv.dst = &ws_p[(((oh + h) + 1) % jcp.dw_conv_ker_h) * jcp.ow *
                                             jcp.oc_block];

                        const int wh = div_up(i_t_overflow, (jcp.dilate_h + 1));
                        par_conv.filt = &weights[conf_.with_groups()
                                                 ? weights_d.blk_off(g, ocb,
                                                                     jcp.ic == 3 ? 0 : icb, wh, 0)
                                                 : weights_d.blk_off(ocb,
                                                                     jcp.ic == 3 ? 0 : icb, wh, 0)];

                        if (icb == 0) {
                            if (bias)
                                par_conv.bias =
                                        &bias[bias_d.blk_off(_oc * jcp.oc_block)];
                            par_conv.flags |= FLAG_IC_FIRST;
                        }

                        if (icb + 1 == jcp.nb_ic) {
                            par_conv.flags |= FLAG_IC_LAST;
                        }

                        par_conv.oc_blocks =
                                nstl::min(ocb + ocb_num, jcp.nb_oc) - ocb;

                        par_conv.kw_padding = 0;
                        const int kh_padding = jcp.kh
                                               - div_up(i_t_overflow, (jcp.dilate_h + 1))
                                               - div_up(i_b_overflow, (jcp.dilate_h + 1));
                        par_conv.kh_padding = nstl::max(0, kh_padding);

                        par_conv.oc_off = _oc * jcp.oc_block * sizeof(float);

                        kernel_->jit_ker(&par_conv);
                    }
                }
            }
        };

        auto compute_row_dw = [&](const float* ws_p, int n, int ocb, int ocb_num,
                                  int dst_idx) {
            for (int chb = ocb; chb < nstl::min(ocb + ocb_num, jcp.nb_oc); chb++) {
                auto par_conv_dw = jit_conv_call_s();

                par_conv_dw.src_row0 = &ws_p[(((dst_idx+1) - 1) % jcp_dw.kh) * jcp_dw.iw * jcp_dw.ch_block +
                                             (chb - ocb) * jcp_dw.kh * jcp_dw.iw * jcp_dw.ch_block];
                par_conv_dw.src_row1 = &ws_p[(((dst_idx+1) - 0) % jcp_dw.kh) * jcp_dw.iw * jcp_dw.ch_block +
                                             (chb - ocb) * jcp_dw.kh * jcp_dw.iw * jcp_dw.ch_block];
                par_conv_dw.src_row2 = &ws_p[(((dst_idx+1) + 1) % jcp_dw.kh) * jcp_dw.iw * jcp_dw.ch_block +
                                             (chb - ocb) * jcp_dw.kh * jcp_dw.iw * jcp_dw.ch_block];

                par_conv_dw.dst = &dst[n*jcp_dw.oc*jcp_dw.oh*jcp_dw.ow + chb*jcp_dw.ch_block*jcp_dw.oh*jcp_dw.ow +
                                       dst_idx/jcp_dw.stride_h*jcp_dw.ow*jcp_dw.ch_block];

                par_conv_dw.kh_padding = jcp_dw.kh;
                par_conv_dw.filt = &jcp.dw_conv_weights[chb * jcp_dw.kh * jcp_dw.kw * jcp_dw.ch_block];
                par_conv_dw.bias = &dw_bias[chb * jcp_dw.ch_block];
                par_conv_dw.ur_w = (size_t)(jcp_dw.ow);

                kernel_dw_->jit_ker(&par_conv_dw);
            }
        };

        size_t start{0}, end{0};
        balance211(work_amount, nthr, ithr, start, end);

        auto pbuf = dw_conv_buffer_ + ithr * dw_conv_buffer_size_;

        size_t n{0}, g{0}, ocbb{0}, oh{0};
        nd_iterator_init(start, n, MB, g, jcp.ngroups, ocbb, ocb_work,
                         oh, jcp.oh);
        for (size_t iwork = start; iwork < end; ++iwork) {
            int ocb = ocbb * jcp.nb_oc_blocking;
            int ocb_num = jcp.nb_oc_blocking;

            if (iwork == start || oh == 0) {
                compute_row_gen(pbuf, n, g, ocb, ocb_num, oh - 1, 2);
            } else {
                compute_row_gen(pbuf, n, g, ocb, ocb_num, oh, 1);
            }

            if (iwork > start && ((oh - 1) % jcp_dw.stride_h == 0) && oh > 0) {
                compute_row_dw(pbuf, n, ocb, ocb_num, oh - 1);
            }

            if ((iwork == end - 1 || (int) oh == jcp.oh - 1) && ((oh) % jcp_dw.stride_h == 0)) {
                compute_row_gen(pbuf, n, g, ocb, ocb_num, oh + 1, 1);
                compute_row_dw(pbuf, n, ocb, ocb_num, oh);
            }

            nd_iterator_step(n, MB, g, jcp.ngroups, ocbb, ocb_work,
                             oh, jcp.oh);
        }
    };

    if (conf_.want_padded_bias()) {
        for (int oc = 0; oc < jcp.oc_without_padding; ++oc)
            padded_bias_[oc] = bias[oc];
        bias = padded_bias_;

        for (int oc = 0; oc < jcp.oc_without_padding; ++oc)
            dw_padded_bias_[oc] = dw_bias[oc];
        dw_bias = dw_padded_bias_;
    }

    parallel(0, ker);
}

template void _jit_sse42_convolution_fwd_t<true>::execute_forward();
template void _jit_sse42_convolution_fwd_t<false>::execute_forward();
template void _jit_sse42_convolution_fwd_t<true>::execute_forward_fusing();
template void _jit_sse42_convolution_fwd_t<false>::execute_forward_fusing();

}
}
}
