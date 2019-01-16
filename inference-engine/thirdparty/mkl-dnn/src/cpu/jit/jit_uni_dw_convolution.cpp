/*******************************************************************************
* Copyright 2018 Intel Corporation
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
#include "jit_uni_dw_convolution.hpp"
#include "mkldnn_thread.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;

template <cpu_isa_t isa, bool with_relu>
void _jit_uni_dw_convolution_fwd_t<isa, with_relu>::execute_forward() {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto bias = reinterpret_cast<const data_t *>(this->input_memory(2));
    auto dst = reinterpret_cast<data_t *>(this->memory());

    const memory_desc_wrapper src_d(conf_.src_pd());
    const memory_desc_wrapper dst_d(conf_.dst_pd());
    const memory_desc_wrapper weights_d(conf_.weights_pd(0));
    const memory_desc_wrapper bias_d(conf_.weights_pd(1));

    const auto &jcp = kernel_->jcp;

    if (conf_.want_padded_bias()) {
        for (int oc = 0; oc < jcp.oc_without_padding; ++oc)
            padded_bias_[oc] = bias[oc];
        bias = padded_bias_;
    }

    int dil_h = jcp.dilate_h + 1;
    int dil_w = jcp.dilate_w + 1;
    int str_h = jcp.stride_h;
    int str_w = jcp.stride_w;

    auto kernel_params = [&](int ur_w_step, int ow, int oh, int ih, int kh,
            int kh_padding, int ch, int ch_num, int n) {
        auto par_conv = jit_conv_call_s();

        const int i_l_overflow = nstl::max(0, (jcp.l_pad - ow * str_w));
        const int i_r_overflow = nstl::max(jcp.iw, (ow * str_w
            + (jcp.kw - 1)*dil_w - jcp.l_pad + 1)) - jcp.iw;

        const int iw = nstl::max((ow*str_w - jcp.l_pad
            + div_up(i_l_overflow, dil_w)*dil_w), 0);
        const int kw = div_up(i_l_overflow, dil_w);

        const int kw_padding = jcp.kw - div_up(i_l_overflow, dil_w)
            - div_up(i_r_overflow, dil_w);

        par_conv.src = &src[src_d.blk_off(n, ch, ih, iw)];
        par_conv.dst = &dst[dst_d.blk_off(n, ch, oh, ow)];

        par_conv.filt = &weights[weights_d.blk_off(ch, 0, 0, kh, kw)];
        if (bias) par_conv.bias = &bias[bias_d.blk_off(ch*jcp.ch_block)];

        par_conv.kh_padding = (size_t)nstl::max(0, kh_padding);
        par_conv.kw_padding = (size_t)nstl::max(0, kw_padding);

        par_conv.ur_w = (size_t)ur_w_step;

        par_conv.ch_blocks = nstl::min(ch + ch_num, jcp.nb_ch) - ch;
        par_conv.oc_off = ch * jcp.ch_block * sizeof(float);

        return par_conv;
    };

    int MB = conf_.MB();
    const int chb_work = utils::div_up(jcp.nb_ch, jcp.nb_ch_blocking);
    parallel_nd(MB, chb_work, jcp.oh,
            [&](int n, int chb, int oh) {
        int ch = chb * jcp.nb_ch_blocking;
        int ch_num = jcp.nb_ch_blocking;

        const int i_t_overflow = nstl::max(0, (int)(jcp.t_pad - oh*str_h));
        const int i_b_overflow = nstl::max(jcp.ih,
            (int)(oh*str_h + (jcp.kh - 1)*dil_h - jcp.t_pad + 1)) - jcp.ih;

        const int ih = nstl::max((int)(oh*str_h - jcp.t_pad
            + div_up(i_t_overflow, dil_h)*dil_h), 0);
        const int kh = div_up(i_t_overflow, dil_h);
        const int kh_padding = jcp.kh - div_up(i_t_overflow, dil_h)
            - div_up(i_b_overflow, dil_h);

        // left border
        int ow = 0;
        int l_border = nstl::min(div_up(jcp.l_pad, str_w), jcp.ow);
        int ur_w_step = 1;
        for (; ow < l_border; ow++) {
            jit_conv_call_s par_conv = kernel_params(ur_w_step, ow, oh, ih,
                                        kh, kh_padding, ch, ch_num, n);

            kernel_->jit_ker(&par_conv);
        }

        // main loop
        ur_w_step = (jcp.iw - (jcp.kw - 1)*dil_w + jcp.l_pad - 1)
            / jcp.stride_w - ow + 1;
        if (ur_w_step > 0) {
            jit_conv_call_s par_conv = kernel_params(ur_w_step, ow, oh, ih,
                                        kh, kh_padding, ch, ch_num, n);

            kernel_->jit_ker(&par_conv);

            ow += ur_w_step;
        }

        // right border
        ur_w_step = 1;
        for (; ow < jcp.ow; ow++) {
            jit_conv_call_s par_conv = kernel_params(ur_w_step, ow, oh, ih,
                                        kh, kh_padding, ch, ch_num, n);

            kernel_->jit_ker(&par_conv);
        }
    });
}

template void _jit_uni_dw_convolution_fwd_t<avx512_common, false>
    ::execute_forward();
template void _jit_uni_dw_convolution_fwd_t<avx2, false>
    ::execute_forward();
template void _jit_uni_dw_convolution_fwd_t<sse42, false>
    ::execute_forward();

template void _jit_uni_dw_convolution_fwd_t<avx512_common, true>
    ::execute_forward();
template void _jit_uni_dw_convolution_fwd_t<avx2, true>
    ::execute_forward();
template void _jit_uni_dw_convolution_fwd_t<sse42, true>
    ::execute_forward();

template <cpu_isa_t isa>
void _jit_uni_dw_convolution_bwd_data_t<isa>::execute_backward_data() {
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto diff_src = reinterpret_cast<data_t *>(this->memory());

    const memory_desc_wrapper diff_dst_d(conf_.diff_dst_pd());
    const memory_desc_wrapper diff_src_d(conf_.diff_src_pd());
    const memory_desc_wrapper weights_d(conf_.weights_pd(0));

    const auto &jcp = kernel_->jcp;

    auto kernel_params = [&](int ur_str_w, int iw, int oh, int ih,
            int i_t_overflow, int i_b_overflow, int stride_off_h,
            int ch, int ch_num, int n) {
        auto par_conv = jit_conv_call_s();

        const int i_l_overflow = nstl::max(0, (jcp.kw - 1 - iw - jcp.l_pad));
        const int i_r_overflow = nstl::max(0, (jcp.kw - 1 - (jcp.iw - 1 - iw)
            - jcp.r_pad));

        int ow = iw + jcp.l_pad - i_r_overflow;
        int stride_off_w = ow % jcp.stride_w;
        ow /= jcp.stride_w;

        par_conv.src = &diff_src[diff_src_d.blk_off(n, ch, ih, iw)];
        par_conv.dst = &diff_dst[diff_dst_d.blk_off(n, ch, oh, ow)];
        par_conv.filt = &weights[weights_d.blk_off(ch, 0, 0, i_b_overflow
            + stride_off_h, i_r_overflow + stride_off_w)];

        par_conv.kh_padding = nstl::max(0, jcp.kh - i_t_overflow - i_b_overflow
            - stride_off_h);
        par_conv.kw_padding = nstl::max(0, jcp.kw - i_l_overflow - i_r_overflow
            - stride_off_w);

        par_conv.ur_str_w = ur_str_w;

        par_conv.ch_blocks = nstl::min(ch + ch_num, jcp.nb_ch) - ch;

        return par_conv;
    };

    int MB = conf_.MB();
    const int chb_work = utils::div_up(jcp.nb_ch, jcp.nb_ch_blocking);
    parallel_nd(MB, chb_work, jcp.ih,
        [&](int n, int chb, int ih) {
        int ch = chb * jcp.nb_ch_blocking;
        int ch_num = jcp.nb_ch_blocking;

        const int i_t_overflow = nstl::max(0, (int)(jcp.kh - 1 - ih
            - jcp.t_pad));
        const int i_b_overflow = nstl::max(0, (int)(jcp.kh - 1
            - (jcp.ih - 1 - ih) - jcp.b_pad));

        int oh = ih + jcp.t_pad - i_b_overflow;
        int stride_off_h = oh % jcp.stride_h;
        oh /= jcp.stride_h;

        for (int i_str_w = 0; i_str_w < jcp.stride_w; i_str_w++) {
            // left border
            int iw = i_str_w;
            int l_border = nstl::min(jcp.kw - 1 - jcp.l_pad, jcp.iw);
            int ur_str_w = 1;
            for (; iw < l_border; iw += jcp.stride_w) {
                jit_conv_call_s par_conv = kernel_params(ur_str_w, iw, oh,
                                             ih, i_t_overflow, i_b_overflow,
                                             stride_off_h, ch, ch_num, n);

                kernel_->jit_ker(&par_conv);
            }

            // main loop
            ur_str_w = nstl::min((jcp.iw - jcp.kw + jcp.r_pad - iw)
                 / jcp.stride_w, jcp.iw);
            if (ur_str_w > 0) {
                jit_conv_call_s par_conv = kernel_params(ur_str_w, iw, oh,
                                             ih, i_t_overflow, i_b_overflow,
                                             stride_off_h, ch, ch_num, n);

                kernel_->jit_ker(&par_conv);

                iw += ur_str_w * jcp.stride_w;
            }

            // right border
            ur_str_w = 1;
            for (; iw < jcp.iw; iw += jcp.stride_w) {
                jit_conv_call_s par_conv = kernel_params(ur_str_w, iw, oh,
                                             ih, i_t_overflow, i_b_overflow,
                                             stride_off_h, ch, ch_num, n);

                kernel_->jit_ker(&par_conv);
            }
        }
    });
}

template void _jit_uni_dw_convolution_bwd_data_t<avx512_common>
    ::execute_backward_data();
template void _jit_uni_dw_convolution_bwd_data_t<avx2>
    ::execute_backward_data();
template void _jit_uni_dw_convolution_bwd_data_t<sse42>
    ::execute_backward_data();

}
}
}
