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
#include "jit_uni_x8s8s32x_dw_convolution.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;

template <cpu_isa_t isa, bool with_relu, data_type_t src_type, data_type_t dst_type>
void _jit_uni_x8s8s32x_dw_convolution_fwd_t<isa, with_relu, src_type, dst_type>::execute_forward() {
    auto src = reinterpret_cast<const src_data_t*>(this->input_memory(0));
    auto weights = reinterpret_cast<const wei_data_t*>(this->input_memory(1));
    auto bias = reinterpret_cast<const char*>(this->input_memory(2));
    auto dst = reinterpret_cast<dst_data_t*>(this->memory());

    const memory_desc_wrapper src_d(conf_.src_pd());
    const memory_desc_wrapper dst_d(conf_.dst_pd());
    const memory_desc_wrapper weights_d(conf_.weights_pd(0));
    const memory_desc_wrapper bias_d(conf_.weights_pd(1));

    const auto &jcp = kernel_->jcp;

    int dil_h = jcp.dilate_h + 1;
    int dil_w = jcp.dilate_w + 1;
    int str_h = jcp.stride_h;
    int str_w = jcp.stride_w;

    const size_t bia_dt_size = conf_.with_bias()
        ? types::data_type_size(conf_.cdesc()->bias_desc.data_type) : 0;

    const auto &oscales = conf_.attr()->output_scales_;

    int MB = jcp.mb;
    int chb_work = utils::div_up(jcp.nb_ch, jcp.nb_ch_blocking);
    const size_t work_amount = MB * chb_work * jcp.oh;

    auto kernel_params = [&](int ur_w_step, int ow, int oh, int ih, int kh,
            int kh_padding, int ch, int ch_num, int n) {
        jit_conv_call_s par_conv = {};

        const int i_l_overflow = nstl::max(0, (jcp.l_pad - ow * str_w));
        const int i_r_overflow = nstl::max(jcp.iw, (ow * str_w
            + (jcp.kw - 1)*dil_w - jcp.l_pad + 1)) - jcp.iw;

        const int iw = nstl::max((ow*str_w - jcp.l_pad
            + div_up(i_l_overflow, dil_w)*dil_w), 0);
        const int kw = div_up(i_l_overflow, dil_w);

        const int kw_padding = jcp.kw - div_up(i_l_overflow, dil_w)
            - div_up(i_r_overflow, dil_w);

        int src_off = src_d.blk_off(n, ch*jcp.ch_block, ih, iw);
        int dst_off = dst_d.blk_off(n, ch*jcp.ch_block, oh, ow);

        par_conv.src = &src[src_off];
        par_conv.dst = &dst[dst_off];

        par_conv.filt = &weights[weights_d.blk_off(ch, 0, 0, kh, kw)];
        if (bias) par_conv.bias = &bias[bias_d.blk_off(ch*jcp.ch_block*bia_dt_size)];

        par_conv.kh_padding = (size_t)nstl::max(0, kh_padding);
        par_conv.kw_padding = (size_t)nstl::max(0, kw_padding);

        par_conv.ur_w = (size_t)ur_w_step;

        par_conv.ch_work = nstl::min((ch + ch_num) * jcp.ch_block, jcp.oc) - ch*jcp.ch_block;

        par_conv.scales = &oscales.scales_[jcp.is_oc_scale * ch * jcp.ch_block];

        return par_conv;
    };

    auto ker = [&](const int ithr, const int nthr) {
        size_t start{0}, end{0};
        balance211(work_amount, nthr, ithr, start, end);

        size_t n{0}, chb{0}, oh{0};
        nd_iterator_init(start, n, MB, chb, chb_work, oh, jcp.oh);
        for (size_t iwork = start; iwork < end; ++iwork) {
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

            nd_iterator_step(n, MB, chb, chb_work, oh, jcp.oh);
        }
    };

    parallel(0, ker);
}

template void _jit_uni_x8s8s32x_dw_convolution_fwd_t<avx2, true, data_type::u8, data_type::u8>::execute_forward();
template void _jit_uni_x8s8s32x_dw_convolution_fwd_t<avx2, true, data_type::u8, data_type::s8>::execute_forward();
template void _jit_uni_x8s8s32x_dw_convolution_fwd_t<avx2, true, data_type::u8, data_type::s32>::execute_forward();
template void _jit_uni_x8s8s32x_dw_convolution_fwd_t<avx2, true, data_type::u8, data_type::f32>::execute_forward();
template void _jit_uni_x8s8s32x_dw_convolution_fwd_t<avx2, false, data_type::u8, data_type::u8>::execute_forward();
template void _jit_uni_x8s8s32x_dw_convolution_fwd_t<avx2, false, data_type::u8, data_type::s8>::execute_forward();
template void _jit_uni_x8s8s32x_dw_convolution_fwd_t<avx2, false, data_type::u8, data_type::s32>::execute_forward();
template void _jit_uni_x8s8s32x_dw_convolution_fwd_t<avx2, false, data_type::u8, data_type::f32>::execute_forward();

template void _jit_uni_x8s8s32x_dw_convolution_fwd_t<sse42, true, data_type::u8, data_type::u8>::execute_forward();
template void _jit_uni_x8s8s32x_dw_convolution_fwd_t<sse42, true, data_type::u8, data_type::s8>::execute_forward();
template void _jit_uni_x8s8s32x_dw_convolution_fwd_t<sse42, true, data_type::u8, data_type::s32>::execute_forward();
template void _jit_uni_x8s8s32x_dw_convolution_fwd_t<sse42, true, data_type::u8, data_type::f32>::execute_forward();
template void _jit_uni_x8s8s32x_dw_convolution_fwd_t<sse42, false, data_type::u8, data_type::u8>::execute_forward();
template void _jit_uni_x8s8s32x_dw_convolution_fwd_t<sse42, false, data_type::u8, data_type::s8>::execute_forward();
template void _jit_uni_x8s8s32x_dw_convolution_fwd_t<sse42, false, data_type::u8, data_type::s32>::execute_forward();
template void _jit_uni_x8s8s32x_dw_convolution_fwd_t<sse42, false, data_type::u8, data_type::f32>::execute_forward();

}
}
}
