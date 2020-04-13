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

#include <common/memory_tracking.hpp>
#include "mkldnn_types.h"
#include "c_types_map.hpp"
#include "jit_uni_x8s8s32x_dw_convolution.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::memory_tracking::names;
using namespace mkldnn::impl::utils;

template <cpu_isa_t isa, data_type_t src_type, data_type_t dst_type>
void _jit_uni_x8s8s32x_dw_convolution_fwd_t<isa, src_type, dst_type>::execute_forward() const {
    auto src = reinterpret_cast<const src_data_t*>(this->input_memory(0));
    auto weights = reinterpret_cast<const wei_data_t*>(this->input_memory(1));
    auto bias = reinterpret_cast<const char*>(this->input_memory(2));
    auto dst = reinterpret_cast<dst_data_t*>(this->memory());

    const memory_desc_wrapper src_d(pd()->src_pd());
    const memory_desc_wrapper dst_d(pd()->dst_pd());
    const memory_desc_wrapper weights_d(pd()->weights_pd(0));
    const memory_desc_wrapper bias_d(pd()->weights_pd(1));

    const auto &jcp = kernel_->jcp;

    int dil_d = jcp.dilate_d + 1;
    int dil_h = jcp.dilate_h + 1;
    int dil_w = jcp.dilate_w + 1;
    int str_d = jcp.stride_d;
    int str_h = jcp.stride_h;
    int str_w = jcp.stride_w;

    const size_t bia_dt_size = pd()->with_bias()
        ? types::data_type_size(pd()->desc()->bias_desc.data_type) : 0;

    const auto &oscales = pd()->attr()->output_scales_;
    const int32_t* compensation = (jcp.with_input_zp) ? pd()->attr()->output_compensations_.shifts_ : nullptr;
    const uint8_t* input_zp = pd()->attr()->input_zero_points_.shifts_;
    int32_t *weights_zp = nullptr;
    if (jcp.with_weights_zp) {
        weights_zp = scratchpad().template get<int32_t>(key_weights_zp);
        for (int i = 0; i < pd()->attr()->weights_zero_points_.count_; i++) {
            weights_zp[i] = (int32_t)pd()->attr()->weights_zero_points_.shifts_[i];
        }
    }

    int MB = jcp.mb;
    int chb_work = utils::div_up(jcp.nb_ch, jcp.nb_ch_blocking);
    const size_t work_amount = MB * chb_work * jcp.od * jcp.oh;

    auto kernel_params = [&](int ur_w_step, int ow, int oh, int od, int ih, int id, int kh, int kd, int kh_padding, int kd_padding,
            int ch, int ch_num, int n, int i_t_overflow, int i_b_overflow, int i_front_overflow, int i_back_overflow) {
        auto par_conv = jit_conv_call_s();

        const int iw_ = ow * str_w;
        const int i_l_overflow = nstl::min(jcp.kw, div_up(nstl::max(0, jcp.l_pad - iw_), dil_w));
        const int i_r_overflow = nstl::min(jcp.kw, div_up(nstl::max(jcp.iw, iw_ + (jcp.kw-1) * dil_w - jcp.l_pad+1) - jcp.iw, dil_w));
        const int iw = nstl::max(iw_ - jcp.l_pad + i_l_overflow * dil_w, 0);
        const int kw = (!jcp.with_input_zp) ? i_l_overflow : 0;
        const int kw_padding = jcp.kw - i_l_overflow - i_r_overflow;

        size_t src_off = (jcp.ndims == 5) ? src_d.blk_off(n, ch*jcp.ch_block, id, ih, iw)
                                          : src_d.blk_off(n, ch*jcp.ch_block, ih, iw);
        size_t dst_off = (jcp.ndims == 5) ? dst_d.blk_off(n, ch*jcp.ch_block, od, oh, ow)
                                          : dst_d.blk_off(n, ch*jcp.ch_block, oh, ow);
        size_t wei_off = (jcp.ndims == 5) ? weights_d.blk_off(ch, 0, 0, kd, kh, kw)
                                          : weights_d.blk_off(ch, 0, 0, kh, kw);

        par_conv.src = &src[src_off];
        par_conv.dst = &dst[dst_off];
        par_conv.filt = &weights[wei_off];
        if (bias) par_conv.bias = &bias[bias_d.blk_off(ch*jcp.ch_block*bia_dt_size)];

        par_conv.kd_padding = (size_t)nstl::max(0, kd_padding);
        par_conv.kh_padding = (size_t)nstl::max(0, kh_padding);
        par_conv.kw_padding = (size_t)nstl::max(0, kw_padding);

        par_conv.l_overflow = i_l_overflow;
        par_conv.r_overflow = i_r_overflow;
        par_conv.t_overflow = i_t_overflow;
        par_conv.b_overflow = i_b_overflow;
        par_conv.front_overflow = i_front_overflow;
        par_conv.back_overflow = i_back_overflow;

        par_conv.ur_w = (size_t)ur_w_step;

        par_conv.ch_work = nstl::min((ch + ch_num) * jcp.ch_block, jcp.oc) - ch*jcp.ch_block;

        par_conv.scales = &oscales.scales_[jcp.is_oc_scale * ch * jcp.ch_block];
        par_conv.oc_off = ch * jcp.ch_block * sizeof(float);

        if (jcp.with_input_zp) {
            par_conv.compensation = compensation + ch * jcp.ch_block;
            par_conv.input_zp = input_zp + ch * jcp.ch_block;
        }

        if (jcp.with_weights_zp) {
            par_conv.weights_zp = weights_zp + ch * jcp.ch_block;
        }

        return par_conv;
    };

    auto ker = [&](const int ithr, const int nthr) {
        size_t start{0}, end{0};
        balance211(work_amount, nthr, ithr, start, end);

        size_t n{0}, chb{0}, oh{0}, od{0};
        nd_iterator_init(start, n, MB, chb, chb_work, od, jcp.od, oh, jcp.oh);
        for (size_t iwork = start; iwork < end; ++iwork) {
            int ch = chb * jcp.nb_ch_blocking;
            int ch_num = jcp.nb_ch_blocking;

            const int id_ = od * str_d;
            const int i_front_overflow = nstl::min(jcp.kd, div_up(nstl::max(0, jcp.f_pad - id_), dil_d));
            const int i_back_overflow = nstl::min(jcp.kd, div_up(nstl::max(jcp.id, id_ + (jcp.kd - 1) * dil_d - jcp.f_pad + 1) - jcp.id, dil_d));
            const int id = nstl::max(id_ - jcp.f_pad + i_front_overflow * dil_d, 0);
            const int kd = (!jcp.with_input_zp) ? i_front_overflow : 0;
            const int kd_padding = jcp.kd - i_front_overflow - i_back_overflow;

            const int ih_ = oh * str_h;
            const int i_t_overflow = nstl::min(jcp.kh, div_up(nstl::max(0, jcp.t_pad - ih_), dil_h));
            const int i_b_overflow = nstl::min(jcp.kh, div_up(nstl::max(jcp.ih, ih_ + (jcp.kh-1) * dil_h - jcp.t_pad+1) - jcp.ih, dil_h));
            const int ih = nstl::max(ih_ - jcp.t_pad + i_t_overflow * dil_h, 0);
            const int kh = (!jcp.with_input_zp) ? i_t_overflow : 0;
            const int kh_padding = jcp.kh - i_t_overflow - i_b_overflow;

            // left border
            int ow = 0;
            int l_border = nstl::min(div_up(jcp.l_pad, str_w), jcp.ow);
            int ur_w_step = 1;
            for (; ow < l_border; ow++) {
                jit_conv_call_s par_conv = kernel_params(ur_w_step, ow, oh, od, ih, id, kh, kd, kh_padding, kd_padding, ch, ch_num, n,
                                                         i_t_overflow, i_b_overflow, i_front_overflow, i_back_overflow);

                kernel_->jit_ker(&par_conv);
            }

            // main loop
            ur_w_step = (jcp.iw - (jcp.kw - 1)*dil_w + jcp.l_pad - 1)
                / jcp.stride_w - ow + 1;
            if (ur_w_step > 0) {
                jit_conv_call_s par_conv = kernel_params(ur_w_step, ow, oh, od, ih, id, kh, kd, kh_padding, kd_padding, ch, ch_num, n,
                                                         i_t_overflow, i_b_overflow, i_front_overflow, i_back_overflow);

                kernel_->jit_ker(&par_conv);

                ow += ur_w_step;
            }

            // right border
            ur_w_step = 1;
            for (; ow < jcp.ow; ow++) {
                jit_conv_call_s par_conv = kernel_params(ur_w_step, ow, oh, od, ih, id, kh, kd, kh_padding, kd_padding, ch, ch_num, n,
                                                         i_t_overflow, i_b_overflow, i_front_overflow, i_back_overflow);

                kernel_->jit_ker(&par_conv);
            }

            nd_iterator_step(n, MB, chb, chb_work, od, jcp.od, oh, jcp.oh);
        }
    };

    parallel(0, work_amount, ker);
}

template struct _jit_uni_x8s8s32x_dw_convolution_fwd_t<avx2, data_type::u8, data_type::u8>;
template struct _jit_uni_x8s8s32x_dw_convolution_fwd_t<avx2, data_type::u8, data_type::s8>;
template struct _jit_uni_x8s8s32x_dw_convolution_fwd_t<avx2, data_type::u8, data_type::s32>;
template struct _jit_uni_x8s8s32x_dw_convolution_fwd_t<avx2, data_type::u8, data_type::f32>;

template struct _jit_uni_x8s8s32x_dw_convolution_fwd_t<avx2, data_type::s8, data_type::u8>;
template struct _jit_uni_x8s8s32x_dw_convolution_fwd_t<avx2, data_type::s8, data_type::s8>;
template struct _jit_uni_x8s8s32x_dw_convolution_fwd_t<avx2, data_type::s8, data_type::s32>;
template struct _jit_uni_x8s8s32x_dw_convolution_fwd_t<avx2, data_type::s8, data_type::f32>;

template struct _jit_uni_x8s8s32x_dw_convolution_fwd_t<sse42, data_type::u8, data_type::u8>;
template struct _jit_uni_x8s8s32x_dw_convolution_fwd_t<sse42, data_type::u8, data_type::s8>;
template struct _jit_uni_x8s8s32x_dw_convolution_fwd_t<sse42, data_type::u8, data_type::s32>;
template struct _jit_uni_x8s8s32x_dw_convolution_fwd_t<sse42, data_type::u8, data_type::f32>;

template struct _jit_uni_x8s8s32x_dw_convolution_fwd_t<sse42, data_type::s8, data_type::u8>;
template struct _jit_uni_x8s8s32x_dw_convolution_fwd_t<sse42, data_type::s8, data_type::s8>;
template struct _jit_uni_x8s8s32x_dw_convolution_fwd_t<sse42, data_type::s8, data_type::s32>;
template struct _jit_uni_x8s8s32x_dw_convolution_fwd_t<sse42, data_type::s8, data_type::f32>;

}
}
}
