/*******************************************************************************
* Copyright 2018-2019 Intel Corporation
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
#include "jit_uni_x8s8s32x_convolution.hpp"
#include "utils.hpp"
#include "mkldnn_thread.hpp"
#include "type_helpers.hpp"
#include <cstring>

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::memory_tracking::names;
using namespace mkldnn::impl::utils;

template <cpu_isa_t isa, impl::data_type_t src_type, data_type_t dst_type>
void _jit_uni_x8s8s32x_convolution_fwd_t<isa, src_type, dst_type>::execute_forward() const {
    auto src = reinterpret_cast<const src_data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const wei_data_t *>(this->input_memory(1));
    auto bias = reinterpret_cast<const char *>(this->input_memory(2));
    auto dst = reinterpret_cast<dst_data_t *>(this->memory());

    const memory_desc_wrapper src_d(pd()->src_pd());
    const memory_desc_wrapper dst_d(pd()->dst_pd());
    const memory_desc_wrapper weights_d(pd()->weights_pd(0));
    const memory_desc_wrapper bias_d(pd()->weights_pd(1));

    const auto &jcp = kernel_->jcp;

    size_t offset = (size_t)jcp.ngroups * rnd_up(jcp.oc, jcp.oc_block) * rnd_up(jcp.ic, jcp.ic_block) * jcp.kd * jcp.kh * jcp.kw;
    auto w = const_cast<wei_data_t *>(weights);
    int32_t* compensation = (jcp.signed_input) ? reinterpret_cast<int32_t *>(&w[offset]) : 0;

    if (bias && jcp.oc != jcp.oc_padded) {
        auto padded_bias = this->scratchpad().template get<bia_data_t>(key_conv_padded_bias);
        utils::array_copy(padded_bias, (bia_data_t*)bias, jcp.oc);
        utils::array_set(padded_bias + jcp.oc, 0, jcp.oc_padded - jcp.oc);
        bias = (char *)padded_bias;
    }

    const float *oscales = pd()->attr()->output_scales_.scales_;
    if (jcp.signed_input) {
        auto local_scales = scratchpad().template get<float>(key_conv_adjusted_scales);
        size_t count = pd()->attr()->output_scales_.count_;
        float factor = 1.f / jcp.wei_adj_scale;
        if (count == 1) {
            utils::array_set(local_scales, oscales[0] * factor, 8);
        } else {
            for (size_t c = 0; c < count; c++)
                local_scales[c] = oscales[c] * factor;
        }
        oscales = local_scales;

        if (jcp.oc != jcp.oc_padded) {
            auto padded_compensation = this->scratchpad().template get<int32_t>(key_conv_padded_compensation);
            utils::array_copy(padded_compensation, compensation, jcp.oc);
            utils::array_set(padded_compensation + jcp.oc, 0, jcp.oc_padded - jcp.oc);
            compensation = padded_compensation;
        }
    }

    int ocb_work = div_up(jcp.nb_oc, jcp.nb_oc_blocking);
    const size_t work_amount = jcp.mb * jcp.ngroups * ocb_work * jcp.od * jcp.oh;

    auto ker = [&](const int ithr, const int nthr) {
        size_t start{0}, end{0};
        balance211(work_amount, nthr, ithr, start, end);

        size_t n{0}, g{0}, ocbb{0}, oh{0}, od{0};
        nd_iterator_init(start, n, jcp.mb, g, jcp.ngroups, ocbb, ocb_work,
                         od, jcp.od, oh, jcp.oh);
        for (size_t iwork = start; iwork < end; ++iwork) {
            int ocb = ocbb * jcp.nb_oc_blocking;
            int ocb_num = jcp.nb_oc_blocking;

            auto par_conv = jit_conv_call_s();

            const int ij = oh * jcp.stride_h;
            const int i_t_overflow = nstl::min(jcp.kh, div_up(nstl::max(0, jcp.t_pad - ij), (jcp.dilate_h+1)));
            const int i_b_overflow = nstl::min(jcp.kh, div_up(nstl::max(jcp.ih, ij + (jcp.kh-1) * (jcp.dilate_h+1) -
                                               jcp.t_pad+1) - jcp.ih, (jcp.dilate_h + 1)));

            const int ik = od * jcp.stride_d;
            const int i_front_overflow = nstl::min(jcp.kd, div_up(nstl::max(0, jcp.f_pad - ik), (jcp.dilate_d + 1)));
            const int i_back_overflow = nstl::min(jcp.kd, div_up(nstl::max(jcp.id, ik + (jcp.kd - 1) * (jcp.dilate_d + 1) -
                                               jcp.f_pad + 1) - jcp.id, (jcp.dilate_d + 1)));

            const size_t _oc = g * jcp.nb_oc + ocb;
            const size_t _ic = g * jcp.nb_ic;

            const int ih = nstl::max(ij - jcp.t_pad + i_t_overflow * (jcp.dilate_h + 1), 0);
            const int id = nstl::max(ik - jcp.f_pad + i_front_overflow * (jcp.dilate_d + 1), 0);

            size_t src_off = (jcp.ndims == 5) ? src_d.blk_off(n, _ic*jcp.ic_block, id, ih, 0)
                                              : src_d.blk_off(n, _ic*jcp.ic_block, ih, 0);
            par_conv.src = &src[src_off];

            size_t dst_off = (jcp.ndims == 5) ? dst_d.blk_off(n, _oc*jcp.oc_block, od, oh, 0)
                                              : dst_d.blk_off(n, _oc*jcp.oc_block, oh, 0);
            par_conv.dst = &dst[dst_off];

            const int wh = (!jcp.signed_input) ? i_t_overflow : 0;
            const int wd = (!jcp.signed_input) ? i_front_overflow : 0;

            size_t wei_off = (jcp.ndims == 5) ? pd()->with_groups() ? weights_d.blk_off(g, ocb, 0, wd, wh, 0)
                                                                    : weights_d.blk_off(ocb, 0, wd, wh, 0)
                                              : pd()->with_groups() ? weights_d.blk_off(g, ocb, 0, wh, 0)
                                                                    : weights_d.blk_off(ocb, 0, wh, 0);
            par_conv.filt = &weights[wei_off];

            if (bias)
                par_conv.bias = &bias[bias_d.blk_off(_oc * jcp.oc_block*jcp.typesize_bia)];

            par_conv.oc_work =
                    nstl::min((ocb + ocb_num) * jcp.oc_block, jcp.oc) - ocb*jcp.oc_block;

            par_conv.kw_padding = 0;
            const int kh_padding = jcp.kh - i_t_overflow - i_b_overflow;
            par_conv.kh_padding = nstl::max(0, kh_padding);
            const int kd_padding = jcp.kd - i_front_overflow - i_back_overflow;
            par_conv.kd_padding = nstl::max(0, kd_padding);

            par_conv.scales = &oscales[jcp.is_oc_scale * _oc * jcp.oc_block];

            par_conv.compensation = (jcp.signed_input) ? compensation + _oc * jcp.oc_block : 0;
            par_conv.t_overflow = i_t_overflow;
            par_conv.b_overflow = i_b_overflow;
            par_conv.front_overflow = i_front_overflow;
            par_conv.back_overflow = i_back_overflow;

            par_conv.oc_off = _oc * jcp.oc_block * sizeof(float);

            kernel_->jit_ker(&par_conv);
            nd_iterator_step(n, jcp.mb, g, jcp.ngroups, ocbb, ocb_work, od, jcp.od, oh, jcp.oh);
        }
    };

    parallel(0, work_amount, ker);
}

template <cpu_isa_t isa, impl::data_type_t src_type, data_type_t dst_type>
void _jit_uni_x8s8s32x_convolution_fwd_t<isa, src_type, dst_type>::execute_forward_with_dw_conv() const {
    auto src = reinterpret_cast<const src_data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const wei_data_t *>(this->input_memory(1));
    auto bias = reinterpret_cast<const char *>(this->input_memory(2));
    auto dst = reinterpret_cast<dst_data_t *>(this->memory());

    const memory_desc_wrapper src_d(pd()->src_pd());
    const memory_desc_wrapper weights_d(pd()->weights_pd(0));
    const memory_desc_wrapper bias_d(pd()->weights_pd(1));

    const auto &jcp = kernel_->jcp;
    const auto &jcp_dw = kernel_dw_->jcp;
    const int MB = pd()->MB();

    size_t offset = (size_t)jcp.ngroups * rnd_up(jcp.oc, jcp.oc_block) * rnd_up(jcp.ic, jcp.ic_block) * jcp.kh * jcp.kw;
    auto w = const_cast<wei_data_t *>(weights);
    int32_t* compensation = (jcp.signed_input) ? reinterpret_cast<int32_t *>(&w[offset]) : 0;

    auto dw_bias = jcp_dw.conv_biases;
    auto dw_weights = reinterpret_cast<const wei_data_t *>(jcp_dw.conv_weights);

    if (jcp.oc != jcp.oc_padded) {
        auto padded_bias = this->scratchpad().template get<bia_data_t>(key_conv_padded_bias);
        utils::array_copy(padded_bias, (bia_data_t*)bias, jcp.oc);
        utils::array_set(padded_bias + jcp.oc, 0, jcp.oc_padded - jcp.oc);
        bias = (char *)padded_bias;

        auto dw_padded_bias = this->scratchpad().template get<bia_data_t>(key_dw_conv_padded_bias);
        utils::array_copy(dw_padded_bias, dw_bias, jcp.oc);
        utils::array_set(dw_padded_bias + jcp.oc, 0.f, jcp.oc_padded - jcp.oc);
        dw_bias = dw_padded_bias;
    }

    const float *oscales = pd()->attr()->output_scales_.scales_;
    if (jcp.signed_input) {
        auto local_scales = scratchpad().template get<float>(key_conv_adjusted_scales);
        size_t count = pd()->attr()->output_scales_.count_;
        float factor = 1.f / jcp.wei_adj_scale;
        if (count == 1) {
            utils::array_set(local_scales, oscales[0] * factor, 8);
        } else {
            for (size_t c = 0; c < count; c++)
                local_scales[c] = oscales[c] * factor;
        }
        oscales = local_scales;

        if (jcp.oc != jcp.oc_padded) {
            auto padded_compensation = this->scratchpad().template get<int32_t>(key_conv_padded_compensation);
            utils::array_copy(padded_compensation, compensation, jcp.oc);
            utils::array_set(padded_compensation + jcp.oc, 0, jcp.oc_padded - jcp.oc);
            compensation = padded_compensation;
        }
    }

    int ocb_work = div_up(jcp.nb_oc, jcp.nb_oc_blocking);
    const size_t work_amount = MB * jcp.ngroups * ocb_work * jcp.oh;

    auto ker = [&](const int ithr, const int nthr) {
        auto compute_row_gen = [&](dst_data_t* ws_p, int n, int g, int ocb, int ocb_num, int oh, int num_rows) {
            for (int h = 0; h < num_rows; h++) {
                if ((oh + h) < 0 || (oh + h) >= jcp.oh) {
                    for (int chb = ocb; chb < ocb + ocb_num; chb++) {
                        memset(ws_p + (((oh + h) + 1) % jcp_dw.kh) * jcp.ow * jcp.oc_block +
                               (chb - ocb) * jcp_dw.kh * jcp.ow * jcp.oc_block, 0, jcp.ow * jcp.oc_block * sizeof(dst_data_t));
                    }
                } else {
                    auto par_conv = jit_conv_call_s();

                    const int ij = (oh + h) * jcp.stride_h;
                    const int i_t_overflow = nstl::min(jcp.kh, div_up(nstl::max(0, jcp.t_pad - ij), (jcp.dilate_h+1)));
                    const int i_b_overflow = nstl::min(jcp.kh, div_up(nstl::max(jcp.ih, ij + (jcp.kh-1) * (jcp.dilate_h+1) -
                                                       jcp.t_pad+1) - jcp.ih, (jcp.dilate_h + 1)));

                    const size_t _oc = g * jcp.nb_oc + ocb;
                    const size_t _ic = g * jcp.nb_ic;

                    const int ih = nstl::max(ij - jcp.t_pad + i_t_overflow * (jcp.dilate_h + 1), 0);
                    par_conv.src = &src[src_d.blk_off(n, _ic*jcp.ic_block, ih, 0)];

                    par_conv.dst = &ws_p[(((oh + h) + 1) % jcp_dw.kh) * jcp.ow * jcp.oc_block];

                    const int wh = (!jcp.signed_input) ? i_t_overflow : 0;
                    par_conv.filt = &weights[pd()->with_groups()
                                        ? weights_d.blk_off(g, ocb, 0, wh, 0)
                                        : weights_d.blk_off(ocb, 0, wh, 0)];

                    if (bias)
                        par_conv.bias = &bias[bias_d.blk_off(_oc * jcp.oc_block*jcp.typesize_bia)];

                    par_conv.oc_work =
                            nstl::min((ocb + ocb_num) * jcp.oc_block, jcp.oc) - ocb*jcp.oc_block;

                    par_conv.kw_padding = 0;
                    const int kh_padding = jcp.kh - i_t_overflow - i_b_overflow;
                    par_conv.kh_padding = nstl::max(0, kh_padding);

                    par_conv.scales = &oscales[jcp.is_oc_scale * _oc * jcp.oc_block];
                    par_conv.compensation = (jcp.signed_input) ? compensation + _oc * jcp.oc_block : 0;
                    par_conv.t_overflow = i_t_overflow;
                    par_conv.b_overflow = i_b_overflow;

                    par_conv.oc_off = _oc * jcp.oc_block * sizeof(float);

                    kernel_->jit_ker(&par_conv);
                }
            }
        };

        auto compute_row_dw = [&](const dst_data_t* ws_p, int n, int ocb, int ocb_num, int dst_idx) {
            for (int chb = ocb; chb < nstl::min(ocb + ocb_num, jcp.nb_oc); chb++) {
                auto par_conv_dw = jit_conv_call_s();

                par_conv_dw.src_row0 = &ws_p[(((dst_idx+1) - 1) % jcp_dw.kh) * jcp_dw.iw * jcp_dw.ch_block +
                                             (chb - ocb) * jcp_dw.kh * jcp_dw.iw * jcp_dw.ch_block];
                par_conv_dw.src_row1 = &ws_p[(((dst_idx+1) - 0) % jcp_dw.kh) * jcp_dw.iw * jcp_dw.ch_block +
                                             (chb - ocb) * jcp_dw.kh * jcp_dw.iw * jcp_dw.ch_block];
                par_conv_dw.src_row2 = &ws_p[(((dst_idx+1) + 1) % jcp_dw.kh) * jcp_dw.iw * jcp_dw.ch_block +
                                             (chb - ocb) * jcp_dw.kh * jcp_dw.iw * jcp_dw.ch_block];

                par_conv_dw.dst = &dst[n*jcp_dw.oc*jcp_dw.oh*jcp_dw.ow + dst_idx/jcp_dw.stride_h*jcp_dw.ow*jcp_dw.oc + chb*jcp_dw.ch_block];

                par_conv_dw.kh_padding = jcp_dw.kh;
                par_conv_dw.filt = &dw_weights[chb * jcp_dw.kh * jcp_dw.kw * jcp_dw.ch_block];
                par_conv_dw.bias = &dw_bias[chb * jcp_dw.ch_block];
                par_conv_dw.ur_w = (size_t)(jcp_dw.ow);
                par_conv_dw.oc_work = nstl::min((chb + 1) * jcp_dw.ch_block, (int)jcp_dw.oc) - chb*jcp_dw.ch_block;
                par_conv_dw.oc_off = chb * jcp_dw.ch_block * sizeof(float);

                kernel_dw_->jit_ker(&par_conv_dw);
            }
        };

        size_t start{0}, end{0};
        balance211(work_amount, nthr, ithr, start, end);

        auto dw_conv_buffer = scratchpad().template get<dst_data_t>(key_dw_conv_buffer);
        size_t dw_conv_buffer_size_ = (size_t)jcp_dw.kh * jcp_dw.iw * jcp_dw.ch_block * jcp.nb_oc_blocking;
        auto pbuf = dw_conv_buffer + ithr * dw_conv_buffer_size_;

        size_t n{0}, g{0}, ocbb{0}, oh{0};
        nd_iterator_init(start, n, MB, g, jcp.ngroups, ocbb, ocb_work, oh, jcp.oh);
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

            nd_iterator_step(n, MB, g, jcp.ngroups, ocbb, ocb_work, oh, jcp.oh);
        }
    };

    parallel(0, work_amount, ker);
}

template struct _jit_uni_x8s8s32x_convolution_fwd_t<avx2, data_type::u8, data_type::u8>;
template struct _jit_uni_x8s8s32x_convolution_fwd_t<avx2, data_type::u8, data_type::s8>;
template struct _jit_uni_x8s8s32x_convolution_fwd_t<avx2, data_type::u8, data_type::s32>;
template struct _jit_uni_x8s8s32x_convolution_fwd_t<avx2, data_type::u8, data_type::f32>;

template struct _jit_uni_x8s8s32x_convolution_fwd_t<avx2, data_type::s8, data_type::u8>;
template struct _jit_uni_x8s8s32x_convolution_fwd_t<avx2, data_type::s8, data_type::s8>;
template struct _jit_uni_x8s8s32x_convolution_fwd_t<avx2, data_type::s8, data_type::s32>;
template struct _jit_uni_x8s8s32x_convolution_fwd_t<avx2, data_type::s8, data_type::f32>;

template struct _jit_uni_x8s8s32x_convolution_fwd_t<sse42, data_type::u8, data_type::u8>;
template struct _jit_uni_x8s8s32x_convolution_fwd_t<sse42, data_type::u8, data_type::s8>;
template struct _jit_uni_x8s8s32x_convolution_fwd_t<sse42, data_type::u8, data_type::s32>;
template struct _jit_uni_x8s8s32x_convolution_fwd_t<sse42, data_type::u8, data_type::f32>;

template struct _jit_uni_x8s8s32x_convolution_fwd_t<sse42, data_type::s8, data_type::u8>;
template struct _jit_uni_x8s8s32x_convolution_fwd_t<sse42, data_type::s8, data_type::s8>;
template struct _jit_uni_x8s8s32x_convolution_fwd_t<sse42, data_type::s8, data_type::s32>;
template struct _jit_uni_x8s8s32x_convolution_fwd_t<sse42, data_type::s8, data_type::f32>;

}
}
}
