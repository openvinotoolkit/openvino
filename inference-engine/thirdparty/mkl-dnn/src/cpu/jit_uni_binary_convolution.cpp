    /*******************************************************************************
* Copyright 2019 Intel Corporation
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
#include "jit_uni_binary_convolution.hpp"
#include "utils.hpp"
#include "mkldnn_thread.hpp"
#include "type_helpers.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::memory_tracking::names;
using namespace mkldnn::impl::utils;

template <cpu_isa_t isa>
void jit_uni_binary_convolution_fwd_t<isa>::execute_forward() const {
    auto src = reinterpret_cast<const uint8_t*>(this->input_memory(0));
    auto weights = reinterpret_cast<const uint8_t*>(this->input_memory(1));
    auto dst_u8 = reinterpret_cast<uint8_t*>(this->memory());
    auto dst_f32 = reinterpret_cast<float*>(this->memory());

    const memory_desc_wrapper src_d(pd()->src_pd());
    const memory_desc_wrapper dst_d(pd()->dst_pd());
    const memory_desc_wrapper weights_d(pd()->weights_pd(0));

    const auto &jcp = kernel_->jcp;
    const int MB = pd()->MB();

    int ocb_work = div_up(jcp.nb_oc, jcp.nb_oc_blocking);
    const size_t work_amount = MB * jcp.ngroups * ocb_work * jcp.oh;

    int nbits = 8;

    auto ker = [&](const int ithr, const int nthr) {
        size_t start{0}, end{0};
        balance211(work_amount, nthr, ithr, start, end);

        size_t n{0}, g{0}, ocbb{0}, oh{0};
        nd_iterator_init(start, n, MB, g, jcp.ngroups, ocbb, ocb_work, oh, jcp.oh);
        for (size_t iwork = start; iwork < end; ++iwork) {
            int ocb = ocbb * jcp.nb_oc_blocking;
            int ocb_num = jcp.nb_oc_blocking;

            auto par_conv = jit_conv_call_s();

            const int ij = oh * jcp.stride_h;
            const int i_t_overflow = nstl::min(jcp.kh, div_up(nstl::max(0, jcp.t_pad - ij), (jcp.dilate_h+1)));
            const int i_b_overflow = nstl::min(jcp.kh, div_up(nstl::max(jcp.ih, ij + (jcp.kh-1) * (jcp.dilate_h+1) -
                                                                                jcp.t_pad+1) - jcp.ih, (jcp.dilate_h + 1)));

            const size_t _oc = g * jcp.nb_oc + ocb;
            const size_t _ic = g * jcp.nb_ic;

            const int ih = nstl::max(ij - jcp.t_pad + i_t_overflow * (jcp.dilate_h + 1), 0);
            par_conv.src = &src[src_d.blk_off(n, _ic*jcp.ic_block, ih, 0) / nbits];

            if (jcp.with_binarization) {
                par_conv.dst = &dst_u8[dst_d.blk_off(n, _oc*jcp.oc_block, oh, 0) / nbits];
            } else {
                par_conv.dst = &dst_f32[dst_d.blk_off(n, _oc*jcp.oc_block, oh, 0)];
            }

            const int wh = jcp.exclude_pad ? i_t_overflow : 0;
            int widx = weights_d.blk_off(ocb, 0, wh, 0);
            par_conv.filt = &weights[widx / nbits];

            par_conv.oc_work = nstl::min((ocb + ocb_num) * jcp.oc_block, jcp.oc) - ocb*jcp.oc_block;

            par_conv.kw_padding = 0;
            const int kh_padding = jcp.kh - i_t_overflow - i_b_overflow;
            par_conv.kh_padding = nstl::max(0, kh_padding);
            par_conv.t_overflow = i_t_overflow;
            par_conv.b_overflow = i_b_overflow;

            par_conv.oc_off = _oc * jcp.oc_block * sizeof(float);

            kernel_->jit_ker(&par_conv);

            nd_iterator_step(n, MB, g, jcp.ngroups, ocbb, ocb_work, oh, jcp.oh);
        }
    };

    parallel(0, (size_t)work_amount, ker);
}

template <cpu_isa_t isa>
void jit_uni_binary_convolution_fwd_t<isa>::execute_forward_with_dw_conv() const {
    auto src = reinterpret_cast<const uint8_t*>(this->input_memory(0));
    auto weights = reinterpret_cast<const uint8_t*>(this->input_memory(1));
    auto dst_u8 = reinterpret_cast<uint8_t*>(this->memory());
    auto dst_f32 = reinterpret_cast<float*>(this->memory());

    const memory_desc_wrapper src_d(pd()->src_pd());
    const memory_desc_wrapper weights_d(pd()->weights_pd(0));

    const auto &jcp = kernel_->jcp;
    const auto &jcp_dw_conv = dw_conv_kernel_->jcp;
    const int MB = pd()->MB();

    auto dw_conv_bias = jcp_dw_conv.conv_biases;
    auto dw_conv_weights = reinterpret_cast<const float*>(jcp_dw_conv.conv_weights);

    int ocb_work = div_up(jcp.nb_oc, jcp.nb_oc_blocking);
    const size_t work_amount = MB * jcp.ngroups * ocb_work * jcp.oh;

    int nbits = 8;

    auto ker = [&](const int ithr, const int nthr) {
        auto compute_row_generic_conv = [&](float* ws_p, int n, int g, int ocb, int ocb_num, int oh, int num_rows) {
            for (int h = 0; h < num_rows; h++) {
                if ((oh + h) < 0 || (oh + h) >= jcp.oh) {
                    for (int chb = ocb; chb < ocb + ocb_num; chb++) {
                        memset(ws_p + (((oh + h) + 1) % jcp_dw_conv.kh) * jcp.ow * jcp.oc_block +
                               (chb - ocb) * jcp_dw_conv.kh * jcp.ow * jcp.oc_block, 0, jcp.ow * jcp.oc_block * sizeof(float));
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
                    par_conv.src = &src[src_d.blk_off(n, _ic*jcp.ic_block, ih, 0) / nbits];

                    par_conv.dst = &ws_p[(((oh + h) + 1) % jcp_dw_conv.kh) * jcp.ow * jcp.oc_block];

                    const int wh = jcp.exclude_pad ? i_t_overflow : 0;
                    int widx = weights_d.blk_off(ocb, 0, wh, 0);
                    par_conv.filt = &weights[widx / nbits];

                    par_conv.oc_work = nstl::min((ocb + ocb_num) * jcp.oc_block, jcp.oc) - ocb*jcp.oc_block;

                    par_conv.kw_padding = 0;
                    const int kh_padding = jcp.kh - i_t_overflow - i_b_overflow;
                    par_conv.kh_padding = nstl::max(0, kh_padding);
                    par_conv.t_overflow = i_t_overflow;
                    par_conv.b_overflow = i_b_overflow;

                    par_conv.oc_off = _oc * jcp.oc_block * sizeof(float);

                    kernel_->jit_ker(&par_conv);
                }
            }
        };

        auto compute_row_dw_conv = [&](const float* ws_p, int n, int ocb, int ocb_num, int dst_idx) {
            for (int chb = ocb; chb < nstl::min(ocb + ocb_num, jcp.nb_oc); chb++) {
                auto par_conv_dw = jit_conv_call_s();

                par_conv_dw.src_row0 = &ws_p[(((dst_idx+1) - 1) % jcp_dw_conv.kh) * jcp_dw_conv.iw * jcp_dw_conv.ch_block +
                                             (chb - ocb) * jcp_dw_conv.kh * jcp_dw_conv.iw * jcp_dw_conv.ch_block];
                par_conv_dw.src_row1 = &ws_p[(((dst_idx+1) - 0) % jcp_dw_conv.kh) * jcp_dw_conv.iw * jcp_dw_conv.ch_block +
                                             (chb - ocb) * jcp_dw_conv.kh * jcp_dw_conv.iw * jcp_dw_conv.ch_block];
                par_conv_dw.src_row2 = &ws_p[(((dst_idx+1) + 1) % jcp_dw_conv.kh) * jcp_dw_conv.iw * jcp_dw_conv.ch_block +
                                             (chb - ocb) * jcp_dw_conv.kh * jcp_dw_conv.iw * jcp_dw_conv.ch_block];

                if (jcp_dw_conv.with_binarization) {
                    int nbits = 8;

                    int didx = n*jcp_dw_conv.oc*jcp_dw_conv.oh*jcp_dw_conv.ow +
                               dst_idx/jcp_dw_conv.stride_h*jcp_dw_conv.ow*jcp_dw_conv.oc + chb*jcp_dw_conv.ch_block;
                    par_conv_dw.dst = &dst_u8[didx / nbits];
                } else {
                    par_conv_dw.dst = &dst_f32[n*jcp_dw_conv.oc*jcp_dw_conv.oh*jcp_dw_conv.ow +
                                               dst_idx/jcp_dw_conv.stride_h*jcp_dw_conv.ow*jcp_dw_conv.oc + chb*jcp_dw_conv.ch_block];
                }

                par_conv_dw.kh_padding = jcp_dw_conv.kh;
                par_conv_dw.filt = &dw_conv_weights[chb * jcp_dw_conv.kh * jcp_dw_conv.kw * jcp_dw_conv.ch_block];
                par_conv_dw.bias = &dw_conv_bias[chb * jcp_dw_conv.ch_block];
                par_conv_dw.ur_w = (size_t)(jcp_dw_conv.ow);
                par_conv_dw.oc_work = nstl::min((chb + 1) * jcp_dw_conv.ch_block, jcp_dw_conv.oc) - chb*jcp_dw_conv.ch_block;
                par_conv_dw.oc_off = chb * jcp_dw_conv.ch_block * sizeof(float);

                dw_conv_kernel_->jit_ker(&par_conv_dw);
            }
        };

        size_t start{0}, end{0};
        balance211(work_amount, nthr, ithr, start, end);
        auto dw_conv_buffer_ = scratchpad().template get<float>(key_dw_conv_buffer);
        size_t dw_conv_buffer_size_ = (size_t)jcp_dw_conv.kh * jcp_dw_conv.iw * jcp_dw_conv.ch_block * jcp.nb_oc_blocking;
        auto pbuf = dw_conv_buffer_ + ithr * dw_conv_buffer_size_;

        size_t n{0}, g{0}, ocbb{0}, oh{0};
        nd_iterator_init(start, n, MB, g, jcp.ngroups, ocbb, ocb_work, oh, jcp.oh);
        for (size_t iwork = start; iwork < end; ++iwork) {
            int ocb = ocbb * jcp.nb_oc_blocking;
            int ocb_num = jcp.nb_oc_blocking;

            if (iwork == start || oh == 0) {
                compute_row_generic_conv(pbuf, n, g, ocb, ocb_num, oh - 1, 2);
            } else {
                compute_row_generic_conv(pbuf, n, g, ocb, ocb_num, oh, 1);
            }

            if (iwork > start && ((oh - 1) % jcp_dw_conv.stride_h == 0) && oh > 0) {
                compute_row_dw_conv(pbuf, n, ocb, ocb_num, oh - 1);
            }

            if ((iwork == end - 1 || (int) oh == jcp.oh - 1) && ((oh) % jcp_dw_conv.stride_h == 0)) {
                compute_row_generic_conv(pbuf, n, g, ocb, ocb_num, oh + 1, 1);
                compute_row_dw_conv(pbuf, n, ocb, ocb_num, oh);
            }

            nd_iterator_step(n, MB, g, jcp.ngroups, ocbb, ocb_work, oh, jcp.oh);
        }
    };

    if (jcp.oc != jcp.oc_padded) {
        auto dw_conv_padded_bias = scratchpad().template get<float>(key_dw_conv_padded_bias);
        utils::array_copy(dw_conv_padded_bias, dw_conv_bias, jcp.oc);
        utils::array_set(dw_conv_padded_bias + jcp.oc, 0.f, jcp.oc_padded - jcp.oc);
        dw_conv_bias = dw_conv_padded_bias;
    }

    parallel(0, (size_t)work_amount, ker);
}

template struct jit_uni_binary_convolution_fwd_t<avx512_common>;
template struct jit_uni_binary_convolution_fwd_t<avx2>;
template struct jit_uni_binary_convolution_fwd_t<sse42>;

}
}
}
