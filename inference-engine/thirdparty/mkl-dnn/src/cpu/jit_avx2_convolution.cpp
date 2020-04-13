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

#include "c_types_map.hpp"
#include "mkldnn_thread.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"
#include <cstring>

#include "jit_avx2_convolution.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::memory_tracking::names;
using namespace mkldnn::impl::utils;
using namespace nstl;

#define src_blk_off(f, n, c, d, h, w) \
    (pd()->ndims() == 3) \
    ? (f).blk_off(n, c, w) \
    : (pd()->ndims() == 4) \
    ? (f).blk_off(n, c, h, w) \
    : (f).blk_off(n, c, d, h, w)

#define wht_blk_off_(f, g, ...) \
    pd()->with_groups() ? (f).blk_off(g, __VA_ARGS__) : (f).blk_off(__VA_ARGS__)
#define wht_blk_off(f, g, oc, ic, kd, kh, kw) \
    (pd()->ndims() == 3) \
    ? wht_blk_off_(f, g, oc, ic, kw) \
    : (pd()->ndims() == 4) \
    ? wht_blk_off_(f, g, oc, ic, kh, kw) \
    : wht_blk_off_(f, g, oc, ic, kd, kh, kw)

void jit_avx2_convolution_fwd_t::execute_forward() const {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto bias = reinterpret_cast<const data_t *>(this->input_memory(2));
    auto dst = reinterpret_cast<data_t *>(this->memory());

    const memory_desc_wrapper src_d(pd()->src_pd());
    const memory_desc_wrapper dst_d(pd()->dst_pd());
    const memory_desc_wrapper weights_d(pd()->weights_pd(0));
    const memory_desc_wrapper bias_d(pd()->weights_pd(1));

    const auto &jcp = kernel_->jcp;
    const int MB = pd()->MB();

    int ocb_work = div_up(jcp.nb_oc, jcp.nb_oc_blocking);
    const size_t work_amount = MB * jcp.ngroups * ocb_work * jcp.od
        * jcp.oh;

    auto ker = [&](const int ithr, const int nthr) {
        size_t start{0}, end{0};
        balance211(work_amount, nthr, ithr, start, end);

        int icbb = 0;
        while (icbb < jcp.nb_ic) {
            int icb_step = jcp.nb_ic_blocking;
            int icb_step_rem = jcp.nb_ic - icbb;
            if (icb_step_rem < jcp.nb_ic_blocking_max)
                icb_step = icb_step_rem;

            size_t n{0}, g{0}, ocbb{0}, oh{0}, od{0};
            nd_iterator_init(start, n, MB, g, jcp.ngroups, ocbb, ocb_work,
                             od, jcp.od, oh, jcp.oh);
            for (size_t iwork = start; iwork < end; ++iwork) {
                int ocb = ocbb * jcp.nb_oc_blocking;
                int ocb_num = jcp.nb_oc_blocking;

                for (int icb = icbb; icb < icbb + icb_step; ++icb) {
                    auto par_conv = jit_conv_call_s();

                    const int ij = oh * jcp.stride_h;
                    const int i_t_overflow = nstl::max(0, jcp.t_pad - ij);
                    const int i_b_overflow = nstl::max(jcp.ih, ij
                        + (jcp.kh-1) * (jcp.dilate_h+1) - jcp.t_pad+1) - jcp.ih;

                    const int dj = od * jcp.stride_d;
                    const int d_t_overflow = nstl::max(0, jcp.f_pad - dj);
                    const int d_b_overflow = nstl::max(jcp.id, dj
                        + (jcp.kd-1) * (jcp.dilate_d+1) - jcp.f_pad+1) - jcp.id;

                    const size_t _oc = g * jcp.nb_oc + ocb;
                    const size_t _ic = g * jcp.nb_ic * jcp.nonblk_group_off + icb;

                    const int ih = nstl::max(ij - jcp.t_pad
                        + div_up(i_t_overflow,
                                 (jcp.dilate_h+1)) * (jcp.dilate_h + 1), 0);

                    const int id = nstl::max(dj - jcp.f_pad
                        + div_up(d_t_overflow,
                                 (jcp.dilate_d+1)) * (jcp.dilate_d + 1), 0);

                    par_conv.src = &src[src_blk_off(src_d, n, _ic, id, ih, 0)];

                    par_conv.dst = &dst[src_blk_off(dst_d, n, _oc, od, oh, 0)];

                    const int wh = div_up(i_t_overflow, (jcp.dilate_h + 1));
                    const int wd = div_up(d_t_overflow, (jcp.dilate_d + 1));
                    par_conv.filt = &weights[wht_blk_off(
                            weights_d, g, ocb, icb, wd, wh, 0)];

                    if (icb == 0) {
                        if (bias)
                            par_conv.bias =
                                    &bias[bias_d.blk_off(_oc * jcp.oc_block)];
                        par_conv.flags |= FLAG_IC_FIRST;
                    }

                    if (icb + 1 == jcp.nb_ic) {
                        par_conv.flags |= FLAG_IC_LAST;
                    }

                    par_conv.oc_off = _oc * jcp.oc_block * sizeof(float);

                    par_conv.oc_blocks =
                            nstl::min(ocb + ocb_num, jcp.nb_oc) - ocb;

                    par_conv.kw_padding = 0;
                    const int kh_padding = jcp.kh
                        - div_up(i_t_overflow, (jcp.dilate_h + 1))
                        - div_up(i_b_overflow, (jcp.dilate_h + 1));
                    par_conv.kh_padding = nstl::max(0, kh_padding);

                    const int kd_padding = jcp.kd
                        - div_up(d_t_overflow, (jcp.dilate_d + 1))
                        - div_up(d_b_overflow, (jcp.dilate_d + 1));
                    par_conv.kd_padding = nstl::max(0, kd_padding);

                    kernel_->jit_ker(&par_conv);
                }
                nd_iterator_step(n, MB, g, jcp.ngroups, ocbb, ocb_work,
                                od, jcp.od, oh, jcp.oh);
            }
            icbb += icb_step;
        }
    };

    if (pd()->wants_padded_bias()) {
        auto padded_bias = scratchpad().get<data_t>(key_conv_padded_bias);
        utils::array_copy(padded_bias, bias, jcp.oc_without_padding);
        utils::array_set(padded_bias + jcp.oc_without_padding, 0.f,
                jcp.oc - jcp.oc_without_padding);
        bias = padded_bias;
    }

    parallel(0, work_amount, ker);

    if (pd()->wants_zero_pad_dst())
        output_memory_primitive(0)->zero_pad();
}

void jit_avx2_convolution_fwd_t::execute_forward_with_dw_conv() const {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto bias = reinterpret_cast<const data_t *>(this->input_memory(2));
    auto dst = reinterpret_cast<data_t *>(this->memory());

    const memory_desc_wrapper src_d(pd()->src_pd());
    const memory_desc_wrapper weights_d(pd()->weights_pd(0));
    const memory_desc_wrapper bias_d(pd()->weights_pd(1));

    const auto &jcp = kernel_->jcp;
    const auto &jcp_dw = kernel_dw_->jcp;
    const int MB = pd()->MB();

    auto dw_bias = jcp_dw.conv_biases;

    int ocb_work = div_up(jcp.nb_oc, jcp.nb_oc_blocking);
    const size_t work_amount = MB * jcp.ngroups * ocb_work * jcp.oh;

    auto ker = [&](const int ithr, const int nthr) {
        auto compute_row_gen = [&](float* ws_p, int n, int g, int ocb, int ocb_num, int oh, int num_rows) {
            for (int h = 0; h < num_rows; h++) {
                if ((oh + h) < 0 || (oh + h) >= jcp.oh) {
                    for (int chb = ocb; chb < ocb + ocb_num; chb++) {
                        memset(ws_p + (((oh + h) + 1) % jcp_dw.kh) * jcp.ow * jcp.oc_block +
                               (chb - ocb) * jcp_dw.kh * jcp.ow * jcp.oc_block, 0, jcp.ow * jcp.oc_block * sizeof(float));
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

                        par_conv.dst = &ws_p[(((oh + h) + 1) % jcp_dw.kh) * jcp.ow *
                                             jcp.oc_block];

                        const int wh = div_up(i_t_overflow, (jcp.dilate_h + 1));
                        par_conv.filt = &weights[pd()->with_groups()
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

                        par_conv.oc_off = _oc * jcp.oc_block * sizeof(float);

                        par_conv.oc_blocks =
                                nstl::min(ocb + ocb_num, jcp.nb_oc) - ocb;

                        par_conv.kw_padding = 0;
                        const int kh_padding = jcp.kh
                                               - div_up(i_t_overflow, (jcp.dilate_h + 1))
                                               - div_up(i_b_overflow, (jcp.dilate_h + 1));
                        par_conv.kh_padding = nstl::max(0, kh_padding);
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
                par_conv_dw.filt = &jcp_dw.conv_weights[chb * jcp_dw.kh * jcp_dw.kw * jcp_dw.ch_block];
                par_conv_dw.bias = &dw_bias[chb * jcp_dw.ch_block];
                par_conv_dw.ur_w = (size_t)(jcp_dw.ow);
                par_conv_dw.oc_work = nstl::min((chb + 1) * jcp_dw.ch_block, (int)jcp_dw.oc) - chb*jcp_dw.ch_block;
                par_conv_dw.oc_off = chb * jcp_dw.ch_block * sizeof(float);

                kernel_dw_->jit_ker(&par_conv_dw);
            }
        };

        size_t start{0}, end{0};
        balance211(work_amount, nthr, ithr, start, end);

        auto dw_conv_buffer = scratchpad().get<data_t>(key_dw_conv_buffer);
        size_t dw_conv_buffer_size_ = (size_t)jcp_dw.kh * jcp_dw.iw * jcp_dw.ch_block * jcp.nb_oc_blocking;
        auto pbuf = dw_conv_buffer + ithr * dw_conv_buffer_size_;

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

    if (pd()->wants_padded_bias()) {
        auto padded_bias = scratchpad().get<data_t>(key_conv_padded_bias);
        utils::array_copy(padded_bias, bias, jcp.oc_without_padding);
        utils::array_set(padded_bias + jcp.oc_without_padding, 0.f,
                jcp.oc - jcp.oc_without_padding);
        bias = padded_bias;

        auto dw_padded_bias = scratchpad().get<data_t>(key_dw_conv_padded_bias);
        utils::array_copy(dw_padded_bias, dw_bias, jcp.oc_without_padding);
        utils::array_set(dw_padded_bias + jcp.oc_without_padding, 0.f,
                         jcp.oc - jcp.oc_without_padding);
        dw_bias = dw_padded_bias;
    }

    parallel(0, work_amount, ker);

    if (pd()->wants_zero_pad_dst())
        output_memory_primitive(0)->zero_pad();
}

void jit_avx2_convolution_bwd_data_t::execute_backward_data() const {
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto diff_src = reinterpret_cast<data_t *>(this->memory());

    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_pd());
    const memory_desc_wrapper diff_src_d(pd()->diff_src_pd());
    const memory_desc_wrapper weights_d(pd()->weights_pd(0));

    const auto &jcp = kernel_->jcp;
    const int MB = pd()->MB();

    int icb_work = jcp.nb_ic / jcp.nb_ic_blocking;
    int ih_block_size = jcp.ih;
    int num_ih_blocks = utils::div_up(jcp.ih, ih_block_size);
    size_t work_amount = MB * jcp.ngroups * icb_work * num_ih_blocks;
    if (work_amount < (size_t)2 * mkldnn_get_max_threads()) {
        ih_block_size = 1;
        num_ih_blocks = utils::div_up(jcp.ih, ih_block_size);
        work_amount *= num_ih_blocks;
    }

    auto ker = [&](const int ithr, const int nthr) {
        size_t start{0}, end{0};
        balance211(work_amount, nthr, ithr, start, end);

        size_t n{0}, g{0}, icbb{0}, ihb{0};
        nd_iterator_init(start, n, MB, g, jcp.ngroups, icbb, icb_work,
                         ihb, num_ih_blocks);

        for (size_t iwork = start; iwork < end; ++iwork) {
            for (int oc = 0; oc < jcp.nb_oc; oc += jcp.nb_oc_blocking)
            for (int id = 0; id < jcp.id; ++id) {
                auto par_conv = jit_conv_call_s();

                const int idp = jcp.id + 2 * jcp.f_pad;
                const int d_t_overflow = nstl::max(0,
                        jcp.kd - 1 - id - jcp.f_pad);
                const int back_pad = idp - jcp.id - jcp.f_pad;
                const int d_b_overflow = nstl::max(0,
                        jcp.kd - 1 - (jcp.id - 1 - id) - back_pad);
                const int od = id + jcp.f_pad - d_b_overflow;

                int ih_start = ihb * ih_block_size;
                int ih_end = nstl::min(jcp.ih, ih_start + ih_block_size);
                for (int ih = ih_start; ih < ih_end; ++ih) {

                    const int i_t_overflow = nstl::max(0, (jcp.kh - 1
                                        - ih - jcp.t_pad) / jcp.stride_h);
                    const int i_b_overflow = nstl::max(0, (jcp.kh - jcp.ih
                                        + ih - jcp.b_pad) / jcp.stride_h);
                    int overflow_kh_hi = jcp.kh - 1
                            - modulo(jcp.ih - 1 + jcp.b_pad - ih, jcp.stride_h);
                    int overflow_kh_lo = (ih + jcp.t_pad) % jcp.stride_h;

                    par_conv.kd_padding = jcp.kd - d_t_overflow - d_b_overflow;
                    par_conv.kh_padding = (overflow_kh_hi - overflow_kh_lo)
                              / jcp.stride_h + 1 - i_t_overflow - i_b_overflow;
                    par_conv.kw_padding = 0;

                    const int k_lo = overflow_kh_lo
                                   + i_b_overflow * jcp.stride_h;
                    const int oh = (ih + jcp.t_pad - k_lo) / jcp.stride_h;

                    par_conv.src = &diff_src[src_blk_off(diff_src_d, n,
                        g * jcp.nb_ic + jcp.nb_ic_blocking * icbb, id, ih, 0)];
                    par_conv.dst = &diff_dst[src_blk_off(diff_dst_d,
                            n, g * jcp.nb_oc + oc, od, oh, 0)];
                    par_conv.filt = &weights[wht_blk_off(weights_d, g, oc,
                            jcp.nb_ic_blocking * icbb, d_b_overflow, k_lo, 0)];

                    par_conv.ic_off = (g * jcp.nb_ic + jcp.nb_ic_blocking * icbb) * jcp.ic_block * sizeof(float);

                    par_conv.src_prf = nullptr;
                    par_conv.dst_prf = nullptr;
                    par_conv.filt_prf = nullptr;
                    par_conv.channel = oc;
                    par_conv.ch_blocks = nstl::min(jcp.nb_oc - oc,
                                       jcp.nb_oc_blocking);

                    kernel_->jit_ker(&par_conv);
                }
            }
            nd_iterator_step(n, MB, g, jcp.ngroups, icbb, icb_work, ihb,
                             num_ih_blocks);
        }
    };

    parallel(0, work_amount, ker);
}

void jit_avx2_convolution_bwd_weights_t::execute_backward_weights() const {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto diff_weights = reinterpret_cast<data_t *>(this->memory(0));
    auto diff_bias_in = reinterpret_cast<data_t *>(this->memory(1));

    auto scratchpad = this->scratchpad();

    data_t *diff_bias = pd()->wants_padded_bias()
        ? scratchpad.get<data_t>(key_conv_padded_bias) : diff_bias_in;

    const memory_desc_wrapper src_d(pd()->src_pd(0));
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_pd());
    const memory_desc_wrapper diff_weights_d(pd()->diff_weights_pd(0));

    const auto &jcp = kernel_->jcp;

    auto reducer_bia_scratchpad = memory_tracking::grantor_t(scratchpad,
            prefix_reducer_bia);
    auto rb = this->reducer_bias_;
    rb->init(reducer_bia_scratchpad);

    auto reducer_wei_scratchpad = memory_tracking::grantor_t(scratchpad,
            prefix_reducer_wei);
    auto rw = this->reducer_weights_;
    rw->init(reducer_wei_scratchpad);

    auto ker = [&](int ithr, int nthr) {
        assert(nthr == rw->balancer().nthr_);

        const int w_job_start = rw->balancer().ithr_job_off(ithr);
        const int w_njobs = rw->balancer().ithr_njobs(ithr);

        if (w_njobs == 0) return;

        /* reduction dimension */
        int img_od_start{0}, img_od_end{0}, img{0}, od_s{0};
        balance211(jcp.mb * jcp.od, rw->balancer().nthr_per_group_,
                rw->balancer().id_in_group(ithr), img_od_start, img_od_end);

        int img_start = img_od_start, img_end = img_od_end;
        nd_iterator_init(img_start, img, jcp.mb, od_s, jcp.od);
        const int img_first = img;

        /* jobs */
        int g_start{0}, ocb_start{0}, icb_start{0};
        nd_iterator_init(w_job_start, g_start, jcp.ngroups, ocb_start,
                jcp.nb_oc, icb_start, jcp.nb_ic);

        while (img_start < img_end) {
            int g = g_start, ocb = ocb_start, icb = icb_start;

            const int work_rem = img_end - img_start;
            const int od_e = od_s + work_rem > jcp.od ? jcp.od : od_s + work_rem;
            const int id_s = od_s * jcp.stride_d;
            const int idp = jcp.id + jcp.f_pad + jcp.back_pad;

            if (id_s < idp - jcp.back_pad - jcp.kd + 1)
            for (int w_job_loc = 0; w_job_loc < w_njobs; ++w_job_loc) {
                const size_t _oc = g * jcp.nb_oc + ocb;
                const size_t _ic = g * jcp.nb_ic + icb;

                /* TODO: put dw <-- 0 in kernel */
                if (img == img_first)
                    array_set(rw->get_local_ptr(ithr, diff_weights,
                                reducer_wei_scratchpad) +
                            w_job_loc * rw->balancer().job_size_, 0,
                            rw->balancer().job_size_);

                for (int od = od_s; od < od_e; ++od) {
                    const int id = od * jcp.stride_d;
                    if (id >= jcp.id - jcp.back_pad - jcp.kd + 1) break;

                    auto par_conv = jit_conv_call_s();
                    par_conv.src = &src[src_blk_off(src_d, img, _ic, id, 0, 0)];
                    par_conv.dst =
                        &diff_dst[src_blk_off(diff_dst_d, img, _oc, od, 0, 0)];
                    par_conv.filt = rw->get_local_ptr(ithr, diff_weights,
                            reducer_wei_scratchpad) +
                        w_job_loc * rw->balancer().job_size_;

                    kernel_->jit_ker(&par_conv);
                }
                nd_iterator_step(g, jcp.ngroups, ocb, jcp.nb_oc, icb,
                        jcp.nb_ic);
            }
            nd_iterator_jump(img_start, img_end, img, jcp.mb, od_s, jcp.od);
        }
        rw->reduce(ithr, diff_weights, reducer_wei_scratchpad);
    };

    auto ker_bias = [&](int ithr, int nthr) {
        assert(nthr == rb->balancer().nthr_);

        const int b_job_start = rb->balancer().ithr_job_off(ithr);
        const int b_njobs = rb->balancer().ithr_njobs(ithr);

        if (b_njobs == 0) return;

        /* reduction dimension */
        int img_start{0}, img_end{0};
        balance211(jcp.mb, rb->balancer().nthr_per_group_,
                rb->balancer().id_in_group(ithr), img_start, img_end);

        /* jobs */
        int g_start{0}, ocb_start{0};
        nd_iterator_init(b_job_start, g_start, jcp.ngroups, ocb_start,
                jcp.nb_oc);

        for (int img = img_start; img < img_end; ++img) {
            int g = g_start, ocb = ocb_start;
            for (int b_job_loc = 0; b_job_loc < b_njobs; ++b_job_loc) {
                const size_t _oc = g * jcp.nb_oc + ocb;

                const data_t *d_dst = &diff_dst[diff_dst_d.blk_off(img, _oc)];
                data_t *d_bias = rb->get_local_ptr(ithr, diff_bias,
                        reducer_bia_scratchpad) +
                    b_job_loc * rb->balancer().job_size_;

                if (img == img_start)
                    for (int o = 0; o < 8; ++o)
                        d_bias[o] = 0.;

                for (int dhw = 0; dhw < jcp.od * jcp.oh * jcp.ow; ++dhw) {
                    PRAGMA_OMP_SIMD()
                    for (int o = 0; o < 8; ++o)
                        d_bias[o] += d_dst[o];
                    d_dst += 8;
                }

                nd_iterator_step(g, jcp.ngroups, ocb, jcp.nb_oc);
            }
        }
        rb->reduce(ithr, diff_bias, reducer_bia_scratchpad);
    };

    parallel(0, (size_t)mkldnn_get_max_threads(), [&](const int ithr, const int nthr) {
        ker(ithr, nthr);
        if (pd()->with_bias())
            ker_bias(ithr, nthr);
    });

    /* TODO: put this in ker_bias */
    if (pd()->wants_padded_bias()) {
        assert(jcp.ngroups == 1);
        for (int oc = 0; oc < jcp.oc_without_padding; ++oc)
            diff_bias_in[oc] = diff_bias[oc];
    }
}

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
