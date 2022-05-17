/*******************************************************************************
* Copyright 2016-2020 Intel Corporation
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

/* [todo] antonvor:
 * This file contains the old plugin behavior in order to fix performance
 * problems after upgrading to OneDNN v1.6. This kernel is executed only on
 * machines with avx2 instruction set support and in the case of a fused
 * convolution. Remove after problems are fixed.
*/

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/x64/jit_generator.hpp"

#include "cpu/x64/jit_avx2_1x1_convolution_with_dw_conv.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::status;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;

#define data_blk_off(f, n, c, d, h, w) \
((ndims == 3) ? (f).blk_off(n, c, w) \
  : ((ndims == 4) ? (f).blk_off(n, c, h, w) \
                  : (f).blk_off(n, c, d, h, w)))
/* convolution forward */

void jit_avx2_1x1_convolution_with_dw_conv_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const data_t *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const data_t *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(data_t *, DNNL_ARG_DST);

    auto weights_dw = CTX_IN_MEM(
            const data_t *, DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS);
    auto bias_dw = CTX_IN_MEM(
            const data_t *, DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_BIAS);

    const auto &jcp = kernel_old_->jcp;
    const auto &jcp_dw = kernel_dw_->jcp;

    const auto post_ops_binary_rhs_arg_vec
            = binary_injector::prepare_binary_args(jcp.post_ops, ctx);
    const auto post_ops_binary_rhs_arg_vec_dw = binary_injector::prepare_binary_args(jcp_dw.post_ops, ctx, jcp.post_ops.entry_.size() + 1);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));

    auto scratchpad = ctx.get_scratchpad_grantor();

    auto rtus_space = pd()->rtus_.reduce_src_
                      ? scratchpad.get<data_t>(key_conv_rtus_space)
                      : nullptr;

    const int MB = pd()->MB();

    int ocb_work = jcp.with_dw_conv ? utils::div_up(jcp.nb_load, jcp.nb_load_blocking) : 1;
    const int work_amount = MB * jcp.ngroups * ocb_work * jcp.nb_bcast;

    auto step = [](int default_step, int remaining, int tail_step) {
        assert(default_step <= tail_step);
        return remaining < tail_step ? remaining : default_step;
    };

    auto ker = [&](const int ithr, const int nthr) {
        // TODO (Roma): remove this restriction
        assert(jcp.stride_w == 1 && jcp.stride_h == 1);

        auto compute_block_1x1 = [&](float* ws_p, int n, int g, int oh, int ow, int ih, int iw, int os, int os_block, int bcast_step, int ocb, int load_step,
                                     int num_rows) {
            auto rp = rtus_driver_t<avx2>::call_params_t();
            auto p = jit_1x1_conv_call_s();

            for (int h = 0; h < num_rows; h++) {
                ih = nstl::max((oh + h) * jcp.stride_h - jcp.t_pad, 0);

                if ((oh + h) < 0 || (oh + h) >= jcp.ih) {
                    for (int chb = ocb; chb < ocb + load_step; chb++) {
                        memset(ws_p + (((oh + h) + 1) % jcp_dw.kh) * jcp.ow * jcp.oc_block +
                               (chb - ocb) * jcp_dw.kh * jcp.ow * jcp.oc_block, 0, jcp.ow * jcp.oc_block * sizeof(float));
                    }
                } else {
                    const int _ocb = g * jcp.nb_load + ocb;

                    rp.iw_start = iw;
                    p.bcast_dim = this_block_size(os, jcp.os, bcast_step * os_block);

                    rp.os = p.bcast_dim;
                    p.load_dim = this_block_size(ocb * jcp.oc_block, jcp.oc, load_step * jcp.oc_block);

                    p.output_data = &ws_p[(((oh + h) + 1) % jcp_dw.kh) * jcp.ow * jcp.oc_block];

                    p.bias_data = &bias[_ocb * jcp.oc_block];

                    for (int icb = 0; icb < jcp.nb_reduce; icb += jcp.nb_reduce_blocking) {
                        p.first_last_flag = 0
                                            | (icb == 0 ? FLAG_REDUCE_FIRST : 0)
                                            | (icb + jcp.nb_reduce_blocking >= jcp.nb_reduce
                                               ? FLAG_REDUCE_LAST : 0);

                        p.reduce_dim = this_block_size(icb * jcp.ic_block, jcp.ic,
                                                       jcp.nb_reduce_blocking * jcp.ic_block);
                        rp.icb = p.reduce_dim / jcp.reduce_block;

                        p.load_data = &weights[pd()->with_groups()
                                               ? weights_d.blk_off(g, ocb, icb)
                                               : weights_d.blk_off(ocb, icb)];

                        const int _icb = g * jcp.nb_reduce + icb;
                        if (pd()->rtus_.reduce_src_) {
                            rp.ws = rtus_space
                                    + ithr * pd()->rtus_.space_per_thread_
                                    + _icb * jcp.is * jcp.ic_block;

                            if (ocb == 0) {
                                rp.src = src + src_d.blk_off(n, _icb, ih, iw);
                                (*rtus_driver_)(&rp);
                            }

                            p.bcast_data = rp.ws;
                        } else {
                            p.bcast_data = src + src_d.blk_off(n, _icb, ih, iw);
                        }

                        p.oc_off = _ocb * jcp.oc_block * sizeof(float);
                        p.post_ops_binary_rhs_arg_vec = post_ops_binary_rhs_arg_vec.data();

                        (*kernel_old_)(&p);
                    }
                }
            }
        };

        auto compute_row_dw = [&](const float* ws_p, int n, int ocb, int load_step, int dst_idx) {

            for (int chb = ocb; chb < ocb + load_step; chb++) {
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
                par_conv_dw.filt = &weights_dw[chb * jcp_dw.kh * jcp_dw.kw * jcp_dw.ch_block];
                par_conv_dw.bias = &bias_dw[chb * jcp_dw.ch_block];
                par_conv_dw.ur_w = (size_t)(jcp_dw.ow);
                par_conv_dw.oc_work = nstl::min((chb + 1) * jcp_dw.ch_block, (int)jcp_dw.oc) - chb*jcp_dw.ch_block;
                par_conv_dw.oc_off = chb * jcp_dw.ch_block * sizeof(float);
                par_conv_dw.post_ops_binary_rhs_arg_vec = post_ops_binary_rhs_arg_vec_dw.data();

                (*kernel_dw_)(&par_conv_dw);
            }
        };

        assert(jcp.stride_w == 1 && jcp.stride_h == 1);

        int start{0}, end{0};
        balance211(work_amount, nthr, ithr, start, end);

        auto dw_conv_buffer = scratchpad.get<data_t>(key_dw_conv_buffer);
        size_t dw_conv_buffer_size_ = (size_t)jcp_dw.kh * jcp_dw.iw * jcp_dw.ch_block * (jcp.oc / jcp.oc_block);
        auto pbuf = dw_conv_buffer + ithr * dw_conv_buffer_size_;

        const int os_block = jcp.iw;

        int iwork = start;
        while (iwork < end) {
            int n{0}, g{0}, ocbb{0}, osb{0};
            nd_iterator_init(iwork, n, MB, g, jcp.ngroups, ocbb, ocb_work, osb,
                             jcp.nb_bcast);
            int bcast_step = 1;

            const int os = osb * os_block;
            const int oh = os / jcp.ow;
            const int ow = os % jcp.ow;

            const int ih = nstl::max(oh * jcp.stride_h - jcp.t_pad, 0);
            const int iw = nstl::max(ow * jcp.stride_w - jcp.l_pad, 0);

            int ocb = ocbb * jcp.nb_load_blocking;

            const int load_step = step(jcp.nb_load_blocking,
                                       jcp.nb_load - ocb, jcp.nb_load_blocking_max);

            if (iwork == start || oh == 0) {
                bcast_step = nstl::min(1, end - iwork);
                compute_block_1x1(pbuf, n, g, oh - 1, ow, ih, iw, os, os_block, bcast_step, ocb, load_step, bcast_step + 2);
            } else {
                bcast_step = nstl::min(1, end - iwork);
                compute_block_1x1(pbuf, n, g, oh + 1, ow, ih, iw, os, os_block, bcast_step, ocb, load_step, bcast_step);
            }

            if ((oh % jcp_dw.stride_h == 0)) {
                compute_row_dw(pbuf, n, ocb, load_step, oh);
            }

            iwork += bcast_step;
        }
    };

    if (pd()->wants_padded_bias()) {
        auto padded_bias = scratchpad.get<data_t>(key_conv_padded_bias);
        utils::array_copy(padded_bias, bias, jcp.oc_without_padding);
        utils::array_set(padded_bias + jcp.oc_without_padding, 0.f,
                         jcp.oc - jcp.oc_without_padding);
        bias = padded_bias;

        auto dw_padded_bias = scratchpad.get<data_t>(key_dw_conv_padded_bias);
        utils::array_copy(dw_padded_bias, bias_dw, jcp.oc_without_padding);
        utils::array_set(dw_padded_bias + jcp.oc_without_padding, 0.f,
                         jcp.oc - jcp.oc_without_padding);
        bias_dw = dw_padded_bias;
    }

    parallel(0, ker);

    if (pd()->wants_zero_pad_dst()) ctx.zero_pad_output(DNNL_ARG_DST);
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
