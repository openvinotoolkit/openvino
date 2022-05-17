/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"

#include "common/bfloat16.hpp"

#include "jit_uni_fork_dw_convolution.hpp"
#include "cpu/x64/injectors/jit_uni_binary_injector.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::status;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;

template <cpu_isa_t isa, data_type_t src_type, data_type_t dst_type>
void jit_uni_fork_dw_convolution_fwd_t<isa, src_type, dst_type>::execute_forward(
        const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const data_t *, DNNL_ARG_WEIGHTS);
    auto dst = CTX_OUT_MEM(dst_data_t *, DNNL_ARG_DST);
    const auto &jcp = pd()->jcp_;
    const auto post_ops_binary_rhs_arg_vec
            = binary_injector::prepare_binary_args(jcp.post_ops, ctx);

    auto MB = CTX_IN_BATCH(DNNL_ARG_SRC);
    
    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));
    const memory_desc_wrapper bias_d(pd()->weights_md(1));

    f32_data_t *bias = nullptr;
    if (pd()->desc()->bias_desc.data_type == data_type::bf16) {
        auto bias_in = CTX_IN_MEM(const bf16_data_t *, DNNL_ARG_BIAS);
        bias = ctx.get_scratchpad_grantor().template get<f32_data_t>(
                key_conv_bias_bf16_convert_wsp);
        cvt_bfloat16_to_float(bias, bias_in, jcp.oc_without_padding);
        utils::array_set(bias + jcp.oc_without_padding, 0.f,
                         jcp.oc - jcp.oc_without_padding);
    } else {
        auto bias_in = CTX_IN_MEM(const f32_data_t *, DNNL_ARG_BIAS);
        if (pd()->wants_padded_bias()) {
            auto padded_bias
                    = ctx.get_scratchpad_grantor().template get<f32_data_t>(
                            key_conv_padded_bias);
            utils::array_copy(padded_bias, bias_in, jcp.oc_without_padding);
            utils::array_set(padded_bias + jcp.oc_without_padding, 0.f,
                    jcp.oc - jcp.oc_without_padding);
            bias = padded_bias;
        } else
            bias = const_cast<float*> (bias_in);
    }

    int dil_d = jcp.dilate_d + 1;
    int dil_h = jcp.dilate_h + 1;
    int dil_w = jcp.dilate_w + 1;
    int str_d = jcp.stride_d;
    int str_h = jcp.stride_h;
    int str_w = jcp.stride_w;

    const auto is_src_layout_nxc = one_of(jcp.src_tag, format_tag::nhwc, format_tag::ndhwc);
    const auto is_dst_layout_nxc = one_of(jcp.dst_tag, format_tag::nhwc, format_tag::ndhwc);

    auto kernel_params = [&](int ur_w_step, int ow, int oh, int od, int ih, int id, int kh, int kd,
            int kh_padding, int kd_padding, int ch, int ch_step, int n, int work_rem) {
        auto par_conv = jit_conv_call_s();

        const int i_l_overflow = nstl::max(0, (jcp.l_pad - ow * str_w));
        const int i_r_overflow = nstl::max(jcp.iw, (ow * str_w
            + (jcp.kw - 1)*dil_w - jcp.l_pad + 1)) - jcp.iw;

        const int iw = nstl::max((ow*str_w - jcp.l_pad
            + div_up(i_l_overflow, dil_w)*dil_w), 0);
        const int kw = div_up(i_l_overflow, dil_w);

        const int kw_padding = jcp.kw - div_up(i_l_overflow, dil_w)
            - div_up(i_r_overflow, dil_w);

        const auto ic_off_idx = is_src_layout_nxc ? ch * jcp.ch_block : ch;
        const auto oc_off_idx = is_dst_layout_nxc ? ch * jcp.ch_block : ch;

        size_t src_off = (jcp.ndims == 3) ? src_d.blk_off(n, ic_off_idx, iw) :
                         (jcp.ndims == 4) ? src_d.blk_off(n, ic_off_idx, ih, iw) : src_d.blk_off(n, ic_off_idx, id, ih, iw);
        size_t dst_off = (jcp.ndims == 3) ? dst_d.blk_off(n, oc_off_idx, ow) :
                         (jcp.ndims == 4) ? dst_d.blk_off(n, oc_off_idx, oh, ow) : dst_d.blk_off(n, oc_off_idx, od, oh, ow);
        size_t wei_off = (jcp.ndims == 3) ? weights_d.blk_off(ch, 0, 0, kw) :
                         (jcp.ndims == 4) ? weights_d.blk_off(ch, 0, 0, kh, kw) : weights_d.blk_off(ch, 0, 0, kd, kh, kw);

        par_conv.src = &src[src_off];
        par_conv.dst = &dst[dst_off];
        par_conv.filt = &weights[wei_off];

        if (bias) par_conv.bias = &bias[bias_d.blk_off(ch*jcp.ch_block)];

        par_conv.kd_padding = (size_t)nstl::max(0, kd_padding);
        par_conv.kh_padding = (size_t)nstl::max(0, kh_padding);
        par_conv.kw_padding = (size_t)nstl::max(0, kw_padding);

        par_conv.ur_w = (size_t)ur_w_step;

        assert(IMPLICATION(
                jcp.loop_order == loop_nhwcg, is_src_layout_nxc));
        // For is_src_layout_nxc maximize jit work along contiguous dim.
        par_conv.load_work = utils::this_block_size(ch * jcp.ch_block,
                                                    jcp.oc_without_padding,
                                                    (is_src_layout_nxc ? work_rem * ch_step : ch_step)
                                                    * jcp.ch_block);
        par_conv.oc_off = ch * jcp.ch_block * sizeof(float);
        par_conv.post_ops_binary_rhs_arg_vec = post_ops_binary_rhs_arg_vec.data();

        return par_conv;
    };

    const int ch_step = jcp.nb_ch_blocking;
    const int chb_work = utils::div_up(jcp.nb_ch, ch_step);

    const int work_amount = MB * chb_work * jcp.od * jcp.oh;
    const auto nthr = jcp.nthr;

    parallel(nthr, [&](const int ithr, const int nthr) {
        int start {0}, end {0};
        balance211(work_amount, nthr, ithr, start, end);

        int n {0}, chb {0}, od {0}, oh {0};
        if (jcp.loop_order == loop_ngcw)
            utils::nd_iterator_init(
                    start, n, MB, chb, chb_work, od, jcp.od, oh, jcp.oh);
        else if (jcp.loop_order == loop_nhwcg)
            utils::nd_iterator_init(
                    start, n, MB, od, jcp.od, oh, jcp.oh, chb, chb_work);
        else
            assert(!"unsupported loop order");

        auto iwork = start;
        while (iwork < end) {
            int ch = chb * ch_step;

            const int i_front_overflow = nstl::max(0, (int) (jcp.f_pad - od * str_d));
            const int i_back_overflow = nstl::max(jcp.id,
                                                  (int) (od * str_d + (jcp.kd - 1) * dil_d - jcp.f_pad + 1)) - jcp.id;

            const int i_t_overflow = nstl::max(0, (int) (jcp.t_pad - oh * str_h));
            const int i_b_overflow = nstl::max(jcp.ih,
                                               (int) (oh * str_h + (jcp.kh - 1) * dil_h - jcp.t_pad + 1)) - jcp.ih;

            const int id = nstl::max((int) (od * str_d - jcp.f_pad
                                            + div_up(i_front_overflow, dil_d) * dil_d), 0);
            const int kd = div_up(i_front_overflow, dil_d);
            const int kd_padding = jcp.kd - div_up(i_front_overflow, dil_d)
                                   - div_up(i_back_overflow, dil_d);

            const int ih = nstl::max((int) (oh * str_h - jcp.t_pad
                                            + div_up(i_t_overflow, dil_h) * dil_h), 0);
            const int kh = div_up(i_t_overflow, dil_h);
            const int kh_padding = jcp.kh - div_up(i_t_overflow, dil_h)
                                   - div_up(i_b_overflow, dil_h);

            // left border
            int ow = 0;
            int l_border = nstl::min(div_up(jcp.l_pad, str_w), jcp.ow);
            int ur_w_step = 1;
            for (; ow < l_border; ow++) {
                jit_conv_call_s par_conv = kernel_params(ur_w_step, ow, oh, od, ih, id,
                                                         kh, kd, kh_padding, kd_padding, ch, ch_step, n, end - iwork);

                (*kernel_)(&par_conv);
            }

            // main loop
            ur_w_step = (jcp.iw - (jcp.kw - 1) * dil_w + jcp.l_pad - 1)
                        / jcp.stride_w - ow + 1;
            if (ur_w_step > 0) {
                jit_conv_call_s par_conv = kernel_params(ur_w_step, ow, oh, od, ih, id,
                                                         kh, kd, kh_padding, kd_padding, ch, ch_step, n, end - iwork);

                (*kernel_)(&par_conv);

                ow += ur_w_step;
            }

            // right border
            ur_w_step = 1;
            for (; ow < jcp.ow; ow++) {
                jit_conv_call_s par_conv = kernel_params(ur_w_step, ow, oh, od, ih, id,
                                                         kh, kd, kh_padding, kd_padding, ch, ch_step, n, end - iwork);

                (*kernel_)(&par_conv);
            }

            if (jcp.loop_order == loop_ngcw) {
                ++iwork;
                utils::nd_iterator_step(n, MB, chb, chb_work, od, jcp.od, oh, jcp.oh);
            } else if (jcp.loop_order == loop_nhwcg) {
                utils::nd_iterator_jump(
                        iwork, end, n, MB, od, jcp.od, oh, jcp.oh, chb, chb_work);
            } else
                assert(!"unsupported loop order");
        }
    });

    if (pd()->wants_zero_pad_dst())
        ctx.zero_pad_output(DNNL_ARG_DST);
}

template struct jit_uni_fork_dw_convolution_fwd_t<avx512_core, data_type::bf16,
        data_type::f32>;
template struct jit_uni_fork_dw_convolution_fwd_t<avx512_core, data_type::bf16>;
template struct jit_uni_fork_dw_convolution_fwd_t<avx512_common, data_type::f32>;
template struct jit_uni_fork_dw_convolution_fwd_t<avx2, data_type::f32>;
template struct jit_uni_fork_dw_convolution_fwd_t<sse41, data_type::f32>;

template <cpu_isa_t isa, data_type_t diff_dst_type, data_type_t diff_src_type>
void jit_uni_fork_dw_convolution_bwd_data_t<isa, diff_dst_type, diff_src_type>
        ::execute_backward_data(const exec_ctx_t &ctx) const {
    auto diff_dst = CTX_IN_MEM(const diff_dst_data_t *, DNNL_ARG_DIFF_DST);
    auto weights = CTX_IN_MEM(const wei_data_t *, DNNL_ARG_WEIGHTS);
    auto diff_src = CTX_OUT_MEM(diff_src_data_t *, DNNL_ARG_DIFF_SRC);

    const auto &jcp = pd()->jcp_;
    const auto post_ops_binary_rhs_arg_vec
            = binary_injector::prepare_binary_args(jcp.post_ops, ctx);

    auto MB = CTX_IN_BATCH(DNNL_ARG_DIFF_DST);
    
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper diff_src_d(pd()->diff_src_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));

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
        par_conv.ic_off = ch * jcp.ch_block * sizeof(float);
        par_conv.post_ops_binary_rhs_arg_vec = post_ops_binary_rhs_arg_vec.data();

        return par_conv;
    };

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

                (*kernel_)(&par_conv);
            }

            // main loop
            ur_str_w = nstl::min((jcp.iw - jcp.kw + jcp.r_pad - iw)
                 / jcp.stride_w, jcp.iw);
            if (ur_str_w > 0) {
                jit_conv_call_s par_conv = kernel_params(ur_str_w, iw, oh,
                                             ih, i_t_overflow, i_b_overflow,
                                             stride_off_h, ch, ch_num, n);

                (*kernel_)(&par_conv);

                iw += ur_str_w * jcp.stride_w;
            }

            // right border
            ur_str_w = 1;
            for (; iw < jcp.iw; iw += jcp.stride_w) {
                jit_conv_call_s par_conv = kernel_params(ur_str_w, iw, oh,
                                             ih, i_t_overflow, i_b_overflow,
                                             stride_off_h, ch, ch_num, n);

                (*kernel_)(&par_conv);
            }
        }
    });
}

template struct jit_uni_fork_dw_convolution_bwd_data_t<avx512_core, data_type::bf16,
        data_type::f32>;
template struct jit_uni_fork_dw_convolution_bwd_data_t<avx512_core,
        data_type::bf16>;
template struct jit_uni_fork_dw_convolution_bwd_data_t<avx512_common,
        data_type::f32>;
template struct jit_uni_fork_dw_convolution_bwd_data_t<avx2, data_type::f32>;
template struct jit_uni_fork_dw_convolution_bwd_data_t<sse41, data_type::f32>;

}
}
}
}
