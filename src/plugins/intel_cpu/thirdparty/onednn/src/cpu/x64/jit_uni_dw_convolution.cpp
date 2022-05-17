/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
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

#include "cpu/x64/jit_uni_dw_convolution.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::status;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;

template <cpu_isa_t isa, data_type_t src_type, data_type_t dst_type>
void jit_uni_dw_convolution_fwd_t<isa, src_type, dst_type>::execute_forward(
        const exec_ctx_t &ctx) const {
    const auto &jcp = pd()->jcp_;
    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const data_t *, DNNL_ARG_WEIGHTS);
    auto dst = CTX_OUT_MEM(dst_data_t *, DNNL_ARG_DST);
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
            bias = const_cast<float *>(bias_in);
    }

    const int dil_h = jcp.dilate_h + 1;
    const int str_h = jcp.stride_h;
    const int ch_step = jcp.nb_ch_blocking;
    const int ow = 0;
    const int iw = 0;
    const int kw = 0;
    const int chb_work = utils::div_up(jcp.nb_ch, ch_step);
    const auto is_src_layout_nxc = jcp.src_tag == format_tag::nhwc;
    const auto is_dst_layout_nxc = jcp.dst_tag == format_tag::nhwc;

    const int work_amount = MB * chb_work * jcp.oh;
    const auto nthr = jcp.nthr;

    parallel(nthr, [&](const int ithr, const int nthr) {
        int start {0}, end {0};
        balance211(work_amount, nthr, ithr, start, end);

        int n {0}, chb {0}, oh {0};
        if (jcp.loop_order == loop_ngcw)
            utils::nd_iterator_init(
                    start, n, MB, chb, chb_work, oh, jcp.oh);
        else if (jcp.loop_order == loop_nhwcg)
            utils::nd_iterator_init(
                    start, n, MB, oh, jcp.oh, chb, chb_work);
        else
            assert(!"unsupported loop order");

        auto iwork = start;
        while (iwork < end) {

            int ch = chb * ch_step;

            const int i_t_overflow
                    = nstl::max(0, (int)(jcp.t_pad - oh * str_h));
            const int i_b_overflow
                    = nstl::max(jcp.ih,
                              (int)(oh * str_h + (jcp.kh - 1) * dil_h
                                      - jcp.t_pad + 1))
                    - jcp.ih;

            const int ih
                    = nstl::max((int)(oh * str_h - jcp.t_pad
                                        + div_up(i_t_overflow, dil_h) * dil_h),
                            0);
            const int kh = div_up(i_t_overflow, dil_h);
            const int kh_padding = jcp.kh - div_up(i_t_overflow, dil_h)
                    - div_up(i_b_overflow, dil_h);

            const auto ic_off_idx = is_src_layout_nxc ? ch * jcp.ch_block : ch;
            const auto oc_off_idx = is_dst_layout_nxc ? ch * jcp.ch_block : ch;

            auto par_conv = jit_conv_call_s();
            par_conv.src = jcp.is_fused_conv
                    ? src
                    : &src[src_d.blk_off(n, ic_off_idx, ih, iw)];
            par_conv.dst = &dst[dst_d.blk_off(n, oc_off_idx, oh, ow)];

            par_conv.filt = &weights[weights_d.blk_off(ch, 0, 0, kh, kw)];
            if (bias) par_conv.bias = &bias[bias_d.blk_off(ch * jcp.ch_block)];

            par_conv.kh_padding = (size_t)nstl::max(0, kh_padding);

            assert(IMPLICATION(
                    jcp.loop_order == loop_nhwcg, is_src_layout_nxc));
            // For is_src_layout_nxc maximize jit work along contiguous dim.
            const int work_rem = end - iwork;
            par_conv.load_work = utils::this_block_size(ch * jcp.ch_block,
                    jcp.oc_without_padding,
                    (is_src_layout_nxc ? work_rem * ch_step : ch_step)
                            * jcp.ch_block);

            par_conv.oc_l_off = ch * jcp.ch_block;
            par_conv.post_ops_binary_rhs_arg_vec
                    = post_ops_binary_rhs_arg_vec.data();
            par_conv.dst_orig = dst;
            par_conv.oc_off = ch * jcp.ch_block * sizeof(float);

            (*kernel_)(&par_conv);

            if (jcp.loop_order == loop_ngcw) {
                ++iwork;
                utils::nd_iterator_step(n, MB, chb, chb_work, oh, jcp.oh);
            } else if (jcp.loop_order == loop_nhwcg) {
                utils::nd_iterator_jump(
                        iwork, end, n, MB, oh, jcp.oh, chb, chb_work);
            } else
                assert(!"unsupported loop order");
        }
    });

    if (pd()->wants_zero_pad_dst()) ctx.zero_pad_output(DNNL_ARG_DST);
}

template struct jit_uni_dw_convolution_fwd_t<avx512_core, data_type::bf16,
        data_type::f32>;
template struct jit_uni_dw_convolution_fwd_t<avx512_core, data_type::bf16>;
template struct jit_uni_dw_convolution_fwd_t<avx512_common, data_type::f32>;
template struct jit_uni_dw_convolution_fwd_t<avx2, data_type::f32>;
template struct jit_uni_dw_convolution_fwd_t<sse41, data_type::f32>;

template <cpu_isa_t isa, data_type_t diff_dst_type, data_type_t diff_src_type>
void jit_uni_dw_convolution_bwd_data_t<isa, diff_dst_type,
        diff_src_type>::execute_backward_data(const exec_ctx_t &ctx) const {
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
                                 int i_t_overflow, int i_b_overflow,
                                 int stride_off_h, int ch, int n,
                                 int work_remaining) {
        auto par_conv = jit_conv_call_s();
        const bool is_dsrc_layout_nxc
                = utils::one_of(jcp.src_tag, format_tag::nwc, format_tag::nhwc);
        const bool is_ddst_layout_nxc
                = utils::one_of(jcp.dst_tag, format_tag::nwc, format_tag::nhwc);
        const int nb_ch_blocking = jcp.nb_ch_blocking;

        const int i_l_overflow = nstl::max(0, (jcp.kw - 1 - iw - jcp.l_pad));
        const int i_r_overflow
                = nstl::max(0, (jcp.kw - 1 - (jcp.iw - 1 - iw) - jcp.r_pad));

        int ow = iw + jcp.l_pad - i_r_overflow;
        int stride_off_w = ow % jcp.stride_w;
        ow /= jcp.stride_w;

        const int ic_offset = is_dsrc_layout_nxc ? ch * jcp.ch_block : ch;
        par_conv.src = &diff_src[diff_src_d.blk_off(n, ic_offset, ih, iw)];
        const int oc_offset = is_ddst_layout_nxc ? ch * jcp.ch_block : ch;
        par_conv.dst = &diff_dst[diff_dst_d.blk_off(n, oc_offset, oh, ow)];
        par_conv.filt = &weights[weights_d.blk_off(ch, 0, 0,
                i_b_overflow + stride_off_h, i_r_overflow + stride_off_w)];

        par_conv.kh_padding = nstl::max(
                0, jcp.kh - i_t_overflow - i_b_overflow - stride_off_h);
        par_conv.kw_padding = nstl::max(
                0, jcp.kw - i_l_overflow - i_r_overflow - stride_off_w);

        par_conv.ur_str_w = ur_str_w;

        const size_t ch_work = (is_ddst_layout_nxc ? work_remaining : 1)
                * nb_ch_blocking * jcp.ch_block;
        const size_t load_work
                = utils::this_block_size(static_cast<size_t>(ch * jcp.ch_block),
                        static_cast<size_t>(jcp.oc), ch_work);
        par_conv.ch_blocks = load_work;

        par_conv.ic_off = ch * jcp.ch_block * sizeof(float);
        par_conv.post_ops_binary_rhs_arg_vec
                = post_ops_binary_rhs_arg_vec.data();

        return par_conv;
    };

    const int aux_w
            = nstl::min(jcp.iw, jcp.iw - jcp.kw + jcp.r_pad + jcp.stride_w);
    const int chb_work = utils::div_up(jcp.nb_ch, jcp.nb_ch_blocking);
    const dim_t work_amount = MB * chb_work * jcp.ih;
    const auto nthr = jcp.nthr;

    parallel(nthr, [&](const int ithr, const int nthr) {
        dim_t start {0}, end {0};
        balance211(work_amount, nthr, ithr, start, end);
        dim_t n {0}, chb {0}, ih {0};
        if (jcp.loop_order == loop_ngcw)
            utils::nd_iterator_init(
                    start, n, MB, chb, chb_work, ih, jcp.ih);
        else if (jcp.loop_order == loop_nhwcg)
            utils::nd_iterator_init(
                    start, n, MB, ih, jcp.ih, chb, chb_work);
        else
            assert(!"unsupported loop order");

        auto iwork = start;
        while (iwork < end) {
            int ch = chb * jcp.nb_ch_blocking;

            const int work_rem = end - iwork;
            const dim_t i_t_overflow
                    = nstl::max(dim_t(0), jcp.kh - 1 - ih - jcp.t_pad);
            const dim_t i_b_overflow = nstl::max(
                    dim_t(0), jcp.kh - 1 - (jcp.ih - 1 - ih) - jcp.b_pad);

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
                            ih, i_t_overflow, i_b_overflow, stride_off_h, ch, n,
                            work_rem);

                    (*kernel_)(&par_conv);
                }

                // main loop
                ur_str_w = (aux_w - iw) / jcp.stride_w;
                if (ur_str_w > 0) {
                    jit_conv_call_s par_conv = kernel_params(ur_str_w, iw, oh,
                            ih, i_t_overflow, i_b_overflow, stride_off_h, ch, n,
                            work_rem);

                    (*kernel_)(&par_conv);

                    iw += ur_str_w * jcp.stride_w;
                }

                // right border
                ur_str_w = 1;
                for (; iw < jcp.iw; iw += jcp.stride_w) {
                    jit_conv_call_s par_conv = kernel_params(ur_str_w, iw, oh,
                            ih, i_t_overflow, i_b_overflow, stride_off_h, ch, n,
                            work_rem);

                    (*kernel_)(&par_conv);
                }
            }
            if (jcp.loop_order == loop_ngcw) {
                ++iwork;
                utils::nd_iterator_step(n, MB, chb, chb_work, ih, jcp.ih);
            } else if (jcp.loop_order == loop_nhwcg) {
                utils::nd_iterator_jump(
                        iwork, end, n, MB, ih, jcp.ih, chb, chb_work);
            } else
                assert(!"unsupported loop order");
        }
    });
}

template struct jit_uni_dw_convolution_bwd_data_t<avx512_core, data_type::bf16,
        data_type::f32>;
template struct jit_uni_dw_convolution_bwd_data_t<avx512_core, data_type::bf16>;
template struct jit_uni_dw_convolution_bwd_data_t<avx512_common,
        data_type::f32>;
template struct jit_uni_dw_convolution_bwd_data_t<avx2, data_type::f32>;
template struct jit_uni_dw_convolution_bwd_data_t<sse41, data_type::f32>;

template <cpu_isa_t isa, data_type_t src_type, data_type_t diff_weights_type>
jit_uni_dw_convolution_bwd_weights_t<isa, src_type, diff_weights_type>::
        jit_uni_dw_convolution_bwd_weights_t(const pd_t *apd)
    : primitive_t(apd), acc_ker_(nullptr), kernel_(nullptr) {}

template <cpu_isa_t isa, data_type_t src_type, data_type_t diff_weights_type>
void jit_uni_dw_convolution_bwd_weights_t<isa, src_type,
        diff_weights_type>::execute_backward_weights_nxc(const exec_ctx_t &ctx)
        const {
    const auto &jcp = pd()->jcp_;

    auto diff_dst = CTX_IN_MEM(const diff_dst_data_t *, DNNL_ARG_DIFF_DST);
    auto src = CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC);
    auto diff_weights
            = CTX_OUT_MEM(diff_weights_data_t *, DNNL_ARG_DIFF_WEIGHTS);

    auto diff_wei_reduction_buffer
            = ctx.get_scratchpad_grantor().template get<f32_data_t>(
                    key_conv_wei_reduction);
    auto diff_bias_reduction_buffer
            = ctx.get_scratchpad_grantor().template get<f32_data_t>(
                    key_conv_bia_reduction);

    auto diff_bias_f32_to_bf16_accum
            = ctx.get_scratchpad_grantor().template get<f32_data_t>(
                    key_conv_bias_bf16_convert_wsp);
    float *diff_bias = jcp.bia_dt == data_type::bf16
            ? diff_bias_f32_to_bf16_accum
            : CTX_OUT_MEM(f32_data_t *, DNNL_ARG_DIFF_BIAS);

    const int ch_block = jcp.ch_block;
    parallel(jcp.nthr, [&](const int ithr, const int nthr) {
        auto conv_params = jit_dw_conv_call_s();
        const int h_block_size = jcp.oh_blk_size;

        const int ch_outer_blocks
                = utils::div_up(jcp.nb_ch, jcp.nb_ch_blocking);
        const int ithr_g = ithr % jcp.nthr_g;
        int g_start {0}, g_end {0};
        balance211(ch_outer_blocks, jcp.nthr_g, ithr_g, g_start, g_end);

        const int ithr_mb = (ithr / jcp.nthr_g) % jcp.nthr_mb;
        int mb_start {0}, mb_end {0};
        balance211(jcp.mb, jcp.nthr_mb, ithr_mb, mb_start, mb_end);

        const int ithr_oh = (ithr / (jcp.nthr_mb * jcp.nthr_g)) % jcp.nthr_oh;
        const int nb_oh = div_up(jcp.oh, jcp.oh_blk_size);
        int nb_oh_start {0}, nb_oh_end {0};
        balance211(nb_oh, jcp.nthr_oh, ithr_oh, nb_oh_start, nb_oh_end);

        const size_t wei_size
                = utils::rnd_up(jcp.ngroups, jcp.ch_block) * jcp.kh * jcp.kw;
        const bool main_thread = ithr_mb == 0 && ithr_oh == 0;
        const int offset_wei_buffer
                = diff_weights_type == data_type::f32 ? 1 : 0;
        const int ithr_block = ithr_mb * jcp.nthr_oh + ithr_oh;
        f32_data_t *ithr_diff_weights
                = (main_thread && diff_weights_type == data_type::f32)
                ? (f32_data_t *)diff_weights
                : diff_wei_reduction_buffer
                        + static_cast<size_t>(
                                (ithr_block - offset_wei_buffer) * wei_size);

        const size_t filter_g_step
                = static_cast<size_t>(jcp.kh * jcp.kw * jcp.ch_block);
        const size_t src_h_step = static_cast<size_t>(jcp.iw * jcp.ngroups);
        const size_t ddst_h_step = static_cast<size_t>(jcp.ow * jcp.ngroups);
        const size_t bias_size = static_cast<size_t>(jcp.ngroups);
        auto ithr_diff_bias = main_thread
                ? diff_bias
                : diff_bias_reduction_buffer ? diff_bias_reduction_buffer
                                + (ithr_block - 1) * bias_size
                                             : nullptr;
        const int g_step = jcp.nb_ch_blocking;
        for (int g_ = g_start; g_ < g_end; ++g_) {
            const int g = g_ * jcp.nb_ch_blocking;
            unsigned char last_g_flag
                    = (g + g_step) >= jcp.nb_ch ? FLAG_OC_LAST : 0;
            unsigned char zero_filter_flag = FLAG_ZERO_FILTER;
            unsigned char zero_bias_flag = jcp.with_bias ? FLAG_ZERO_BIAS : 0;
            for (int mb = mb_start; mb < mb_end; mb++) {
                for (int nb_oh = nb_oh_start; nb_oh < nb_oh_end; ++nb_oh) {
                    const int oh_s = nb_oh * h_block_size;
                    const int h_work = nstl::min(h_block_size, jcp.oh - oh_s);
                    const int oh_e = oh_s + h_work;
                    const int ih = -jcp.t_pad + oh_s * jcp.stride_h;
                    const int kh_top_overflow = nstl::max(0, -ih);
                    const int kh_bottom_overflow
                            = nstl::max(0, ih - jcp.ih + jcp.kh);
                    const int kh_padding_offset
                            = nstl::min(jcp.kh - 1, kh_top_overflow);
                    conv_params.kh_count
                            = jcp.kh - kh_top_overflow - kh_bottom_overflow;
                    conv_params.filter_pad_off
                            = static_cast<size_t>(kh_padding_offset * jcp.kw
                                    * ch_block * jcp.typesize_out);
                    const size_t filter_g_offset
                            = static_cast<size_t>(g) * filter_g_step;
                    conv_params.filter = &ithr_diff_weights[filter_g_offset];

                    const size_t g_offset
                            = static_cast<size_t>(g * jcp.ch_block);
                    const size_t src_offset = static_cast<size_t>(mb * jcp.ih
                                                      + ih + kh_top_overflow)
                            * src_h_step;
                    conv_params.input = &src[src_offset + g_offset];
                    const size_t diff_dst_off
                            = static_cast<size_t>(mb * jcp.oh + oh_s)
                            * ddst_h_step;
                    conv_params.output = &diff_dst[diff_dst_off + g_offset];
                    conv_params.oh_index = oh_s;
                    conv_params.oh_count = oh_e;
                    if (jcp.with_bias)
                        conv_params.bias = &ithr_diff_bias[g_offset];

                    conv_params.exec_flags
                            = zero_filter_flag | zero_bias_flag | last_g_flag;
                    (*kernel_)(&conv_params);

                    // flags are only needed during the first kernel call
                    zero_filter_flag &= ~FLAG_ZERO_FILTER;
                    zero_bias_flag &= ~FLAG_ZERO_BIAS;
                }
            }
        }
    });
}

template <cpu_isa_t isa, data_type_t src_type, data_type_t diff_weights_type>
void jit_uni_dw_convolution_bwd_weights_t<isa, src_type,
        diff_weights_type>::execute_backward_weights(const exec_ctx_t &ctx)
        const {
    auto diff_dst = CTX_IN_MEM(const diff_dst_data_t *, DNNL_ARG_DIFF_DST);
    auto src = CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC);
    auto diff_weights
            = CTX_OUT_MEM(diff_weights_data_t *, DNNL_ARG_DIFF_WEIGHTS);

    auto diff_wei_reduction_buf
            = ctx.get_scratchpad_grantor().template get<f32_data_t>(
                    key_conv_wei_reduction);
    auto diff_bia_reduction_buf
            = ctx.get_scratchpad_grantor().template get<f32_data_t>(
                    key_conv_bia_reduction);

    const auto &jcp = pd()->jcp_;

    float *diff_bias = nullptr;
    if (jcp.bia_dt == data_type::bf16) {
        diff_bias = ctx.get_scratchpad_grantor().template get<f32_data_t>(
                key_conv_bias_bf16_convert_wsp);
    } else {
        diff_bias = CTX_OUT_MEM(f32_data_t *, DNNL_ARG_DIFF_BIAS);
    }

    const size_t wei_size = jcp.ngroups * jcp.kh * jcp.kw;
    const size_t bias_size = jcp.with_bias ? jcp.ngroups : 0;

    const int ch_block = jcp.ch_block;

    auto set_kernel_params
            = [&](jit_dw_conv_call_s *conv_params, const int batch,
                      const int group, const int oh_start, const int work_size,
                      const unsigned char exec_flag, const size_t kh_padding,
                      const size_t filter_off) {
                  const int tpad_underflow_off = jcp.t_pad - filter_off;

                  conv_params->exec_flags = exec_flag;
                  conv_params->kh_count = jcp.kh - kh_padding;

                  const int oh_s = oh_start;
                  const int oh_e = oh_start + work_size;
                  const int ih_s = oh_s * jcp.stride_h;

                  conv_params->filter_pad_off
                          = filter_off * jcp.kw * ch_block * jcp.typesize_out;
                  conv_params->oh_index = oh_s;
                  conv_params->oh_count = oh_e;

                  size_t diff_dst_off
                          = ((batch * (jcp.ngroups / ch_block) + group) * jcp.oh
                                    + oh_start)
                          * jcp.ow;

                  size_t src_off
                          = ((batch * (jcp.ngroups / ch_block) + group) * jcp.ih
                                    + ih_s - tpad_underflow_off)
                          * jcp.iw;

                  conv_params->output = &diff_dst[diff_dst_off * ch_block];
                  conv_params->input = &src[src_off * ch_block];
              };

    parallel(jcp.nthr, [&](const int ithr, const int nthr) {
        assert(nthr == jcp.nthr);

        auto conv_params = jit_dw_conv_call_s();
        const int h_block_size = jcp.oh_blk_size;
        const int nb_ch = jcp.nb_ch;

        /* assign iteration space to thread */
        const int ithr_g = ithr % jcp.nthr_g;
        const int ithr_mb = (ithr / jcp.nthr_g) % jcp.nthr_mb;

        /* split dimensions */
        int g_start {0}, g_end {0};
        balance211(nb_ch, jcp.nthr_g, ithr_g, g_start, g_end);

        int mb_start {0}, mb_end {0};
        balance211(jcp.mb, jcp.nthr_mb, ithr_mb, mb_start, mb_end);

        auto i_mb
                = diff_weights_type == data_type::bf16 ? ithr_mb : ithr_mb - 1;
        f32_data_t *diff_wei
                = (ithr_mb == 0 && diff_weights_type == data_type::f32)
                ? (f32_data_t *)diff_weights
                : diff_wei_reduction_buf + i_mb * wei_size;

        auto diff_bia = ithr_mb == 0
                ? diff_bias
                : diff_bia_reduction_buf + (ithr_mb - 1) * bias_size;

        for (int g = g_start; g < g_end; ++g) {
            unsigned char last_g_flag = g == nb_ch - 1 ? FLAG_OC_LAST : 0;
            unsigned char zero_filter_flag = FLAG_ZERO_FILTER;
            unsigned char zero_bias_flag = jcp.with_bias ? FLAG_ZERO_BIAS : 0;

            size_t diff_wei_off = g * jcp.kh * jcp.kw;
            conv_params.filter = &diff_wei[diff_wei_off * ch_block];

            if (jcp.with_bias) conv_params.bias = &diff_bia[g * ch_block];

            for (int mb = mb_start; mb < mb_end; ++mb) {
                int oh = 0;
                while (oh < jcp.oh) {
                    const int h_work = nstl::min(h_block_size, jcp.oh - oh);
                    auto kh_t_padding = nstl::max(0, jcp.t_pad - oh);
                    auto kh_b_padding
                            = (oh * jcp.stride_h + jcp.kh > jcp.ih + jcp.t_pad)
                            ? nstl::max(jcp.b_pad - (h_work - 1), 0)
                            : 0;

                    set_kernel_params(&conv_params, mb, g, oh, h_work,
                            zero_filter_flag | zero_bias_flag | last_g_flag,
                            kh_t_padding + kh_b_padding, kh_t_padding);
                    (*kernel_)(&conv_params);

                    zero_bias_flag &= ~FLAG_ZERO_BIAS;
                    zero_filter_flag &= ~FLAG_ZERO_FILTER;
                    oh += h_work;
                }
            }
        }
    });
}

/* TODO: Performing a Parallel Reduction could potentially improve performance;
 * this should be explored in the future if further optimizations are required.
 */
template <>
void jit_uni_dw_convolution_bwd_weights_t<avx512_core,
        data_type::bf16>::execute_reduction(const exec_ctx_t &ctx) const {

    const auto &jcp = pd()->jcp_;
    assert(jcp.dwei_dt == data_type::bf16);

    auto diff_wei_reduction_buf
            = ctx.get_scratchpad_grantor().template get<f32_data_t>(
                    key_conv_wei_reduction);
    auto diff_bia_reduction_buf
            = ctx.get_scratchpad_grantor().template get<f32_data_t>(
                    key_conv_bia_reduction);
    auto diff_weights
            = CTX_OUT_MEM(diff_weights_data_t *, DNNL_ARG_DIFF_WEIGHTS);
    auto diff_bias_f32_to_bf16_accum
            = ctx.get_scratchpad_grantor().template get<f32_data_t>(
                    key_conv_bias_bf16_convert_wsp);
    float *diff_bias = jcp.bia_dt == data_type::bf16
            ? diff_bias_f32_to_bf16_accum
            : CTX_OUT_MEM(f32_data_t *, DNNL_ARG_DIFF_BIAS);

    const size_t wei_size
            = utils::rnd_up(jcp.ngroups, jcp.ch_block) * jcp.kh * jcp.kw;
    const size_t bias_size = jcp.with_bias ? jcp.ngroups : 0;
    const int ch_block = jcp.ch_block;

    /* Apply single-threaded 'mb' reduction */
    if (jcp.with_bias && jcp.nthr_mb > 1) {
        for (int thr_mb = 1; thr_mb < jcp.nthr_mb; ++thr_mb) {
            size_t b_accum_offset = (thr_mb - 1) * bias_size;
            const int bias_ch_tail = jcp.ch_tail;
            const int nb_ch = bias_ch_tail > 0 ? jcp.nb_ch - 1 : jcp.nb_ch;

            for (int g = 0; g < nb_ch; ++g) {
                /* Reduction on Bias */
                PRAGMA_OMP_SIMD()
                for (int g_block = 0; g_block < ch_block; ++g_block) {
                    size_t bias_offset = g * ch_block + g_block;
                    diff_bias[bias_offset]
                            += diff_bia_reduction_buf[b_accum_offset
                                    + bias_offset];
                }
            }
            for (int g = 0; g < bias_ch_tail; ++g) {
                size_t bias_offset = static_cast<size_t>(nb_ch * ch_block + g);
                diff_bias[bias_offset]
                        += diff_bia_reduction_buf[b_accum_offset + bias_offset];
            }
        }
    }
    if (jcp.bia_dt == data_type::bf16) {
        auto diff_bias_in = CTX_OUT_MEM(bf16_data_t *, DNNL_ARG_DIFF_BIAS);
        cvt_float_to_bfloat16(diff_bias_in, diff_bias, jcp.oc_without_padding);
    }
    /* Apply single-threaded 'mb' reduction */
    if (jcp.nthr_mb > 1) {
        for (int thr_mb = 2; thr_mb < jcp.nthr_mb; ++thr_mb) {
            size_t mb_accum_offset = thr_mb * wei_size;
            acc_ker_->accumulate(&diff_wei_reduction_buf[0],
                    &diff_wei_reduction_buf[mb_accum_offset], wei_size);
        }
        add_floats_and_cvt_to_bfloat16((bfloat16_t *)&(diff_weights[0]),
                (float *)&diff_wei_reduction_buf[0],
                (float *)&diff_wei_reduction_buf[wei_size], wei_size);
    } else {
        cvt_float_to_bfloat16((bfloat16_t *)&(diff_weights[0]),
                (const float *)&(diff_wei_reduction_buf[0]), wei_size);
    }
}

template <>
void jit_uni_dw_convolution_bwd_weights_t<sse41,
        data_type::f32>::execute_reduction(const exec_ctx_t &ctx) const {

    auto diff_weights = CTX_OUT_MEM(f32_data_t *, DNNL_ARG_DIFF_WEIGHTS);
    auto diff_bias = CTX_OUT_MEM(f32_data_t *, DNNL_ARG_DIFF_BIAS);
    auto diff_wei_reduction_buffer
            = ctx.get_scratchpad_grantor().template get<f32_data_t>(
                    key_conv_wei_reduction);
    auto diff_bias_reduction_buffer
            = ctx.get_scratchpad_grantor().template get<f32_data_t>(
                    key_conv_bia_reduction);

    const auto &jcp = pd()->jcp_;

    /* Apply single-threaded 'mb' reduction */
    for (int thr_mb = 1; thr_mb < jcp.nthr_mb; ++thr_mb) {
        const int ch_block = jcp.ch_block;
        const size_t wei_size
                = static_cast<size_t>(jcp.ngroups * jcp.kh * jcp.kw);
        const size_t mb_accum_offset = (thr_mb - 1) * wei_size;
        const size_t bias_size = jcp.ngroups;
        const size_t b_accum_offset = (thr_mb - 1) * bias_size;

        const int bias_ch_tail = jcp.ch_tail;
        const int nb_ch = bias_ch_tail > 0 ? jcp.nb_ch - 1 : jcp.nb_ch;
        for (int g = 0; g < nb_ch; ++g) {
            if (jcp.with_bias) {
                PRAGMA_OMP_SIMD()
                for (int g_block = 0; g_block < ch_block; ++g_block) {
                    const size_t bias_offset
                            = static_cast<size_t>(g * ch_block + g_block);
                    diff_bias[bias_offset]
                            += diff_bias_reduction_buffer[b_accum_offset
                                    + bias_offset];
                }
            }
            for_(int kh = 0; kh < jcp.kh; ++kh)
            for (int kw = 0; kw < jcp.kw; ++kw) {
                const size_t wei_sp_offset = (g * jcp.kh + kh) * jcp.kw + kw;
                PRAGMA_OMP_SIMD()
                for (int g_block = 0; g_block < ch_block; ++g_block) {
                    const size_t wei_offset = static_cast<size_t>(
                            wei_sp_offset * ch_block + g_block);
                    diff_weights[wei_offset]
                            += diff_wei_reduction_buffer[mb_accum_offset
                                    + wei_offset];
                }
            }
        }
        // handle reduction for channel tail
        if (jcp.with_bias) {
            for (int g = 0; g < bias_ch_tail; ++g) {
                const size_t bias_offset
                        = static_cast<size_t>(nb_ch * ch_block + g);
                diff_bias[bias_offset]
                        += diff_bias_reduction_buffer[b_accum_offset
                                + bias_offset];
            }
        }
        if (bias_ch_tail > 0) {
            for_(int kh = 0; kh < jcp.kh; ++kh)
            for (int kw = 0; kw < jcp.kw; ++kw) {
                const size_t wei_sp_offset = static_cast<size_t>(
                        ((nb_ch * jcp.kh + kh) * jcp.kw + kw) * ch_block);
                for (int g = 0; g < bias_ch_tail; ++g) {
                    const size_t wei_offset = wei_sp_offset + g;
                    diff_weights[wei_offset]
                            += diff_wei_reduction_buffer[mb_accum_offset
                                    + wei_offset];
                }
            }
        }
    }
}

template <cpu_isa_t isa, data_type_t src_type, data_type_t diff_weights_type>
void jit_uni_dw_convolution_bwd_weights_t<isa, src_type,
        diff_weights_type>::execute_reduction(const exec_ctx_t &ctx) const {

    const auto &jcp = pd()->jcp_;
    assert(everyone_is(data_type::f32, diff_weights_type, jcp.dwei_dt));

    auto diff_weights = CTX_OUT_MEM(f32_data_t *, DNNL_ARG_DIFF_WEIGHTS);
    auto diff_wei_reduction_buffer
            = ctx.get_scratchpad_grantor().template get<f32_data_t>(
                    key_conv_wei_reduction);
    auto diff_bias_reduction_buffer
            = ctx.get_scratchpad_grantor().template get<f32_data_t>(
                    key_conv_bia_reduction);
    auto diff_bias_f32_to_bf16_accum
            = ctx.get_scratchpad_grantor().template get<f32_data_t>(
                    key_conv_bias_bf16_convert_wsp);
    float *diff_bias = jcp.bia_dt == data_type::bf16
            ? diff_bias_f32_to_bf16_accum
            : CTX_OUT_MEM(f32_data_t *, DNNL_ARG_DIFF_BIAS);

    /* Apply single-threaded 'mb' reduction */
    for (int thr_mb = 1; thr_mb < jcp.nthr_mb; ++thr_mb) {
        const int ch_block = jcp.ch_block;
        const size_t wei_size
                = static_cast<size_t>(jcp.ngroups * jcp.kh * jcp.kw);
        const size_t mb_accum_offset = (thr_mb - 1) * wei_size;
        const size_t bias_size = jcp.ngroups;
        const size_t b_accum_offset = (thr_mb - 1) * bias_size;

        if (jcp.with_bias) { // Reduction on Bias:
            const int bias_ch_tail = jcp.ch_tail;
            const int nb_ch = bias_ch_tail > 0 ? jcp.nb_ch - 1 : jcp.nb_ch;
            for (int g = 0; g < nb_ch; ++g) {
                PRAGMA_OMP_SIMD()
                for (int g_block = 0; g_block < ch_block; ++g_block) {
                    const size_t bias_offset
                            = static_cast<size_t>(g * ch_block + g_block);
                    diff_bias[bias_offset]
                            += diff_bias_reduction_buffer[b_accum_offset
                                    + bias_offset];
                }
            }
            // handle reduction for channel tail
            for (int g = 0; g < bias_ch_tail; g++) {
                const size_t bias_offset
                        = static_cast<size_t>(nb_ch * ch_block + g);
                diff_bias[bias_offset]
                        += diff_bias_reduction_buffer[b_accum_offset
                                + bias_offset];
            }
        }
        acc_ker_->accumulate(&diff_weights[0],
                &diff_wei_reduction_buffer[mb_accum_offset], wei_size);
    }

    if (jcp.bia_dt == data_type::bf16) {
        auto diff_bias_in = CTX_OUT_MEM(bf16_data_t *, DNNL_ARG_DIFF_BIAS);
        cvt_float_to_bfloat16(diff_bias_in, diff_bias, jcp.ngroups);
    }
}

template <cpu_isa_t isa, data_type_t src_type, data_type_t diff_weights_type>
void jit_uni_dw_convolution_bwd_weights_t<isa, src_type,
        diff_weights_type>::execute_reduction_nxc(const exec_ctx_t &ctx) const {

    const auto &jcp = pd()->jcp_;
    auto diff_weights = CTX_OUT_MEM(f32_data_t *, DNNL_ARG_DIFF_WEIGHTS);
    auto diff_wei_reduction_buffer
            = ctx.get_scratchpad_grantor().template get<f32_data_t>(
                    key_conv_wei_reduction);
    auto diff_bia_reduction_buffer
            = ctx.get_scratchpad_grantor().template get<f32_data_t>(
                    key_conv_bia_reduction);
    auto diff_bias_f32_to_bf16_accum
            = ctx.get_scratchpad_grantor().template get<f32_data_t>(
                    key_conv_bias_bf16_convert_wsp);
    float *diff_bias = jcp.bia_dt == data_type::bf16
            ? diff_bias_f32_to_bf16_accum
            : CTX_OUT_MEM(f32_data_t *, DNNL_ARG_DIFF_BIAS);

    const size_t wei_size = static_cast<size_t>(
            utils::rnd_up(jcp.ngroups, jcp.ch_block) * jcp.kh * jcp.kw);

    // TODO: maybe add 'KH' as another parallel dimension to increase partition
    // space
    parallel_nd(jcp.nb_ch, [&](int NB_CH) {
        const size_t nb_ch_step
                = static_cast<size_t>(jcp.kh * jcp.kw * jcp.ch_block);
        const size_t wei_offset = NB_CH * nb_ch_step;

        f32_data_t *ithr_diff_weights = diff_weights_type == data_type::f32
                ? (f32_data_t *)&diff_weights[wei_offset]
                : &diff_wei_reduction_buffer[wei_offset];
        auto ithr_dwei_reduction_buff = &diff_wei_reduction_buffer[wei_offset];

        const int thr_work = jcp.nthr_mb * jcp.nthr_oh;
        for (int ithr_reduction = 0; ithr_reduction < thr_work - 1;
                ++ithr_reduction) {
            const int mb_ithr = ithr_reduction % jcp.nthr_mb;
            const int oh_ithr = (ithr_reduction / jcp.nthr_mb) % jcp.nthr_oh;
            const size_t ithr_offset
                    = static_cast<size_t>(mb_ithr * jcp.nthr_oh + oh_ithr);
            const int offset_wei_buffer
                    = diff_weights_type == data_type::bf16 ? 1 : 0;
            const size_t reduction_offset
                    = (ithr_offset + offset_wei_buffer) * wei_size;
            const size_t reduction_size
                    = static_cast<size_t>(jcp.kh * jcp.kw * jcp.ch_block);
            acc_ker_->accumulate(&ithr_diff_weights[0],
                    &ithr_dwei_reduction_buff[reduction_offset],
                    reduction_size);

            const bool compute_bias = jcp.with_bias;
            const int ch_block = jcp.ch_block;
            const size_t bias_size = jcp.ngroups;
            const size_t bias_accum_offset = ithr_offset * bias_size;
            if (compute_bias) {
                const size_t nb_ch_offset = NB_CH * ch_block;
                const int bias_ch_tail = jcp.ch_tail;
                const bool compute_ch_tail
                        = (NB_CH == jcp.nb_ch - 1) && bias_ch_tail > 0;
                if (!compute_ch_tail) {
                    PRAGMA_OMP_SIMD()
                    for (int g_block = 0; g_block < ch_block; ++g_block) {
                        const size_t bias_offset
                                = static_cast<size_t>(nb_ch_offset + g_block);
                        diff_bias[bias_offset]
                                += diff_bia_reduction_buffer[bias_accum_offset
                                        + bias_offset];
                    }
                } else {
                    // handle reduction for channel tail
                    for (int g = 0; g < bias_ch_tail; g++) {
                        const size_t bias_offset
                                = static_cast<size_t>(nb_ch_offset + g);
                        diff_bias[bias_offset]
                                += diff_bia_reduction_buffer[bias_accum_offset
                                        + bias_offset];
                    }
                }
            }
        }
    });

    if (diff_weights_type == data_type::bf16) {
        cvt_float_to_bfloat16((bfloat16_t *)&(diff_weights[0]),
                (const float *)&(diff_wei_reduction_buffer[0]), wei_size);
    }

    if (jcp.bia_dt == data_type::bf16) {
        auto diff_bias_in = CTX_OUT_MEM(bf16_data_t *, DNNL_ARG_DIFF_BIAS);
        cvt_float_to_bfloat16(diff_bias_in, diff_bias, jcp.oc_without_padding);
    }
}

template <>
void jit_uni_dw_convolution_bwd_weights_t<sse41,
        data_type::f32>::execute_reduction_nxc(const exec_ctx_t &ctx) const {

    auto diff_weights = CTX_OUT_MEM(f32_data_t *, DNNL_ARG_DIFF_WEIGHTS);
    auto diff_bias = CTX_OUT_MEM(f32_data_t *, DNNL_ARG_DIFF_BIAS);

    auto diff_wei_reduction_buffer
            = ctx.get_scratchpad_grantor().template get<f32_data_t>(
                    key_conv_wei_reduction);
    auto diff_bia_reduction_buffer
            = ctx.get_scratchpad_grantor().template get<f32_data_t>(
                    key_conv_bia_reduction);

    const auto &jcp = pd()->jcp_;

    const int thr_work = jcp.nthr_mb * jcp.nthr_oh;
    int ithr_reduction = 1;
    while (ithr_reduction < thr_work) {
        const int mb_ithr = (ithr_reduction - 1) % jcp.nthr_mb;
        const int oh_ithr = ((ithr_reduction - 1) / jcp.nthr_mb) % jcp.nthr_oh;
        const size_t ithr_offset
                = static_cast<size_t>(mb_ithr * jcp.nthr_oh + oh_ithr);
        const size_t wei_size = static_cast<size_t>(
                utils::rnd_up(jcp.ngroups, jcp.ch_block) * jcp.kh * jcp.kw);
        const size_t reduction_offset = ithr_offset * wei_size;

        const int ch_block = jcp.ch_block;
        const size_t bias_size = jcp.ngroups;
        size_t b_accum_offset = ithr_offset * bias_size;

        const bool compute_bias = jcp.with_bias;
        const int bias_ch_tail = jcp.ch_tail;
        const int nb_ch = bias_ch_tail > 0 ? jcp.nb_ch - 1 : jcp.nb_ch;
        for (int g = 0; g < nb_ch; ++g) {
            if (compute_bias) {
                PRAGMA_OMP_SIMD()
                for (int g_block = 0; g_block < ch_block; ++g_block) {
                    const size_t bias_offset
                            = static_cast<size_t>(g * ch_block + g_block);
                    diff_bias[bias_offset]
                            += diff_bia_reduction_buffer[b_accum_offset
                                    + bias_offset];
                }
            }
            for_(int kh = 0; kh < jcp.kh; ++kh)
            for (int kw = 0; kw < jcp.kw; ++kw) {
                const size_t wei_sp_offset
                        = static_cast<size_t>((g * jcp.kh + kh) * jcp.kw + kw);
                PRAGMA_OMP_SIMD()
                for (int g_block = 0; g_block < ch_block; ++g_block) {
                    const size_t wei_offset = static_cast<size_t>(
                            wei_sp_offset * ch_block + g_block);
                    diff_weights[wei_offset]
                            += diff_wei_reduction_buffer[reduction_offset
                                    + wei_offset];
                }
            }
        }
        // handle reduction for channel tail
        if (compute_bias) {
            for (int g = 0; g < bias_ch_tail; ++g) {
                const size_t bias_offset
                        = static_cast<size_t>(nb_ch * ch_block + g);
                diff_bias[bias_offset]
                        += diff_bia_reduction_buffer[b_accum_offset
                                + bias_offset];
            }
        }
        if (bias_ch_tail > 0) {
            for_(int kh = 0; kh < jcp.kh; ++kh)
            for (int kw = 0; kw < jcp.kw; ++kw) {
                const size_t wei_sp_offset = static_cast<size_t>(
                        (nb_ch * jcp.kh + kh) * jcp.kw + kw);
                for (int g = 0; g < bias_ch_tail; ++g) {
                    const size_t wei_offset
                            = static_cast<size_t>(wei_sp_offset * ch_block + g);
                    diff_weights[wei_offset]
                            += diff_wei_reduction_buffer[reduction_offset
                                    + wei_offset];
                }
            }
        }

        ithr_reduction++;
    }
}

template struct jit_uni_dw_convolution_bwd_weights_t<avx512_core,
        data_type::bf16>;
template struct jit_uni_dw_convolution_bwd_weights_t<avx512_core,
        data_type::bf16, data_type::f32>;
template struct jit_uni_dw_convolution_bwd_weights_t<avx512_common,
        data_type::f32>;
template struct jit_uni_dw_convolution_bwd_weights_t<avx2, data_type::f32>;
template struct jit_uni_dw_convolution_bwd_weights_t<sse41, data_type::f32>;
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
