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

template <cpu_isa_t isa>
_jit_uni_dw_convolution_bwd_weights_t<isa>::
        _jit_uni_dw_convolution_bwd_weights_t(const pd_t *pd,
                const input_vector &inputs, const output_vector &outputs)
    : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd) {

    const auto &jcp = conf_.jcp_;

    kernel_ = new jit_uni_dw_conv_bwd_weights_kernel_f32<isa>(jcp);

    const int max_threads
            = (mkldnn_in_parallel()) ? 1 : mkldnn_get_max_threads();
    nthr_ = max_threads;

    nthr_g_ = nthr_mb_ = 1;

    /* Basic-Heuristics for parallel strategy:
     * 1) Tries to parallel on the number of Groups (g) where tasks are
     * independent. Otherwise,
     * 2) Tries to split the work across g and MiniBatch (mb).
     * Parallelizing on mb requires computing a reduction for weights.
     *
     * NOTE: because of 'task partitioning' scheme, there will be unbalanced
     * per-thread load when the number of threads is high (e.g. > 16).
     */
    nthr_g_ = nstl::min(jcp.nb_ch, nthr_);
    nthr_mb_ = nstl::min(nstl::max(1, nthr_ / nthr_g_), jcp.mb);

    nthr_ = nthr_g_ * nthr_mb_;

    /* Notes: if splitting thread work on 'mb', then a reduction has to take
     * place. Hence, allocate a per-thread, local weights-buffer for the
     * reduction */
    if (nthr_mb_ > 1) {
        const size_t wei_size = jcp.ngroups * jcp.kh * jcp.kw;
        ws_reduction_ = (data_t *)malloc(
                (nthr_mb_ - 1) * wei_size * sizeof(data_t), 64);

        if (jcp.with_bias) {
            const size_t bias_size = jcp.ngroups;
            bias_reduction_ = (data_t *)malloc(
                    (nthr_mb_ - 1) * bias_size * sizeof(data_t), 64);
        }

        /* Used when executing a parallel reduction */
        if(do_parallel_reduction()){
            acc_ker_ = new cpu_accumulator_1d_t<data_type::f32>();
            simple_barrier::ctx_init(&reduction_bctx_);
        }
    }
}
template <cpu_isa_t isa>
void _jit_uni_dw_convolution_bwd_weights_t<isa>::execute_backward_weights() {

    auto src
            = (data_t *)reinterpret_cast<const data_t *>(this->input_memory(0));
    auto diff_dst
            = (data_t *)reinterpret_cast<const data_t *>(this->input_memory(1));
    const auto &jcp = kernel_->jcp;

    /* JIT-code skips the unnecessary computations within the padded region. */
    const int SKIP_TOP_PADDING = 0;

    const size_t wei_size = jcp.ngroups * jcp.kh * jcp.kw;
    const size_t bias_size = jcp.with_bias ? jcp.ngroups : 0;

    const int oh_blk_size = jcp.oh_blk_size;

    //const int simd_w = jcp.ch_block;
    const int ch_block = jcp.ch_block;

    auto set_kernel_params = [&](jit_dw_conv_call_s *conv_params,
            const int batch, const int group, const int oh_block,
            const unsigned char table_idx, const int negative_padding_offset,
            const unsigned char exec_flag) {

        const int ih_block = oh_block * jcp.stride_h;

        conv_params->table_idx = table_idx;
        conv_params->exec_flag = exec_flag;

        size_t diff_dst_off
                = ((batch * (jcp.ngroups / ch_block) + group) * jcp.oh + oh_block)
                * jcp.ow;

        size_t src_off = ((batch * (jcp.ngroups / ch_block) + group) * jcp.ih
                              + ih_block - negative_padding_offset)
                * jcp.iw;

        conv_params->output = &diff_dst[diff_dst_off * ch_block];
        conv_params->input = &src[src_off * ch_block];
    };

    parallel(nthr_, [&](const int ithr, const int nthr_) {
        auto conv_params = jit_dw_conv_call_s();

        /* assign iteration space to thread */
        const int ithr_g = ithr % nthr_g_;
        const int ithr_mb = (ithr / nthr_g_) % nthr_mb_;

        /* split dimensions */
        int g_start{ 0 }, g_end{ 0 };
        balance211(jcp.nb_ch, nthr_g_, ithr_g, g_start, g_end);

        int mb_start{ 0 }, mb_end{ 0 };
        balance211(jcp.mb, nthr_mb_, ithr_mb, mb_start, mb_end);

        auto diff_wei = ithr_mb == 0 ?
                (data_t *)reinterpret_cast<data_t *>(this->memory(0)) :
                (data_t *)ws_reduction_ + (ithr_mb - 1) * wei_size;

        auto diff_bias = ithr_mb == 0 ?
                (data_t *)reinterpret_cast<const data_t *>(this->memory(1)) :
                (data_t *)bias_reduction_ + (ithr_mb - 1) * bias_size;

        for (int g = g_start; g < g_end; ++g) {

            /* This flag controls whether the kernel loads weights from memory
             * or initializes the 'weight accummulator' registers to '0'. The
             * latter happens at the beginning of each group/16 computation. */
            unsigned char zero_filter_flag = ~FLAG_ZERO_FILTER;
            unsigned char zero_bias_flag = jcp.with_bias ? ~FLAG_ZERO_BIAS : 0;

            size_t diff_wei_off = g * jcp.kh * jcp.kw;
            conv_params.filter = &diff_wei[diff_wei_off * ch_block];

            if (jcp.with_bias)
                conv_params.bias = &diff_bias[g * ch_block];

            for (int mb = mb_start; mb < mb_end; ++mb) {

                /* The 'table index' parameter controls the table entry for the
                 * inner kernel execution. For more details see
                 * jit_uni_dw_conv_kernel_f32. */
                int table_idx = 0;

                /* OH_BLOCK is unrolled to separate the computations according
                 * to numerous condition-setting 'h' parameter. */
                int oh_blk = 0;

                /* Top-padding case - this case always executes. */
                set_kernel_params(&conv_params, mb, g, oh_blk, table_idx,
                        SKIP_TOP_PADDING, zero_filter_flag & zero_bias_flag);
                kernel_->jit_ker(&conv_params);

                zero_bias_flag |= FLAG_ZERO_BIAS;
                zero_filter_flag |= FLAG_ZERO_FILTER;
                oh_blk += oh_blk_size;

                /* Middle OH_BLOCK cases. */
                for (; oh_blk < (jcp.oh - oh_blk_size); oh_blk += oh_blk_size) {
                    table_idx = 1;
                    set_kernel_params(&conv_params, mb, g, oh_blk, table_idx,
                            jcp.t_pad, zero_filter_flag & zero_bias_flag);
                    kernel_->jit_ker(&conv_params);
                }
                table_idx++;

                /* Bottom block */
                if (oh_blk < jcp.oh) {
                    set_kernel_params(&conv_params, mb, g, oh_blk, table_idx,
                            jcp.t_pad, zero_filter_flag & zero_bias_flag);
                    kernel_->jit_ker(&conv_params);
                }
            }
        }
        if (do_parallel_reduction() && nthr_mb_ > 1) {

            size_t reduct_start{ 0 }, reduct_end{ 0 };
            balance211(wei_size, nthr_, ithr, reduct_start, reduct_end);

            const size_t reduct_off = reduct_start;

            auto *acc_data
                    = (data_t *)reinterpret_cast<data_t *>(this->memory(0))
                    + reduct_off;

            const int acc_size = reduct_end - reduct_start;

            simple_barrier::barrier(&reduction_bctx_, nthr_);

            for (int thr_mb = 1; thr_mb < nthr_mb_; ++thr_mb) {

                auto *src_data = (data_t *)ws_reduction_
                        + (thr_mb - 1) * wei_size + reduct_off;

                acc_ker_->accumulate(acc_data, src_data, acc_size);
            }
        }
    });

    /* Apply single-threaded 'mb' reduction */
    if (nthr_mb_ > 1) {

        auto diff_weights
                = (data_t *)reinterpret_cast<data_t *>(this->memory(0));
        auto diff_bias
                = (data_t *)reinterpret_cast<const data_t *>(this->memory(1));

        for (int thr_mb = 1; thr_mb < nthr_mb_; ++thr_mb) {

            size_t mb_accum_offset = (thr_mb - 1) * wei_size;
            size_t b_accum_offset = (thr_mb - 1) * bias_size;

            for (int g = 0; g < jcp.nb_ch; ++g) {

                /* Reduction on Bias */
                if (jcp.with_bias) {
                    PRAGMA_OMP_SIMD()
                    for (int g_block = 0; g_block < ch_block; ++g_block) {
                        size_t bias_offset = g * ch_block + g_block;
                        diff_bias[bias_offset] += bias_reduction_[b_accum_offset
                                + bias_offset];
                    }
                }
                if (!do_parallel_reduction()) {
                    for (int kh = 0; kh < jcp.kh; ++kh) {
                        for (int kw = 0; kw < jcp.kw; ++kw) {

                            size_t wei_offset = (g * jcp.kh + kh) * jcp.kw + kw;
                            PRAGMA_OMP_SIMD()
                            for (int g_block = 0; g_block < ch_block; ++g_block) {
                                diff_weights[wei_offset * ch_block + g_block]
                                        += ws_reduction_[mb_accum_offset
                                                + wei_offset * ch_block
                                                + g_block];
                            }
                        }
                    }
                }
            }
        }
    }
}

template _jit_uni_dw_convolution_bwd_weights_t<avx512_common>::
        _jit_uni_dw_convolution_bwd_weights_t(const pd_t *pd,
                const input_vector &inputs, const output_vector &outputs);
template _jit_uni_dw_convolution_bwd_weights_t<avx2>::
        _jit_uni_dw_convolution_bwd_weights_t(const pd_t *pd,
                const input_vector &inputs, const output_vector &outputs);
template _jit_uni_dw_convolution_bwd_weights_t<sse42>::
        _jit_uni_dw_convolution_bwd_weights_t(const pd_t *pd,
                const input_vector &inputs, const output_vector &outputs);

template void _jit_uni_dw_convolution_bwd_weights_t<avx512_common>::
        execute_backward_weights();
template void _jit_uni_dw_convolution_bwd_weights_t<avx2>::
        execute_backward_weights();
template void _jit_uni_dw_convolution_bwd_weights_t<sse42>::
        execute_backward_weights();

}
}
}
