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

#include "c_types_map.hpp"
#include "memory_tracking.hpp"
#include "mkldnn_thread.hpp"

#include "bfloat16_utils.hpp"

#include "jit_uni_dw_convolution.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::memory_tracking::names;
using namespace mkldnn::impl::utils;

template <cpu_isa_t isa, data_type_t src_type, data_type_t dst_type>
void _jit_uni_dw_convolution_fwd_t<isa, src_type, dst_type>::execute_forward()
        const {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto dst = reinterpret_cast<dst_data_t *>(this->memory());

    const memory_desc_wrapper src_d(pd()->src_pd());
    const memory_desc_wrapper dst_d(pd()->dst_pd());
    const memory_desc_wrapper weights_d(pd()->weights_pd(0));
    const memory_desc_wrapper bias_d(pd()->weights_pd(1));

    const auto &jcp = pd()->jcp_;

    f32_data_t *bias = nullptr;
    if (pd()->desc()->bias_desc.data_type == data_type::bf16) {
        auto bias_in = reinterpret_cast<const bf16_data_t *>(this->input_memory(2));
        bias = scratchpad().template get<float>(key_conv_bias_bf16_convert_wsp);
        bf16_cvt_utils::cvt_bfloat16_to_float(bias, bias_in, jcp.oc_without_padding);
        utils::array_set(bias + jcp.oc_without_padding, 0.f,
                jcp.oc - jcp.oc_without_padding);
    } else {
        auto bias_in = reinterpret_cast<const f32_data_t *>(this->input_memory(2));
        if (pd()->wants_padded_bias()) {
            auto padded_bias = this->scratchpad().template get<f32_data_t>(
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

    auto kernel_params = [&](int ur_w_step, int ow, int oh, int od, int ih, int id, int kh, int kd,
            int kh_padding, int kd_padding, int ch, int ch_num, int n) {
        auto par_conv = jit_conv_call_s();

        const int i_l_overflow = nstl::max(0, (jcp.l_pad - ow * str_w));
        const int i_r_overflow = nstl::max(jcp.iw, (ow * str_w
            + (jcp.kw - 1)*dil_w - jcp.l_pad + 1)) - jcp.iw;

        const int iw = nstl::max((ow*str_w - jcp.l_pad
            + div_up(i_l_overflow, dil_w)*dil_w), 0);
        const int kw = div_up(i_l_overflow, dil_w);

        const int kw_padding = jcp.kw - div_up(i_l_overflow, dil_w)
            - div_up(i_r_overflow, dil_w);

        size_t src_off = (jcp.ndims == 5) ? src_d.blk_off(n, ch, id, ih, iw) : src_d.blk_off(n, ch, ih, iw);
        size_t dst_off = (jcp.ndims == 5) ? dst_d.blk_off(n, ch, od, oh, ow) : dst_d.blk_off(n, ch, oh, ow);
        size_t wei_off = (jcp.ndims == 5) ? weights_d.blk_off(ch, 0, 0, kd, kh, kw) : weights_d.blk_off(ch, 0, 0, kh, kw);

        par_conv.src = &src[src_off];
        par_conv.dst = &dst[dst_off];
        par_conv.filt = &weights[wei_off];

        if (bias) par_conv.bias = &bias[bias_d.blk_off(ch*jcp.ch_block)];

        par_conv.kd_padding = (size_t)nstl::max(0, kd_padding);
        par_conv.kh_padding = (size_t)nstl::max(0, kh_padding);
        par_conv.kw_padding = (size_t)nstl::max(0, kw_padding);

        par_conv.ur_w = (size_t)ur_w_step;

        par_conv.ch_blocks = nstl::min(ch + ch_num, jcp.nb_ch) - ch;
        par_conv.oc_off = ch * jcp.ch_block * sizeof(float);

        return par_conv;
    };

    int MB = pd()->MB();
    const int chb_work = utils::div_up(jcp.nb_ch, jcp.nb_ch_blocking);
    parallel_nd(MB, chb_work, jcp.od, jcp.oh,
            [&](int n, int chb, int od, int oh) {
        int ch = chb * jcp.nb_ch_blocking;
        int ch_num = jcp.nb_ch_blocking;

        const int i_front_overflow = nstl::max(0, (int)(jcp.f_pad - od*str_d));
        const int i_back_overflow = nstl::max(jcp.id,
            (int)(od*str_d + (jcp.kd - 1)*dil_d - jcp.f_pad + 1)) - jcp.id;

        const int i_t_overflow = nstl::max(0, (int)(jcp.t_pad - oh*str_h));
        const int i_b_overflow = nstl::max(jcp.ih,
            (int)(oh*str_h + (jcp.kh - 1)*dil_h - jcp.t_pad + 1)) - jcp.ih;

        const int id = nstl::max((int)(od*str_d - jcp.f_pad
            + div_up(i_front_overflow, dil_d)*dil_d), 0);
        const int kd = div_up(i_front_overflow, dil_d);
        const int kd_padding = jcp.kd - div_up(i_front_overflow, dil_d)
            - div_up(i_back_overflow, dil_d);

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
            jit_conv_call_s par_conv = kernel_params(ur_w_step, ow, oh, od, ih, id,
                                        kh, kd, kh_padding, kd_padding, ch, ch_num, n);

            kernel_->jit_ker(&par_conv);
        }

        // main loop
        ur_w_step = (jcp.iw - (jcp.kw - 1)*dil_w + jcp.l_pad - 1)
            / jcp.stride_w - ow + 1;
        if (ur_w_step > 0) {
            jit_conv_call_s par_conv = kernel_params(ur_w_step, ow, oh, od, ih, id,
                                        kh, kd, kh_padding, kd_padding, ch, ch_num, n);

            kernel_->jit_ker(&par_conv);

            ow += ur_w_step;
        }

        // right border
        ur_w_step = 1;
        for (; ow < jcp.ow; ow++) {
            jit_conv_call_s par_conv = kernel_params(ur_w_step, ow, oh, od, ih, id,
                                        kh, kd, kh_padding, kd_padding, ch, ch_num, n);

            kernel_->jit_ker(&par_conv);
        }
    });

    if (pd()->wants_zero_pad_dst())
        output_memory_primitive(0)->zero_pad();
}

template struct _jit_uni_dw_convolution_fwd_t<avx512_core, data_type::bf16,
        data_type::f32>;
template struct _jit_uni_dw_convolution_fwd_t<avx512_core, data_type::bf16>;
template struct _jit_uni_dw_convolution_fwd_t<avx512_common, data_type::f32>;
template struct _jit_uni_dw_convolution_fwd_t<avx2, data_type::f32>;
template struct _jit_uni_dw_convolution_fwd_t<sse42, data_type::f32>;

template <cpu_isa_t isa, data_type_t diff_dst_type, data_type_t diff_src_type>
void _jit_uni_dw_convolution_bwd_data_t<isa, diff_dst_type,
        diff_src_type>::execute_backward_data() const {
    auto diff_dst
            = reinterpret_cast<const diff_dst_data_t *>(this->input_memory(0));
    auto weights
            = reinterpret_cast<const diff_wei_data_t *>(this->input_memory(1));
    auto diff_src = reinterpret_cast<diff_src_data_t *>(this->memory());

    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_pd());
    const memory_desc_wrapper diff_src_d(pd()->diff_src_pd());
    const memory_desc_wrapper weights_d(pd()->weights_pd(0));

    const auto &jcp = pd()->jcp_;

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

        return par_conv;
    };

    const int aux_w
            = nstl::min(jcp.iw, jcp.iw - jcp.kw + jcp.r_pad + jcp.stride_w);
    int MB = pd()->MB();
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
            ur_str_w = (aux_w - iw) / jcp.stride_w;
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

template struct _jit_uni_dw_convolution_bwd_data_t<avx512_core, data_type::bf16,
        data_type::f32>;
template struct _jit_uni_dw_convolution_bwd_data_t<avx512_core,
        data_type::bf16>;
template struct _jit_uni_dw_convolution_bwd_data_t<avx512_common,
        data_type::f32>;
template struct _jit_uni_dw_convolution_bwd_data_t<avx2, data_type::f32>;
template struct _jit_uni_dw_convolution_bwd_data_t<sse42, data_type::f32>;

template <cpu_isa_t isa, data_type_t src_type, data_type_t diff_weights_type>
void _jit_uni_dw_convolution_bwd_weights_t<isa, src_type,
        diff_weights_type>::execute_backward_weights() const {
    auto src = reinterpret_cast<const src_data_t *>(this->input_memory(0));
    auto diff_dst
            = reinterpret_cast<const diff_dst_data_t *>(this->input_memory(1));
    auto diff_weights
            = reinterpret_cast<diff_weights_data_t *>(this->memory(0));

    auto diff_wei_reduction_buf
            = scratchpad().template get<f32_data_t>(key_conv_wei_reduction);
    auto diff_bia_reduction_buf
            = scratchpad().template get<f32_data_t>(key_conv_bia_reduction);

    const auto &jcp = pd()->jcp_;

    float *diff_bias = nullptr;
    if (jcp.bia_dt == data_type::bf16) {
        diff_bias = scratchpad().template get<float>(key_conv_bias_bf16_convert_wsp);
    } else {
        diff_bias = reinterpret_cast<f32_data_t *>(this->memory(1));
    }

    const size_t wei_size = jcp.ngroups * jcp.kh * jcp.kw;
    const size_t bias_size = jcp.with_bias ? jcp.ngroups : 0;

    const int ch_block = jcp.ch_block;

    auto set_kernel_params = [&](jit_dw_conv_call_s *conv_params,
            const int batch, const int group, const int oh_start,
            const int work_size, const unsigned char exec_flag,
            const size_t kh_padding, const size_t filter_off) {
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

        size_t src_off = ((batch * (jcp.ngroups / ch_block) + group) * jcp.ih
                                 + ih_s - tpad_underflow_off)
                * jcp.iw;

        conv_params->output = &diff_dst[diff_dst_off * ch_block];
        conv_params->input = &src[src_off * ch_block];
    };

    parallel(jcp.nthr, (size_t)mkldnn_get_max_threads(), [&](const int ithr, const int nthr) {
        assert(nthr == jcp.nthr);

        auto conv_params = jit_dw_conv_call_s();
        const int h_block_size = 15;

        /* assign iteration space to thread */
        const int ithr_g = ithr % jcp.nthr_g;
        const int ithr_mb = (ithr / jcp.nthr_g) % jcp.nthr_mb;

        /* split dimensions */
        int g_start{ 0 }, g_end{ 0 };
        balance211(jcp.nb_ch, jcp.nthr_g, ithr_g, g_start, g_end);

        int mb_start{ 0 }, mb_end{ 0 };
        balance211(jcp.mb, jcp.nthr_mb, ithr_mb, mb_start, mb_end);

        auto i_mb
                = diff_weights_type == data_type::bf16 ? ithr_mb : ithr_mb - 1;
        f32_data_t *diff_wei
                = (ithr_mb == 0 && diff_weights_type == data_type::f32)
                ? (f32_data_t *)diff_weights
                : diff_wei_reduction_buf + i_mb * wei_size;

        auto diff_bia = ithr_mb == 0 ? diff_bias : diff_bia_reduction_buf
                        + (ithr_mb - 1) * bias_size;

        for (int g = g_start; g < g_end; ++g) {
            unsigned char zero_filter_flag = FLAG_ZERO_FILTER;
            unsigned char zero_bias_flag = jcp.with_bias ? FLAG_ZERO_BIAS : 0;

            size_t diff_wei_off = g * jcp.kh * jcp.kw;
            conv_params.filter = &diff_wei[diff_wei_off * ch_block];

            if (jcp.with_bias)
                conv_params.bias = &diff_bia[g * ch_block];

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
                            zero_filter_flag | zero_bias_flag,
                            kh_t_padding + kh_b_padding, kh_t_padding);
                    kernel_->jit_ker(&conv_params);

                    zero_bias_flag &= ~FLAG_ZERO_BIAS;
                    zero_filter_flag &= ~FLAG_ZERO_FILTER;
                    oh += h_work;
                }
            }
        }
    });

    if (jcp.bia_dt == data_type::bf16) {
        auto diff_bias_in = reinterpret_cast<mkldnn_bfloat16_t *>(this->memory(1));
        bf16_cvt_utils::cvt_float_to_bfloat16(diff_bias_in, diff_bias, jcp.ngroups);
    }
}

/* TODO: Performing a Parallel Reduction could potentially improve performance;
 * this should be explored in the future if further optimizations are required.
 */
template <>
void _jit_uni_dw_convolution_bwd_weights_t<avx512_core,
        data_type::bf16>::execute_reduction() const {

    auto diff_wei_reduction_buf
            = scratchpad().template get<f32_data_t>(key_conv_wei_reduction);
    auto diff_bia_reduction_buf
            = scratchpad().template get<f32_data_t>(key_conv_bia_reduction);
    auto diff_weights
            = reinterpret_cast<diff_weights_data_t *>(this->memory(0));

    const auto &jcp = pd()->jcp_;
    assert(jcp.dwei_dt == data_type::bf16);

    const size_t wei_size = jcp.ngroups * jcp.kh * jcp.kw;
    const size_t bias_size = jcp.with_bias ? jcp.ngroups : 0;

    const int ch_block = jcp.ch_block;

    float *diff_bias = nullptr;
    if (jcp.bia_dt == data_type::bf16) {
        diff_bias = scratchpad().template get<float>(key_conv_bias_bf16_convert_wsp);
    } else {
        diff_bias = reinterpret_cast<f32_data_t *>(this->memory(1));
    }

    /* Apply single-threaded 'mb' reduction */
    if (jcp.with_bias && jcp.nthr_mb > 1) {
        for (int thr_mb = 1; thr_mb < jcp.nthr_mb; ++thr_mb) {
            size_t b_accum_offset = (thr_mb - 1) * bias_size;

            for (int g = 0; g < jcp.nb_ch; ++g) {
                /* Reduction on Bias */
                PRAGMA_OMP_SIMD()
                for (int g_block = 0; g_block < ch_block; ++g_block) {
                    size_t bias_offset = g * ch_block + g_block;
                    diff_bias[bias_offset]
                            += diff_bia_reduction_buf[b_accum_offset
                                    + bias_offset];
                }
            }
        }
    }
    if (jcp.bia_dt == data_type::bf16) {
        auto diff_bias_in = reinterpret_cast<bf16_data_t *>(this->memory(1));
        bf16_cvt_utils::cvt_float_to_bfloat16(diff_bias_in, diff_bias, jcp.ngroups);
    }
    /* Apply single-threaded 'mb' reduction */
    if (jcp.nthr_mb > 1) {
        for (int thr_mb = 2; thr_mb < jcp.nthr_mb; ++thr_mb) {
            size_t mb_accum_offset = thr_mb * wei_size;
            acc_ker_->accumulate(&diff_wei_reduction_buf[0],
                    &diff_wei_reduction_buf[mb_accum_offset], wei_size);
        }
        bf16_cvt_utils::add_floats_and_cvt_to_bfloat16(
                (mkldnn_bfloat16_t *)&(diff_weights[0]),
                (float *)&diff_wei_reduction_buf[0],
                (float *)&diff_wei_reduction_buf[wei_size], wei_size);
    } else {
        bf16_cvt_utils::cvt_float_to_bfloat16(
                (mkldnn_bfloat16_t *)&(diff_weights[0]),
                (const float *)&(diff_wei_reduction_buf[0]), wei_size);
    }
}

template <>
void _jit_uni_dw_convolution_bwd_weights_t<sse42,
        data_type::f32>::execute_reduction() const {

    auto diff_bias = reinterpret_cast<f32_data_t *>(this->memory(1));
    auto diff_wei_reduction_buf
            = scratchpad().template get<f32_data_t>(key_conv_wei_reduction);
    auto diff_bia_reduction_buf
            = scratchpad().template get<f32_data_t>(key_conv_bia_reduction);
    auto diff_weights = reinterpret_cast<f32_data_t *>(this->memory(0));

    const auto &jcp = pd()->jcp_;

    const size_t wei_size = jcp.ngroups * jcp.kh * jcp.kw;
    const size_t bias_size = jcp.with_bias ? jcp.ngroups : 0;

    const int ch_block = jcp.ch_block;

    /* Apply single-threaded 'mb' reduction */
    for (int thr_mb = 1; thr_mb < jcp.nthr_mb; ++thr_mb) {
        size_t mb_accum_offset = (thr_mb - 1) * wei_size;
        size_t b_accum_offset = (thr_mb - 1) * bias_size;

        for (int g = 0; g < jcp.nb_ch; ++g) {
            /* Reduction on Bias */
            if (jcp.with_bias) {
                PRAGMA_OMP_SIMD()
                for (int g_block = 0; g_block < ch_block; ++g_block) {
                    size_t bias_offset = g * ch_block + g_block;
                    diff_bias[bias_offset]
                            += diff_bia_reduction_buf[b_accum_offset
                                    + bias_offset];
                }
            }
            for (int kh = 0; kh < jcp.kh; ++kh)
            for (int kw = 0; kw < jcp.kw; ++kw) {
                size_t wei_offset = (g * jcp.kh + kh) * jcp.kw + kw;
                PRAGMA_OMP_SIMD()
                for (int g_block = 0; g_block < ch_block; ++g_block) {
                    const size_t off = wei_offset * ch_block + g_block;
                    diff_weights[off]
                            += diff_wei_reduction_buf[mb_accum_offset
                                    + off];
                }
            }
        }
    }
}

template <cpu_isa_t isa, data_type_t src_type, data_type_t diff_weights_type>
void _jit_uni_dw_convolution_bwd_weights_t<isa, src_type,
        diff_weights_type>::execute_reduction() const {

    auto diff_wei_reduction_buf
            = scratchpad().template get<f32_data_t>(key_conv_wei_reduction);
    auto diff_bia_reduction_buf
            = scratchpad().template get<f32_data_t>(key_conv_bia_reduction);
    auto diff_weights = reinterpret_cast<f32_data_t *>(this->memory(0));

    const auto &jcp = pd()->jcp_;

    const size_t wei_size = jcp.ngroups * jcp.kh * jcp.kw;
    const size_t bias_size = jcp.with_bias ? jcp.ngroups : 0;

    const int ch_block = jcp.ch_block;

    assert(diff_weights_type == data_type::f32
            && jcp.dwei_dt == data_type::f32);

    float *diff_bias = nullptr;
    if (jcp.bia_dt == data_type::bf16) {
        diff_bias = scratchpad().template get<float>(key_conv_bias_bf16_convert_wsp);
    } else {
        diff_bias = reinterpret_cast<f32_data_t *>(this->memory(1));
    }

    /* Apply single-threaded 'mb' reduction */
    for (int thr_mb = 1; thr_mb < jcp.nthr_mb; ++thr_mb) {
        size_t mb_accum_offset = (thr_mb - 1) * wei_size;
        size_t b_accum_offset = (thr_mb - 1) * bias_size;

        for (int g = 0; g < jcp.nb_ch; ++g) {
            /* Reduction on Bias */
            if (jcp.with_bias) {
                PRAGMA_OMP_SIMD()
                for (int g_block = 0; g_block < ch_block; ++g_block) {
                    size_t bias_offset = g * ch_block + g_block;
                    diff_bias[bias_offset]
                            += diff_bia_reduction_buf[b_accum_offset
                                    + bias_offset];
                }
            }
        }
        acc_ker_->accumulate(&diff_weights[0],
                &diff_wei_reduction_buf[mb_accum_offset], wei_size);
    }

    if (jcp.bia_dt == data_type::bf16) {
        auto diff_bias_in = reinterpret_cast<bf16_data_t *>(this->memory(1));
        bf16_cvt_utils::cvt_float_to_bfloat16(diff_bias_in, diff_bias, jcp.ngroups);
    }

}

template struct _jit_uni_dw_convolution_bwd_weights_t<avx512_core,
        data_type::bf16>;
template struct _jit_uni_dw_convolution_bwd_weights_t<avx512_core,
        data_type::bf16, data_type::f32>;
template struct _jit_uni_dw_convolution_bwd_weights_t<avx512_common,
        data_type::f32>;
template struct _jit_uni_dw_convolution_bwd_weights_t<avx2, data_type::f32>;
template struct _jit_uni_dw_convolution_bwd_weights_t<sse42, data_type::f32>;
}
}
}
