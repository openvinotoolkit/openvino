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

#include "mkldnn_types.h"
#include "c_types_map.hpp"
#include "mkldnn_thread.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "jit_avx512_core_x8s8s32x_convolution.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;

using namespace nstl;

using jit_conv_ker_t = void (*)(jit_conv_call_s *);

#define wht_blk_off(d, g, ...) \
        (conf_.with_groups() \
         ? (d).blk_off((g), __VA_ARGS__) \
         : (d).blk_off(__VA_ARGS__))

template <bool with_relu, data_type_t src_type, data_type_t dst_type>
void _jit_avx512_core_x8s8s32x_convolution_fwd_t<with_relu, src_type, dst_type>::
execute_forward()
{
    auto src = reinterpret_cast<const src_data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const wei_data_t *>(this->input_memory(1));
    auto bias = reinterpret_cast<const char *>(this->input_memory(2));
    auto dst = reinterpret_cast<dst_data_t *>(this->memory());

    const memory_desc_wrapper src_d(conf_.src_pd());
    const memory_desc_wrapper dst_d(conf_.dst_pd());
    const memory_desc_wrapper weights_d(conf_.weights_pd(0));
    const memory_desc_wrapper bias_d(conf_.weights_pd(1));

    const size_t bia_dt_size = conf_.with_bias()
        ? types::data_type_size(conf_.cdesc()->bias_desc.data_type) : 0;

    const auto &jcp = kernel_->jcp;
    assert(jcp.nb_oc % jcp.nb_oc_blocking == 0);

    size_t offset = (size_t)jcp.ngroups * jcp.oc * jcp.ic * jcp.kh * jcp.kw;
    auto w = const_cast<wei_data_t *>(weights);
    int32_t* compensation = (jcp.signed_input)
                                ? reinterpret_cast<int32_t *>(&w[offset]) : 0;
    const auto &oscales = conf_.attr()->output_scales_;
    int oc_chunks = jcp.nb_oc / jcp.nb_oc_blocking;
    int nb_groups = jcp.nb_ch;
    int group_block = jcp.ch_block;
    int work_amount = jcp.mb * nb_groups * oc_chunks * jcp.oh * jcp.nb_ow;

    parallel(0, [&](const int ithr, const int nthr) {

        int start{0}, end{0};
        balance211(work_amount, nthr, ithr, start, end);

        auto p = jit_conv_call_s();

        size_t src_h_stride = src_d.blk_off(0, 0, 1);
        size_t dst_h_stride = dst_d.blk_off(0, 0, 1);
        size_t wht_h_stride = wht_blk_off(weights_d, 0, 0, 0, 1);

        int n{ 0 }, gb{ 0 }, occ{ 0 }, oh_s{ 0 }, owb{ 0 };
        if (jcp.loop_order == loop_cwgn)
            nd_iterator_init(start, occ, oc_chunks, owb, jcp.nb_ow, gb,
                    nb_groups, n, jcp.mb, oh_s, jcp.oh);
        else if (jcp.loop_order == loop_gncw)
            nd_iterator_init(start, gb, nb_groups, n, jcp.mb, occ, oc_chunks,
                    owb, jcp.nb_ow, oh_s, jcp.oh);
        else if (jcp.loop_order == loop_ngcw)
            nd_iterator_init(start, n, jcp.mb, gb, nb_groups, occ, oc_chunks,
                    owb, jcp.nb_ow, oh_s, jcp.oh);
        else
            assert(!"unsupported loop order");
        while (start < end) {
            int ocb = occ * jcp.nb_oc_blocking;
            int g = gb * group_block;
            int g_oc = (g * jcp.nb_oc + ocb) * jcp.oc_block;

            int g_ic = g * jcp.nb_ic * jcp.ic_block;

            int work_rem = end - start;
            int ih_s = -jcp.t_pad + oh_s * jcp.stride_h;
            int oh_e = oh_s + work_rem > jcp.oh ? jcp.oh : oh_s + work_rem;
            int ow_s = owb * jcp.ow_block;
            int iw_s = ow_s * jcp.stride_w;

            auto bias_w = bias
                ? bias + (bias_d.blk_off(g_oc) * bia_dt_size)
                : 0;
            int32_t *compensation_w = (jcp.signed_input)
                                                    ? compensation + g_oc : 0;

            auto dst_w = dst + dst_d.blk_off(n, g_oc, oh_s, ow_s);
            auto src_w = src + src_d.blk_off(n, g_ic, ih_s, iw_s);
            auto wht_w = weights + wht_blk_off(weights_d, gb, ocb, 0);

            auto scales = (jcp.signed_input && jcp.ver != ver_vnni)
                ? &local_scales_[jcp.is_oc_scale * g_oc]
                : &oscales.scales_[jcp.is_oc_scale * g_oc];

            for (int oj = oh_s, ij = ih_s; oj < oh_e;
                ++oj, ij += jcp.stride_h) {
                int dilate_h = jcp.dilate_h + 1;
                int i_t_overflow = nstl::min(jcp.kh,
                                                div_up(max(0, -ij), dilate_h));
                int i_b_overflow = nstl::min(jcp.kh, div_up(
                        max(0, ij - jcp.ih + (jcp.kh - 1) * dilate_h + 1),
                        dilate_h));
                int kh_padding = nstl::max(0,
                    jcp.kh - i_t_overflow - i_b_overflow);

                size_t wei_stride = (!jcp.signed_input)
                                            ? i_t_overflow * wht_h_stride : 0;
                p.src = src_w + i_t_overflow * dilate_h * src_h_stride;
                p.dst = dst_w;
                p.filt = wht_w + wei_stride;
                p.bias = bias_w;
                p.compensation = compensation_w;
                p.oc_blocks = jcp.is_depthwise ? gb : ocb;
                p.kh_padding = kh_padding;
                p.scales = scales;
                p.t_overflow = i_t_overflow;
                p.b_overflow = i_b_overflow;
                p.owb = owb;

                kernel_->jit_ker(&p);

                src_w += src_h_stride * jcp.stride_h;
                dst_w += dst_h_stride;
            }
            if (jcp.loop_order == loop_cwgn)
                nd_iterator_jump(start, end, occ, oc_chunks, owb, jcp.nb_ow, gb,
                        nb_groups, n, jcp.mb, oh_s, jcp.oh);
            else if (jcp.loop_order == loop_gncw)
                nd_iterator_jump(start, end, gb, nb_groups, n, jcp.mb, occ,
                        oc_chunks, owb, jcp.nb_ow, oh_s, jcp.oh);
            else if (jcp.loop_order == loop_ngcw)
                nd_iterator_jump(start, end, n, jcp.mb, gb, nb_groups, occ,
                        oc_chunks, owb, jcp.nb_ow, oh_s, jcp.oh);
            else
                assert(!"unsupported loop order");
        }
    });
}

template struct _jit_avx512_core_x8s8s32x_convolution_fwd_t<false,
                                                data_type::s8, data_type::u8>;
template struct _jit_avx512_core_x8s8s32x_convolution_fwd_t<true,
                                                data_type::s8, data_type::u8>;
template struct _jit_avx512_core_x8s8s32x_convolution_fwd_t<false,
                                                data_type::u8, data_type::u8>;
template struct _jit_avx512_core_x8s8s32x_convolution_fwd_t<true,
                                                data_type::u8, data_type::u8>;
template struct _jit_avx512_core_x8s8s32x_convolution_fwd_t<false,
                                                data_type::s8, data_type::s8>;
template struct _jit_avx512_core_x8s8s32x_convolution_fwd_t<true,
                                                data_type::s8, data_type::s8>;
template struct _jit_avx512_core_x8s8s32x_convolution_fwd_t<false,
                                                data_type::u8, data_type::s8>;
template struct _jit_avx512_core_x8s8s32x_convolution_fwd_t<true,
                                                data_type::u8, data_type::s8>;
template struct _jit_avx512_core_x8s8s32x_convolution_fwd_t<false,
                                                data_type::s8, data_type::s32>;
template struct _jit_avx512_core_x8s8s32x_convolution_fwd_t<true,
                                                data_type::s8, data_type::s32>;
template struct _jit_avx512_core_x8s8s32x_convolution_fwd_t<false,
                                                data_type::u8, data_type::s32>;
template struct _jit_avx512_core_x8s8s32x_convolution_fwd_t<true,
                                                data_type::u8, data_type::s32>;
template struct _jit_avx512_core_x8s8s32x_convolution_fwd_t<false,
                                                data_type::s8, data_type::f32>;
template struct _jit_avx512_core_x8s8s32x_convolution_fwd_t<true,
                                                data_type::s8, data_type::f32>;
template struct _jit_avx512_core_x8s8s32x_convolution_fwd_t<false,
                                                data_type::u8, data_type::f32>;
template struct _jit_avx512_core_x8s8s32x_convolution_fwd_t<true,
                                                data_type::u8, data_type::f32>;
}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
