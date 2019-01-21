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
#include "jit_uni_x8s8s32x_1x1_convolution.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;

template <cpu_isa_t isa, bool with_relu, data_type_t src_type, data_type_t dst_type>
void _jit_uni_x8s8s32x_1x1_convolution_fwd_t<isa, with_relu, src_type, dst_type>::execute_forward() {
    auto src = reinterpret_cast<const src_data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const wei_data_t *>(this->input_memory(1));
    auto bias = reinterpret_cast<const char *>(this->input_memory(2));
    auto dst = reinterpret_cast<dst_data_t *>(this->memory());

    const memory_desc_wrapper src_d(conf_.src_pd());
    const memory_desc_wrapper weights_d(conf_.weights_pd(0));
    const memory_desc_wrapper dst_d(conf_.dst_pd());
    const memory_desc_wrapper bias_d(conf_.weights_pd(1));

    const auto &jcp = kernel_->jcp;

    int ocb_work = utils::div_up(jcp.nb_oc, jcp.nb_oc_blocking);
    int ohb_work = utils::div_up(jcp.oh, jcp.nb_oh_blocking);
    const int work_amount = jcp.mb * jcp.ngroups * ocb_work * ohb_work;

    const int stride_h = conf_.cdesc()->strides[0];
    const int stride_w = conf_.cdesc()->strides[1];
    const int pad_t = conf_.cdesc()->padding[0][0];
    const int pad_l = conf_.cdesc()->padding[0][1];

    const size_t bia_dt_size = conf_.with_bias()
        ? types::data_type_size(conf_.cdesc()->bias_desc.data_type) : 0;

    const auto &oscales = conf_.attr()->output_scales_;

    auto ker = [&](const int ithr, const int nthr) {
        jit_1x1_conv_call_s p = {};
        p.acc_s32 = ws_ + ithr * ws_per_thread_;

        const int oh_block = jcp.ow;

        int start{0}, end{0};
        balance211(work_amount, nthr, ithr, start, end);

        int n{0}, g{0}, ocb{0}, ohb{0};
        nd_iterator_init(start, n, jcp.mb, g, jcp.ngroups, ohb,
                         ohb_work, ocb, ocb_work);

        for (int iwork = start; iwork < end; ++iwork) {
            int oc_ = ocb * jcp.nb_oc_blocking;
            int oc_num = jcp.nb_oc_blocking;

            int oh_ = ohb * jcp.nb_oh_blocking;
            int oh_num = jcp.nb_oh_blocking;

            int oh_step = nstl::min(oh_ + oh_num, jcp.oh) - oh_;

            const int os = oh_ * oh_block;
            const int oh = os / jcp.ow;
            const int ow = os % jcp.ow;

            const int ih = nstl::max(oh * stride_h - pad_t, 0);
            const int iw = nstl::max(ow * stride_w - pad_l, 0);

            p.os_dim = this_block_size(os, jcp.os, oh_step * oh_block);
            p.oc_dim = nstl::min(oc_ + oc_num, jcp.nb_oc) - oc_;

            const size_t dst_off = dst_d.blk_off(n, oc_*jcp.oc_block, oh, ow);
            p.output_data = &dst[dst_off];

            if (bias)
                p.bias_data = &bias[bias_d.blk_off(oc_ * jcp.oc_block * bia_dt_size)];

            p.scales = &oscales.scales_[jcp.is_oc_scale * oc_ * jcp.oc_block];
            p.oc_data = &weights[conf_.with_groups() ? weights_d.blk_off(g, oc_, 0) : weights_d.blk_off(oc_, 0)];
            p.is_data = src + src_d.blk_off(n, 0, ih, iw);

            kernel_->jit_ker(&p);

            nd_iterator_step(n, jcp.mb, g, jcp.ngroups, ohb,
                             ohb_work, ocb, ocb_work);
        }
    };

    parallel(0, ker);
}

template void _jit_uni_x8s8s32x_1x1_convolution_fwd_t<avx2, true, data_type::u8, data_type::u8>::execute_forward();
template void _jit_uni_x8s8s32x_1x1_convolution_fwd_t<avx2, true, data_type::u8, data_type::s8>::execute_forward();
template void _jit_uni_x8s8s32x_1x1_convolution_fwd_t<avx2, true, data_type::u8, data_type::s32>::execute_forward();
template void _jit_uni_x8s8s32x_1x1_convolution_fwd_t<avx2, true, data_type::u8, data_type::f32>::execute_forward();
template void _jit_uni_x8s8s32x_1x1_convolution_fwd_t<avx2, false, data_type::u8, data_type::u8>::execute_forward();
template void _jit_uni_x8s8s32x_1x1_convolution_fwd_t<avx2, false, data_type::u8, data_type::s8>::execute_forward();
template void _jit_uni_x8s8s32x_1x1_convolution_fwd_t<avx2, false, data_type::u8, data_type::s32>::execute_forward();
template void _jit_uni_x8s8s32x_1x1_convolution_fwd_t<avx2, false, data_type::u8, data_type::f32>::execute_forward();

template void _jit_uni_x8s8s32x_1x1_convolution_fwd_t<avx2, true, data_type::s8, data_type::u8>::execute_forward();
template void _jit_uni_x8s8s32x_1x1_convolution_fwd_t<avx2, true, data_type::s8, data_type::s8>::execute_forward();
template void _jit_uni_x8s8s32x_1x1_convolution_fwd_t<avx2, true, data_type::s8, data_type::s32>::execute_forward();
template void _jit_uni_x8s8s32x_1x1_convolution_fwd_t<avx2, true, data_type::s8, data_type::f32>::execute_forward();
template void _jit_uni_x8s8s32x_1x1_convolution_fwd_t<avx2, false, data_type::s8, data_type::u8>::execute_forward();
template void _jit_uni_x8s8s32x_1x1_convolution_fwd_t<avx2, false, data_type::s8, data_type::s8>::execute_forward();
template void _jit_uni_x8s8s32x_1x1_convolution_fwd_t<avx2, false, data_type::s8, data_type::s32>::execute_forward();
template void _jit_uni_x8s8s32x_1x1_convolution_fwd_t<avx2, false, data_type::s8, data_type::f32>::execute_forward();

template void _jit_uni_x8s8s32x_1x1_convolution_fwd_t<sse42, true, data_type::u8, data_type::u8>::execute_forward();
template void _jit_uni_x8s8s32x_1x1_convolution_fwd_t<sse42, true, data_type::u8, data_type::s8>::execute_forward();
template void _jit_uni_x8s8s32x_1x1_convolution_fwd_t<sse42, true, data_type::u8, data_type::s32>::execute_forward();
template void _jit_uni_x8s8s32x_1x1_convolution_fwd_t<sse42, true, data_type::u8, data_type::f32>::execute_forward();
template void _jit_uni_x8s8s32x_1x1_convolution_fwd_t<sse42, false, data_type::u8, data_type::u8>::execute_forward();
template void _jit_uni_x8s8s32x_1x1_convolution_fwd_t<sse42, false, data_type::u8, data_type::s8>::execute_forward();
template void _jit_uni_x8s8s32x_1x1_convolution_fwd_t<sse42, false, data_type::u8, data_type::s32>::execute_forward();
template void _jit_uni_x8s8s32x_1x1_convolution_fwd_t<sse42, false, data_type::u8, data_type::f32>::execute_forward();

template void _jit_uni_x8s8s32x_1x1_convolution_fwd_t<sse42, true, data_type::s8, data_type::u8>::execute_forward();
template void _jit_uni_x8s8s32x_1x1_convolution_fwd_t<sse42, true, data_type::s8, data_type::s8>::execute_forward();
template void _jit_uni_x8s8s32x_1x1_convolution_fwd_t<sse42, true, data_type::s8, data_type::s32>::execute_forward();
template void _jit_uni_x8s8s32x_1x1_convolution_fwd_t<sse42, true, data_type::s8, data_type::f32>::execute_forward();
template void _jit_uni_x8s8s32x_1x1_convolution_fwd_t<sse42, false, data_type::s8, data_type::u8>::execute_forward();
template void _jit_uni_x8s8s32x_1x1_convolution_fwd_t<sse42, false, data_type::s8, data_type::s8>::execute_forward();
template void _jit_uni_x8s8s32x_1x1_convolution_fwd_t<sse42, false, data_type::s8, data_type::s32>::execute_forward();
template void _jit_uni_x8s8s32x_1x1_convolution_fwd_t<sse42, false, data_type::s8, data_type::f32>::execute_forward();

}
}
}
