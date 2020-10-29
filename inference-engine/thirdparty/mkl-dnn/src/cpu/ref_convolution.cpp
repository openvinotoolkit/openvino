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

#include <common/primitive_attr.hpp>
#include <common/math_utils.hpp>
#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "mkldnn_thread.hpp"
#include "mkldnn_traits.hpp"
#include "math_utils.hpp"

#include "ref_convolution.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using math::saturate;
using math::get_bias;
using math::get_sum;

template <data_type_t src_type, data_type_t wei_type,
         data_type_t dst_type, data_type_t acc_type>
void ref_convolution_fwd_t<src_type, wei_type, dst_type, acc_type>
        ::execute_forward() const {
    auto src = reinterpret_cast<const src_data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const wei_data_t *>(this->input_memory(1));
    auto bias = reinterpret_cast<const char *>(this->input_memory(2));
    auto dst = reinterpret_cast<dst_data_t *>(this->memory());

    const uint8_t* input_zero_points = pd()->attr()->input_zero_points_.shifts_;
    size_t input_zero_points_count = pd()->attr()->input_zero_points_.count_;

    const float* weights_zero_points = pd()->attr()->weights_zero_points_.shifts_;
    size_t weights_zero_points_count = pd()->attr()->weights_zero_points_.count_;

    const memory_desc_wrapper src_d(pd()->src_pd());
    const memory_desc_wrapper dst_d(pd()->dst_pd());
    const memory_desc_wrapper weights_d(pd()->weights_pd(0));
    const memory_desc_wrapper bias_d(pd()->weights_pd(1));

    const bool with_groups = pd()->with_groups();

    const int G = pd()->G();
    const int MB = pd()->MB();
    const int OD = pd()->OD();
    const int OH = pd()->OH();
    const int OW = pd()->OW();
    const int ID = pd()->ID();
    const int IH = pd()->IH();
    const int IW = pd()->IW();

    const int OC = pd()->OC() / G;
    const int IC = pd()->IC() / G;
    const int KD = pd()->KD();
    const int KH = pd()->KH();
    const int KW = pd()->KW();

    const int KSD = pd()->KSD();
    const int KSH = pd()->KSH();
    const int KSW = pd()->KSW();

    const int KDD = pd()->KDD() + 1;
    const int KDH = pd()->KDH() + 1;
    const int KDW = pd()->KDW() + 1;

    const int padFront = pd()->padFront();
    const int padT = pd()->padT();
    const int padL = pd()->padL();

    const int ndims = pd()->desc()->src_desc.ndims;

    const auto &p = pd()->attr()->post_ops_;

    auto ker = [=](int g, int mb, int oc, int od, int oh,
            int ow) {
        acc_data_t d = 0;
        for (int ic = 0; ic < IC; ++ic)
        for (int kd = 0; kd < KD; ++kd)
        for (int kh = 0; kh < KH; ++kh)
        for (int kw = 0; kw < KW; ++kw) {
            const int id = od * KSD - padFront + kd * KDD;
            const int ih = oh * KSH - padT + kh * KDH;
            const int iw = ow * KSW - padL + kw * KDW;

            if (id < 0 || id >= ID) continue;
            if (ih < 0 || ih >= IH) continue;
            if (iw < 0 || iw >= IW) continue;

            src_data_t s = 0;
            wei_data_t w = 0;
            if (ndims == 5) {
                s = src[src_d.off(mb, g * IC + ic, id, ih, iw)];
                w = (with_groups
                     ? weights[weights_d.off(g, oc, ic, kd, kh, kw)]
                     : weights[weights_d.off(oc, ic, kd, kh, kw)]);
            } else if (ndims == 4) {
                s = src[src_d.off(mb, g * IC + ic, ih, iw)];
                w = (with_groups
                     ? weights[weights_d.off(g, oc, ic, kh, kw)]
                     : weights[weights_d.off(oc, ic, kh, kw)]);
            } else if (ndims == 3) {
                s = src[src_d.off(mb, g * IC + ic, iw)];
                w = (with_groups
                     ? weights[weights_d.off(g, oc, ic, kw)]
                     : weights[weights_d.off(oc, ic, kw)]);
            } else {
                assert(false);
            }

            acc_data_t i_zp = 0;
            if (input_zero_points)
                i_zp = input_zero_points[input_zero_points_count == 1 ? 0 : g * IC + ic];

            acc_data_t w_zp = 0;
            if (weights_zero_points)
                w_zp = weights_zero_points[weights_zero_points_count == 1 ? 0 : g * OC + oc];

            d += ((acc_data_t)s - i_zp) * ((acc_data_t)w - w_zp);
        }
        return d;
    };

    // help compiler optimize the code
    // constants for plain layouts kernel
    const mkldnn_strides_t &src_str = src_d.blocking_desc().strides[0];
    const ptrdiff_t src_ic_stride = src_str[1];
    const ptrdiff_t src_id_stride = (ndims == 5) ? src_str[2] : 0;
    const ptrdiff_t src_ih_stride = (ndims >= 4) ? src_str[ndims - 2] : 0;
    const ptrdiff_t src_iw_stride = (ndims >= 3) ? src_str[ndims - 1] : 0;
    const mkldnn_strides_t &weights_str = weights_d.blocking_desc().strides[0];
    const int gr_shift = with_groups ? 1 : 0;
    const ptrdiff_t weights_ic_stride = weights_str[1 + gr_shift];
    const ptrdiff_t weights_kd_stride
            = (ndims == 5) ? weights_str[2 + gr_shift] : 0;
    const ptrdiff_t weights_kh_stride
            = (ndims >= 4) ? weights_str[ndims - 2 + gr_shift] : 0;
    const ptrdiff_t weights_kw_stride
            = (ndims >= 3) ? weights_str[ndims - 1 + gr_shift] : 0;

    auto ker_plain = [=](int g, int mb, int oc, int od, int oh, int ow) {
        assert(3 <= ndims && ndims <= 5);
        acc_data_t d = 0;
        const size_t src_loc_off = (ndims == 5)
                ? src_d.off(mb, g * IC, 0, 0, 0)
                : (ndims == 4) ? src_d.off(mb, g * IC, 0, 0)
                               : (ndims == 3) ? src_d.off(mb, g * IC, 0) : 0;

        const size_t weights_loc_off = (ndims == 5)
                ? (with_groups ? weights_d.off(g, oc, 0, 0, 0, 0)
                               : weights_d.off(oc, 0, 0, 0, 0))
                : (ndims == 4) ? (with_groups ? weights_d.off(g, oc, 0, 0, 0)
                                              : weights_d.off(oc, 0, 0, 0))
                               : (ndims == 3)
                                ? (with_groups ? weights_d.off(g, oc, 0, 0)
                                               : weights_d.off(oc, 0, 0))
                                : 0;

        const src_data_t *__restrict src_loc = src + src_loc_off;
        const wei_data_t *__restrict weights_loc = weights + weights_loc_off;

        if (IC > KW) {
            for (ptrdiff_t kd = 0; kd < KD; ++kd)
            for (ptrdiff_t kh = 0; kh < KH; ++kh)
            for (ptrdiff_t kw = 0; kw < KW; ++kw) {
#ifdef __INTEL_COMPILER
                // to avoid excessive compiler optimization
                volatile acc_data_t temp_var{ 0 };
#endif
                const ptrdiff_t id = od * KSD - padFront + kd * KDD;
                const ptrdiff_t ih = oh * KSH - padT + kh * KDH;
                const ptrdiff_t iw = ow * KSW - padL + kw * KDW;
                if (id < 0 || id >= ID || ih < 0 || ih >= IH || iw < 0
                        || iw >= IW)
                    continue;
                for (int ic = 0; ic < IC; ++ic) {
                    const size_t src_off = ic * src_ic_stride
                            + id * src_id_stride + ih * src_ih_stride
                            + iw * src_iw_stride;
                    const size_t weights_off = ic * weights_ic_stride
                            + kd * weights_kd_stride + kh * weights_kh_stride
                            + kw * weights_kw_stride;

                    src_data_t s = src_loc[src_off];
                    wei_data_t w = weights_loc[weights_off];

                    acc_data_t i_zp = 0;
                    if (input_zero_points)
                        i_zp = input_zero_points[input_zero_points_count == 1 ? 0 : g * IC + ic];

                    acc_data_t w_zp = 0;
                    if (weights_zero_points)
                        w_zp = weights_zero_points[weights_zero_points_count == 1 ? 0 : g * OC + oc];

                    d += ((acc_data_t)s - i_zp) * ((acc_data_t)w - w_zp);
                }
            }
        } else {
            for (ptrdiff_t ic = 0; ic < IC; ++ic)
            for (ptrdiff_t kd = 0; kd < KD; ++kd)
            for (ptrdiff_t kh = 0; kh < KH; ++kh)
            for (ptrdiff_t kw = 0; kw < KW; ++kw) {
#ifdef __INTEL_COMPILER
                // to avoid excessive compiler optimization
                volatile acc_data_t temp_var{ 0 };
#endif
                const ptrdiff_t id = od * KSD - padFront + kd * KDD;
                const ptrdiff_t ih = oh * KSH - padT + kh * KDH;
                const ptrdiff_t iw = ow * KSW - padL + kw * KDW;
                if (id < 0 || id >= ID || ih < 0 || ih >= IH || iw < 0
                        || iw >= IW)
                    continue;
                const ptrdiff_t src_off = ic * src_ic_stride
                        + id * src_id_stride + ih * src_ih_stride
                        + iw * src_iw_stride;
                const ptrdiff_t weights_off = ic * weights_ic_stride
                        + kd * weights_kd_stride + kh * weights_kh_stride
                        + kw * weights_kw_stride;

                src_data_t s = src_loc[src_off];
                wei_data_t w = weights_loc[weights_off];

                acc_data_t i_zp = 0;
                if (input_zero_points)
                    i_zp = input_zero_points[input_zero_points_count == 1 ? 0 : g * IC + ic];

                acc_data_t w_zp = 0;
                if (weights_zero_points)
                    w_zp = weights_zero_points[weights_zero_points_count == 1 ? 0 : g * OC + oc];

                d += ((acc_data_t)s - i_zp) * ((acc_data_t)w - w_zp);
            }
        }
        return d;
    };

    parallel_nd(G, MB, OC, OD, OH, OW,
        [&](int g, int mb, int oc, int od, int oh, int ow) {
        float a_fp = bias
            ? get_bias(bias, bias_d.off(g * OC + oc),
                    pd()->desc()->bias_desc.data_type)
            : 0;
        if (src_d.is_plain() && weights_d.is_plain())
            a_fp += ker_plain(g, mb, oc, od, oh, ow);
        else
            a_fp += ker(g, mb, oc, od, oh, ow);

        if (!pd()->attr()->output_scales_.has_default_values()) {
            a_fp *= pd()->attr()->output_scales_.scales_[g * OC + oc];
        }

        int eltwise_inj_idx = 0;
        int depthwise_inj_idx = 0;
        for (int i = 0; i < p.len_; i++) {
            auto &post_op = p.entry_[i];
            if (post_op.is_eltwise()) {
                a_fp = eltwise_injectors[eltwise_inj_idx]->compute_scalar(a_fp);
                eltwise_inj_idx++;
            } else if (post_op.is_depthwise()) {
                auto depthwise_weights = post_op.depthwise.weights_data;
                auto depthwise_bias = post_op.depthwise.biases_data;

                a_fp = depthwise_injectors[depthwise_inj_idx]->compute_scalar(a_fp,
                                                                              depthwise_weights + g * OC + oc,
                                                                              depthwise_bias + g * OC + oc);
                depthwise_inj_idx++;
            } else if (post_op.is_quantization()) {
                auto quant = post_op.quantization;
                float cl = quant.crop_low_data->shifts_[quant.crop_low_data->count_ == 1 ? 0 : g * OC + oc];
                float ch = quant.crop_high_data->shifts_[quant.crop_high_data->count_ == 1 ? 0 : g * OC + oc];
                float isc = quant.input_scale_data->scales_[quant.input_scale_data->count_ == 1 ? 0 : g * OC + oc];
                float ish = quant.input_shift_data->shifts_[quant.input_shift_data->count_ == 1 ? 0 : g * OC + oc];
                float osc = quant.output_scale_data->scales_[quant.output_scale_data->count_ == 1 ? 0 : g * OC + oc];
                float osh = quant.output_shift_data->shifts_[quant.output_shift_data->count_ == 1 ? 0 : g * OC + oc];

                a_fp = nstl::min(ch, nstl::max(cl, a_fp));
                a_fp = a_fp * isc + ish;
                a_fp = roundf(a_fp);
                a_fp = a_fp * osc + osh;
            } else if (post_op.is_sum()) {
                if (ndims == 5)
                    a_fp += get_sum((char*)dst, dst_d.off(mb, g * OC + oc, od, oh, ow), post_op.sum.data_type);
                else if (ndims == 4)
                    a_fp += get_sum((char*)dst, dst_d.off(mb, g * OC + oc, oh, ow), post_op.sum.data_type);
                else if (ndims == 3)
                    a_fp += get_sum((char*)dst, dst_d.off(mb, g * OC + oc, ow), post_op.sum.data_type);
                else
                    assert(false);
            }
        }

        if (data_traits<dst_data_t>::data_type != data_type::f32) {
            switch (pd()->attr()->round_mode_) {
                case round_mode::down:    a_fp = floorf(a_fp); break;
                case round_mode::nearest: a_fp = nearbyintf(a_fp); break;
            }
        }

        if (ndims == 5)
            dst[dst_d.off(mb, g*OC + oc, od, oh, ow)] = saturate<dst_data_t>(a_fp);
        else if (ndims == 4)
            dst[dst_d.off(mb, g*OC + oc, oh, ow)] = saturate<dst_data_t>(a_fp);
        else if (ndims == 3)
            dst[dst_d.off(mb, g*OC + oc, ow)] = saturate<dst_data_t>(a_fp);
        else
            assert(false);
   });
}

template <data_type_t diff_src_type, data_type_t wei_type,
         data_type_t diff_dst_type, data_type_t acc_type>
void ref_convolution_bwd_data_t<diff_src_type, wei_type, diff_dst_type,
     acc_type>::execute_backward_data() const {
    auto diff_dst = reinterpret_cast<const diff_dst_data_t*>(
            this->input_memory(0));
    auto weights = reinterpret_cast<const wei_data_t*>(this->input_memory(1));
    auto bias = reinterpret_cast<const char *>(this->input_memory(2));
    auto diff_src = reinterpret_cast<diff_src_data_t*>(this->memory());

    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_pd());
    const memory_desc_wrapper diff_src_d(pd()->diff_src_pd());
    const memory_desc_wrapper weights_d(pd()->weights_pd(0));
    const memory_desc_wrapper bias_d(pd()->weights_pd(1));

    const bool with_groups = pd()->with_groups();

    const int G = pd()->G();
    const int MB = pd()->MB();
    const int OD = pd()->OD();
    const int OH = pd()->OH();
    const int OW = pd()->OW();
    const int ID = pd()->ID();
    const int IH = pd()->IH();
    const int IW = pd()->IW();

    const int OC = pd()->OC() / G;
    const int IC = pd()->IC() / G;
    const int KD = pd()->KD();
    const int KH = pd()->KH();
    const int KW = pd()->KW();

    const int KSD = pd()->KSD();
    const int KSH = pd()->KSH();
    const int KSW = pd()->KSW();

    const int KDD = pd()->KDD() + 1;
    const int KDH = pd()->KDH() + 1;
    const int KDW = pd()->KDW() + 1;

    const int padFront = pd()->padFront();
    const int padT = pd()->padT();
    const int padL = pd()->padL();

    const int ndims = pd()->desc()->diff_src_desc.ndims;

    const auto &p = pd()->attr()->post_ops_;

    auto ker = [=](int g, int mb, int ic, int id, int ih,
            int iw) {
        acc_data_t d = 0;
        for (int oc = 0; oc < OC; ++oc)
        for (int kd = 0; kd < KD; ++kd)
        for (int kh = 0; kh < KH; ++kh)
        for (int kw = 0; kw < KW; ++kw) {
            if (iw + padL < kw * KDW
                || ih + padT < kh * KDH
                || id + padFront < kd * KDD)
                continue;
            int ow = iw - kw * KDW + padL;
            int oh = ih - kh * KDH + padT;
            int od = id - kd * KDD + padFront;
            if (ow % KSW != 0 || oh % KSH != 0 || od % KSD != 0)
                continue;

            ow /= KSW;
            oh /= KSH;
            od /= KSD;

            if (od < OD && oh < OH && ow < OW) {
                if (ndims == 5)
                    d += (acc_data_t)diff_dst[diff_dst_d.off(mb, g*OC
                        + oc, od, oh, ow)] * (with_groups
                        ? weights[weights_d.off(g, oc, ic, kd, kh, kw)]
                        : weights[weights_d.off(oc, ic, kd, kh, kw)]);
                else if (ndims == 4)
                    d += (acc_data_t)diff_dst[diff_dst_d.off(mb, g*OC
                        + oc, oh, ow)] * (with_groups
                        ? weights[weights_d.off(g, oc, ic, kh, kw)]
                        : weights[weights_d.off(oc, ic, kh, kw)]);
                else if (ndims == 3)
                    d += (acc_data_t)diff_dst[diff_dst_d.off(mb, g*OC
                        + oc, ow)] * (with_groups
                        ? weights[weights_d.off(g, oc, ic, kw)]
                        : weights[weights_d.off(oc, ic, kw)]);
                else
                    assert(false);
            }
        }
        return d;
    };

    // help compiler optimize the code
    // constants for plain layouts kernel
    const mkldnn_strides_t &diff_dst_str
            = diff_dst_d.blocking_desc().strides[0];
    const ptrdiff_t diff_dst_oc_stride = diff_dst_str[1];
    const ptrdiff_t diff_dst_ow_stride = diff_dst_str[ndims - 1];
    const ptrdiff_t diff_dst_oh_stride
            = (ndims >= 4) ? diff_dst_str[ndims - 2] : 0;
    const ptrdiff_t diff_dst_od_stride
            = (ndims >= 5) ? diff_dst_str[ndims - 3] : 0;

    const mkldnn_strides_t &weights_str = weights_d.blocking_desc().strides[0];
    const int gr_shift = with_groups ? 1 : 0;
    const ptrdiff_t weights_oc_stride = weights_str[0 + gr_shift];
    const ptrdiff_t weights_kw_stride = weights_str[ndims - 1 + gr_shift];
    const ptrdiff_t weights_kh_stride
            = (ndims >= 4) ? weights_str[ndims - 2 + gr_shift] : 0;
    const ptrdiff_t weights_kd_stride
            = (ndims >= 4) ? weights_str[ndims - 3 + gr_shift] : 0;

    auto ker_plain = [=](int g, int mb, int ic, int id, int ih, int iw) {
        assert(3 <= ndims && ndims <= 5);
        acc_data_t d = 0;
        const size_t diff_dst_loc_off = (ndims == 5)
                ? diff_dst_d.off(mb, g * OC, 0, 0, 0)
                : (ndims == 4)
                        ? diff_dst_d.off(mb, g * OC, 0, 0)
                        : (ndims == 3) ? diff_dst_d.off(mb, g * OC, 0) : 0;
        const size_t weights_loc_off = (ndims == 5)
                ? with_groups ? weights_d.off(g, 0, ic, 0, 0, 0)
                              : weights_d.off(0, ic, 0, 0, 0)
                : (ndims == 4) ? with_groups ? weights_d.off(g, 0, ic, 0, 0)
                                             : weights_d.off(0, ic, 0, 0)
                               : (ndims == 3) ? with_groups
                                        ? weights_d.off(g, 0, ic, 0)
                                        : weights_d.off(0, ic, 0)
                                              : 0;

        const diff_dst_data_t *__restrict diff_dst_loc
                = diff_dst + diff_dst_loc_off;
        const wei_data_t *__restrict weights_loc = weights + weights_loc_off;

        if (OC > KW) {
            for (ptrdiff_t kd = 0; kd < KD; ++kd)
            for (ptrdiff_t kh = 0; kh < KH; ++kh)
            for (ptrdiff_t kw = 0; kw < KW; ++kw) {
#ifdef __INTEL_COMPILER
                // to avoid excessive compiler optimization
                volatile acc_data_t temp_var{ 0 };
#endif
                ptrdiff_t ow = iw - kw * KDW + padL;
                ptrdiff_t oh = ih - kh * KDH + padT;
                ptrdiff_t od = id - kd * KDD + padFront;
                if (ow < 0 || oh < 0 || od < 0 || ow % KSW != 0 || oh % KSH != 0
                        || od % KSD != 0)
                    continue;
                ow /= KSW;
                oh /= KSH;
                od /= KSD;
                if (od >= OD || oh >= OH || ow >= OW)
                    continue;
                for (ptrdiff_t oc = 0; oc < OC; ++oc) {
                    const ptrdiff_t diff_dst_off = oc * diff_dst_oc_stride
                            + od * diff_dst_od_stride + oh * diff_dst_oh_stride
                            + ow * diff_dst_ow_stride;
                    const ptrdiff_t weights_off = oc * weights_oc_stride
                            + kd * weights_kd_stride + kh * weights_kh_stride
                            + kw * weights_kw_stride;
                    auto dd = (acc_data_t)diff_dst_loc[diff_dst_off]
                            * weights_loc[weights_off];
                    d += dd;
                }
            }
        } else {
            for (ptrdiff_t oc = 0; oc < OC; ++oc)
            for (ptrdiff_t kd = 0; kd < KD; ++kd)
            for (ptrdiff_t kh = 0; kh < KH; ++kh)
            for (ptrdiff_t kw = 0; kw < KW; ++kw) {
#ifdef __INTEL_COMPILER
                // to avoid excessive compiler optimization
                volatile acc_data_t temp_var{ 0 };
#endif
                ptrdiff_t ow = iw - kw * KDW + padL;
                ptrdiff_t oh = ih - kh * KDH + padT;
                ptrdiff_t od = id - kd * KDD + padFront;
                if (ow < 0 || oh < 0 || od < 0 || ow % KSW != 0 || oh % KSH != 0
                        || od % KSD != 0)
                    continue;
                ow /= KSW;
                oh /= KSH;
                od /= KSD;
                if (od >= OD || oh >= OH || ow >= OW)
                    continue;
                const ptrdiff_t diff_dst_off = oc * diff_dst_oc_stride
                        + od * diff_dst_od_stride + oh * diff_dst_oh_stride
                        + ow * diff_dst_ow_stride;
                const ptrdiff_t weights_off = oc * weights_oc_stride
                        + kd * weights_kd_stride + kh * weights_kh_stride
                        + kw * weights_kw_stride;
                d += (acc_data_t)diff_dst_loc[diff_dst_off]
                        * weights_loc[weights_off];
            }
        }
        return d;
    };

    parallel_nd(G, MB, IC, ID, IH, IW,
        [&](int g, int mb, int ic, int id, int ih, int iw) {
        auto ds_idx = (ndims == 5)
            ? diff_src_d.off(mb, g*IC + ic, id, ih, iw)
            : (ndims == 4)
            ? diff_src_d.off(mb, g*IC + ic, ih, iw)
            : diff_src_d.off(mb, g*IC + ic, iw);
        float a = bias
            ? get_bias(bias, bias_d.off(g * IC + ic),
                    pd()->desc()->bias_desc.data_type)
            : 0;
        if (diff_dst_d.is_plain() && weights_d.is_plain())
            a += ker_plain(g, mb, ic, id, ih, iw);
        else
            a += ker(g, mb, ic, id, ih, iw);

        int depthwise_inj_idx = 0;
        for (int i = 0; i < p.len_; i++) {
            auto &post_op = p.entry_[i];
            if (post_op.is_depthwise()) {
                auto depthwise_weights = post_op.depthwise.weights_data;
                auto depthwise_bias = post_op.depthwise.biases_data;

                a = depthwise_injectors[depthwise_inj_idx]->compute_scalar(a, depthwise_weights + g * IC + ic, depthwise_bias + g * IC + ic);
            }
            depthwise_inj_idx++;
        }

        diff_src[ds_idx] = saturate<diff_src_data_t>(a);
    });
}

template <data_type_t src_type, data_type_t diff_wei_type,
         data_type_t diff_dst_type, data_type_t acc_type>
void ref_convolution_bwd_weights_t<src_type, diff_wei_type, diff_dst_type,
     acc_type>::execute_backward_weights() const {
    auto src = reinterpret_cast<const src_data_t *>(this->input_memory(0));
    auto diff_dst = reinterpret_cast<const diff_dst_data_t *>(
            this->input_memory(1));
    auto diff_weights = reinterpret_cast<diff_wei_data_t*>(this->memory(0));
    auto diff_bias = reinterpret_cast<diff_wei_data_t *>(this->memory(1));

    const memory_desc_wrapper src_d(pd()->src_pd());
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_pd());
    const memory_desc_wrapper diff_weights_d(pd()->diff_weights_pd(0));
    const memory_desc_wrapper diff_bias_d(pd()->diff_weights_pd(1));

    const bool with_groups = pd()->with_groups();

    const int G = pd()->G();
    const int MB = pd()->MB();
    const int OD = pd()->OD();
    const int OH = pd()->OH();
    const int OW = pd()->OW();
    const int ID = pd()->ID();
    const int IH = pd()->IH();
    const int IW = pd()->IW();

    const int OC = pd()->OC() / G;
    const int IC = pd()->IC() / G;
    const int KD = pd()->KD();
    const int KH = pd()->KH();
    const int KW = pd()->KW();

    const int KSD = pd()->KSD();
    const int KSH = pd()->KSH();
    const int KSW = pd()->KSW();

    const int KDD = pd()->KDD() + 1;
    const int KDH = pd()->KDH() + 1;
    const int KDW = pd()->KDW() + 1;

    const int padFront = pd()->padFront();
    const int padT = pd()->padT();
    const int padL = pd()->padL();

    const int ndims = pd()->desc()->src_desc.ndims;

auto ker = [=](acc_data_t &d, int g, int oc, int ic, int kd, int kh, int kw) {
        for (int mb = 0; mb < MB; ++mb)
        for (int od = 0; od < OD; ++od)
        for (int oh = 0; oh < OH; ++oh)
        for (int ow = 0; ow < OW; ++ow) {
            if (ow * KSW + kw * KDW < padL || oh * KSH + kh * KDH < padT
                    || od * KSD + kd * KDD < padFront
                    || ow * KSW + kw * KDW >= IW + padL
                    || oh * KSH + kh * KDH >= IH + padT
                    || od * KSD + kd * KDD >= ID + padFront)
                continue;

            int id = od * KSD - padFront + kd * KDD;
            int ih = oh * KSH - padT + kh * KDH;
            int iw = ow * KSW - padL + kw * KDW;
            if (ndims == 5)
                d += (acc_data_t)diff_dst[diff_dst_d.off(mb, g*OC + oc, od,
                    oh, ow)] * src[src_d.off(mb, g*IC + ic, id, ih, iw)];
            else if (ndims == 4)
                d += (acc_data_t)diff_dst[diff_dst_d.off(mb, g*OC + oc, oh, ow)]
                    * src[src_d.off(mb, g*IC + ic, ih, iw)];
            else if (ndims == 3)
                d += (acc_data_t)diff_dst[diff_dst_d.off(mb, g*OC + oc, ow)]
                    * src[src_d.off(mb, g*IC + ic, iw)];
            else
                assert(false);
        }
    };

    auto ker_plain = [=](acc_data_t &d, int g, int oc, int ic, int kd, int kh,
                             int kw) {
        assert(3 <= ndims && ndims <= 5);
        // help compiler optimize the code
        // constants for plain layouts kernel
        const mkldnn_strides_t &diff_dst_str
                = diff_dst_d.blocking_desc().strides[0];
        const ptrdiff_t diff_dst_mb_stride = diff_dst_str[0];
        const ptrdiff_t diff_dst_ow_stride = diff_dst_str[ndims - 1];
        const ptrdiff_t diff_dst_oh_stride
                = (ndims >= 4) ? diff_dst_str[ndims - 2] : 0;
        const ptrdiff_t diff_dst_od_stride
                = (ndims >= 5) ? diff_dst_str[ndims - 3] : 0;
        const mkldnn_strides_t &src_str = src_d.blocking_desc().strides[0];
        const ptrdiff_t src_mb_stride = src_str[0];
        const ptrdiff_t src_iw_stride = src_str[ndims - 1];
        const ptrdiff_t src_ih_stride = (ndims >= 4) ? src_str[ndims - 2] : 0;
        const ptrdiff_t src_id_stride = (ndims >= 5) ? src_str[ndims - 3] : 0;

        const size_t diff_dst_loc_off = (ndims == 5)
                ? diff_dst_d.off(0, g * OC + oc, 0, 0, 0)
                : (ndims == 4)
                        ? diff_dst_d.off(0, g * OC + oc, 0, 0)
                        : (ndims == 3) ? diff_dst_d.off(0, g * OC + oc, 0) : 0;

        const size_t src_loc_off = (ndims == 5)
                ? src_d.off(0, g * IC + ic, 0, 0, 0)
                : (ndims == 4)
                        ? src_d.off(0, g * IC + ic, 0, 0)
                        : (ndims == 3) ? src_d.off(0, g * IC + ic, 0) : 0;

        const diff_dst_data_t *__restrict diff_dst_loc
                = diff_dst + diff_dst_loc_off;
        const src_data_t *__restrict src_loc = src + src_loc_off;

        for (ptrdiff_t mb = 0; mb < MB; ++mb)
        for (ptrdiff_t od = 0; od < OD; ++od)
        for (ptrdiff_t oh = 0; oh < OH; ++oh)
        for (ptrdiff_t ow = 0; ow < OW; ++ow) {
            const ptrdiff_t id = od * KSD - padFront + kd * KDD;
            const ptrdiff_t ih = oh * KSH - padT + kh * KDH;
            const ptrdiff_t iw = ow * KSW - padL + kw * KDW;
            if (id < 0 || id >= ID || ih < 0 || ih >= IH || iw < 0 || iw >= IW)
                continue;
            const ptrdiff_t diff_dst_off = mb * diff_dst_mb_stride
                    + od * diff_dst_od_stride + oh * diff_dst_oh_stride
                    + ow * diff_dst_ow_stride;
            const ptrdiff_t src_off = mb * src_mb_stride + id * src_id_stride
                    + ih * src_ih_stride + iw * src_iw_stride;
            d += (acc_data_t)diff_dst_loc[diff_dst_off] * src_loc[src_off];
        }
    };

    auto ker_bias = [=](acc_data_t &d, int g, int oc) {
        for (int mb = 0; mb < MB; ++mb)
        for (int od = 0; od < OD; ++od)
        for (int oh = 0; oh < OH; ++oh)
        for (int ow = 0; ow < OW; ++ow) {
            if (ndims == 5)
                d += (acc_data_t)diff_dst[diff_dst_d.off(mb, g*OC + oc, od, oh,
                     ow)];
            else if (ndims == 4)
                d += (acc_data_t)diff_dst[diff_dst_d.off(mb, g*OC + oc, oh,
                     ow)];
            else if (ndims == 3)
                d += (acc_data_t)diff_dst[diff_dst_d.off(mb, g*OC + oc, ow)];
            else
                assert(false);
        }
    };

    parallel_nd(G, OC, [&](int g, int oc) {
        if (diff_bias) {
            // XXX: loss of precision when bias is a float...
            acc_data_t db = 0;
            ker_bias(db, g, oc);
            diff_bias[diff_bias_d.off(g*OC+oc)]
                = saturate<diff_wei_data_t>(db);
        }

        for (int ic = 0; ic < IC; ++ic)
        for (int kd = 0; kd < KD; ++kd)
        for (int kh = 0; kh < KH; ++kh)
        for (int kw = 0; kw < KW; ++kw) {
            acc_data_t dw = 0;
            if (diff_dst_d.is_plain() && src_d.is_plain())
                ker_plain(dw, g, oc, ic, kd, kh, kw);
            else
                ker(dw, g, oc, ic, kd, kh, kw);

            if (ndims == 5) {
                auto idx = with_groups
                    ? diff_weights_d.off(g, oc, ic, kd, kh, kw)
                    : diff_weights_d.off(oc, ic, kd, kh, kw);
                    diff_weights[idx] = saturate<diff_wei_data_t>(dw);
            } else if (ndims == 4) {
                auto idx = with_groups
                    ? diff_weights_d.off(g, oc, ic, kh, kw)
                    : diff_weights_d.off(oc, ic, kh, kw);
                    diff_weights[idx] = saturate<diff_wei_data_t>(dw);
            } else if (ndims == 3) {
                auto idx = with_groups
                    ? diff_weights_d.off(g, oc, ic, kw)
                    : diff_weights_d.off(oc, ic, kw);
                    diff_weights[idx] = saturate<diff_wei_data_t>(dw);
            } else {
                 assert(false);
            }
        }
    });
}

using namespace data_type;

template struct ref_convolution_fwd_t<f32>;
template struct ref_convolution_fwd_t<s16, s16, s32, s32>;

template struct ref_convolution_fwd_t<u8, s8, f32, s32>;
template struct ref_convolution_fwd_t<u8, s8, s32, s32>;
template struct ref_convolution_fwd_t<u8, s8, s8, s32>;
template struct ref_convolution_fwd_t<u8, s8, u8, s32>;

template struct ref_convolution_bwd_data_t<f32, f32, f32, f32>;
template struct ref_convolution_bwd_data_t<s32, s16, s16, s32>;

template struct ref_convolution_bwd_data_t<f32, s8, u8, s32>;
template struct ref_convolution_bwd_data_t<s32, s8, u8, s32>;
template struct ref_convolution_bwd_data_t<s8, s8, u8, s32>;
template struct ref_convolution_bwd_data_t<u8, s8, u8, s32>;

template struct ref_convolution_bwd_weights_t<f32, f32, f32, f32>;
template struct ref_convolution_bwd_weights_t<s16, s32, s16, s32>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
