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

#include <common/utils.hpp>
#include <common/primitive_attr.hpp>
#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "mkldnn_thread.hpp"
#include "mkldnn_traits.hpp"
#include "math_utils.hpp"

#include "ref_binary_convolution.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using math::saturate;

void _ref_binary_convolution_fwd_t::execute_forward() const {
    auto src = reinterpret_cast<const uint8_t*>(this->input_memory(0));
    auto weights = reinterpret_cast<const uint8_t*>(this->input_memory(1));

    const memory_desc_wrapper src_d(pd()->src_pd());
    const memory_desc_wrapper dst_d(pd()->dst_pd());
    const memory_desc_wrapper weights_d(pd()->weights_pd(0));

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

    const int KDD = pd()->KDD();
    const int KDH = pd()->KDH();
    const int KDW = pd()->KDW();

    const int padFront = pd()->padFront();
    const int padT = pd()->padT();
    const int padL = pd()->padL();

    const float pad_value = pd()->pad_value();

    const int ndims = pd()->cdesc()->src_desc.ndims;

    const int nbits = 8;

    const auto &p = pd()->attr()->post_ops_;
    bool with_sum = p.find(primitive_kind::sum) != -1;
    bool with_binarization = p.find(primitive_kind::binarization) != -1;

    auto extract_bit = [](uint8_t val, uint8_t bit) -> uint8_t {
        return (uint8_t)((val >> bit) & 0x0001);
    };

    auto ker = [=](int32_t &d, int g, int mb, int oc, int od, int oh, int ow) {
        for (int ic = 0; ic < IC; ++ic)
        for (int kd = 0; kd < KD; ++kd)
        for (int kh = 0; kh < KH; ++kh)
        for (int kw = 0; kw < KW; ++kw) {
            const int id = od * KSD - padFront + kd * (1 + KDD);
            const int ih = oh * KSH - padT + kh * (1 + KDH);
            const int iw = ow * KSW - padL + kw * (1 + KDW);

            size_t iidx = 0;
            size_t widx = 0;
            if (ndims == 5) {
                iidx = src_d.off(mb, g * IC + ic, id, ih, iw);
                widx = with_groups ? weights_d.off(g, oc, ic, kd, kh, kw)
                                   : weights_d.off(oc, ic, kd, kh, kw);
            } else if (ndims == 4) {
                iidx = src_d.off(mb, g * IC + ic, ih, iw);
                widx = with_groups ? weights_d.off(g, oc, ic, kh, kw)
                                   : weights_d.off(oc, ic, kh, kw);
            } else if (ndims == 3) {
                iidx = src_d.off(mb, g * IC + ic, iw);
                widx = with_groups ? weights_d.off(g, oc, ic, kw)
                                   : weights_d.off(oc, ic, kw);
            } else {
                assert(false);
            }


            uint8_t s;
            if (id < 0 || id >= ID || ih < 0 || ih >= IH || iw < 0 || iw >= IW) {
                if (pad_value == 0)
                    continue;
                else {
                    s = pad_value == 1.0f ? (uint8_t)1 : (uint8_t)0;
                }
            }  else {
                s = extract_bit(src[iidx/nbits], (uint8_t)(iidx % nbits));
            }

            uint8_t w = extract_bit(weights[widx/nbits], (uint8_t)(widx % nbits));

            d += (int32_t)(s ^ w);
       }
    };

    if (with_binarization) {
        auto dst = reinterpret_cast<uint8_t*>(this->memory());

        int binarization_idx = p.find(primitive_kind::binarization);
        const float* binarization_weights = p.entry_[binarization_idx].binarization.thresholds_data;
        const uint32_t* binarization_output_mask = (uint32_t*)p.entry_[binarization_idx].binarization.output_mask_data;

        parallel_nd(G, MB, utils::div_up(OC, nbits), OD, OH, OW,
            [&](int g, int mb, int ocb, int od, int oh, int ow) {

            uint8_t bin_val = 0x00;
            for (int oc = ocb * nbits, shift = 0; oc < std::min(OC, (ocb + 1) * nbits); oc++, shift++) {
                int32_t a = 0;
                ker(a, g, mb, oc, od, oh, ow);

                float base_value;
                if (pad_value == 0.0f) {
                    const int i_left_overflow = nstl::max(0, (padL - ow * KSW));
                    const int i_right_overflow = nstl::max(IW, (ow * KSW + (KW - 1) * (KDW + 1) - padL + 1)) - IW;
                    const int kw_padding =
                            KW - utils::div_up(i_left_overflow, (KDW + 1)) - utils::div_up(i_right_overflow, (KDW + 1));

                    const int i_top_overflow = nstl::max(0, (padT - oh * KSH));
                    const int i_bottom_overflow = nstl::max(IH, (oh * KSH + (KH - 1) * (KDH + 1) - padT + 1)) - IH;
                    const int kh_padding =
                            KH - utils::div_up(i_top_overflow, (KDH + 1)) - utils::div_up(i_bottom_overflow, (KDH + 1));

                    const int i_front_overflow = nstl::max(0, (padFront - od * KSD));
                    const int i_back_overflow = nstl::max(ID, (od * KSD + (KD - 1) * (KDD + 1) - padFront + 1)) - ID;
                    const int kd_padding =
                            KD - utils::div_up(i_front_overflow, (KDD + 1)) - utils::div_up(i_back_overflow, (KDD + 1));

                    base_value = IC * kd_padding * kh_padding * kw_padding;
                } else {
                    base_value = IC * KD * KH * KW;
                }

                float a_fp = base_value - (float)(2 * a);

                if (with_sum) {
                    if (ndims == 5)
                        a_fp += dst[dst_d.off(mb, g * OC + oc, od, oh, ow)];
                    else if (ndims == 4)
                        a_fp += dst[dst_d.off(mb, g * OC + oc, oh, ow)];
                    else if (ndims == 3)
                        a_fp += dst[dst_d.off(mb, g * OC + oc, ow)];
                    else
                        assert(false);
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
                    }
                }

                float thr = binarization_weights[g * OC + oc];
                uint32_t out_mask = binarization_output_mask[g * OC + oc];
                uint32_t res = (a_fp > thr) ? 0xffffffff : 0x00000000;

                auto bit = uint8_t((res == out_mask) ? 0x01 : 0x00);
                bin_val |= (bit << shift);
            }

            if (ndims == 5)
                dst[dst_d.off(mb, g*OC + ocb*nbits, od, oh, ow) / nbits] = bin_val;
            else if (ndims == 4)
                dst[dst_d.off(mb, g*OC + ocb*nbits, oh, ow) / nbits] = bin_val;
            else if (ndims == 3)
                dst[dst_d.off(mb, g*OC + ocb*nbits, ow) / nbits] = bin_val;
            else
                assert(false);
        });
    } else {
        auto dst = reinterpret_cast<float*>(this->memory());

        parallel_nd(G, MB, OC, OD, OH, OW,
            [&](int g, int mb, int oc, int od, int oh, int ow) {
            int32_t a = 0;
            ker(a, g, mb, oc, od, oh, ow);

            float base_value;
            if (pad_value == 0.0f) {
                const int i_left_overflow = nstl::max(0, (padL - ow * KSW));
                const int i_right_overflow = nstl::max(IW, (ow * KSW + (KW - 1) * (KDW + 1) - padL + 1)) - IW;
                const int kw_padding =
                        KW - utils::div_up(i_left_overflow, (KDW + 1)) - utils::div_up(i_right_overflow, (KDW + 1));

                const int i_top_overflow = nstl::max(0, (padT - oh * KSH));
                const int i_bottom_overflow = nstl::max(IH, (oh * KSH + (KH - 1) * (KDH + 1) - padT + 1)) - IH;
                const int kh_padding =
                        KH - utils::div_up(i_top_overflow, (KDH + 1)) - utils::div_up(i_bottom_overflow, (KDH + 1));

                const int i_front_overflow = nstl::max(0, (padFront - od * KSD));
                const int i_back_overflow = nstl::max(ID, (od * KSD + (KD - 1) * (KDD + 1) - padFront + 1)) - ID;
                const int kd_padding =
                        KD - utils::div_up(i_front_overflow, (KDD + 1)) - utils::div_up(i_back_overflow, (KDD + 1));

                base_value = IC * kd_padding * kh_padding * kw_padding;
            } else {
                base_value = IC * KD * KH * KW;
            }

            float a_fp = base_value - (float)(2 * a);

            if (with_sum) {
                if (ndims == 5)
                    a_fp += dst[dst_d.off(mb, g*OC + oc, od, oh, ow)];
                else if (ndims == 4)
                    a_fp += dst[dst_d.off(mb, g*OC + oc, oh, ow)];
                else if (ndims == 3)
                    a_fp += dst[dst_d.off(mb, g*OC + oc, ow)];
                else
                    assert(false);
            }

            int eltwise_inj_idx = 0;
            int depthwise_inj_idx = 0;
            for (int i = 0; i < p.len_; i++) {
                auto& post_op = p.entry_[i];
                if (post_op.is_eltwise()) {
                    a_fp = eltwise_injectors[eltwise_inj_idx]->compute_scalar(a_fp);
                    eltwise_inj_idx++;
                } else if (post_op.is_depthwise()) {
                    auto depthwise_weights = post_op.depthwise.weights_data;
                    auto depthwise_bias = post_op.depthwise.biases_data;

                    a_fp = depthwise_injectors[depthwise_inj_idx]->compute_scalar(a_fp, depthwise_weights + g * OC + oc,
                                                                                        depthwise_bias + g * OC + oc);
                    depthwise_inj_idx++;
                }
            }

            if (ndims == 5)
                dst[dst_d.off(mb, g*OC + oc, od, oh, ow)] = a_fp;
            else if (ndims == 4)
                dst[dst_d.off(mb, g*OC + oc, oh, ow)] = a_fp;
            else if (ndims == 3)
                dst[dst_d.off(mb, g*OC + oc, ow)] = a_fp;
            else
                assert(false);
        });
    }
}

}
}
}
