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

#include <iostream>
#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "mkldnn_thread.hpp"
#include "mkldnn_traits.hpp"
#include "math_utils.hpp"

#include "ref_deformable_convolution.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using math::saturate;
using math::get_bias;

void _ref_deformable_convolution_fwd_t::execute_forward() const {
    auto src = reinterpret_cast<const src_data_t *>(this->input_memory(0));
    auto offset = reinterpret_cast<const src_data_t *>(this->input_memory(1));
    auto weights = reinterpret_cast<const wei_data_t *>(this->input_memory(2));
    auto bias = reinterpret_cast<const char *>(this->input_memory(3));
    auto dst = reinterpret_cast<dst_data_t *>(this->memory());

    const memory_desc_wrapper src_d(pd()->src_pd(0));
    const memory_desc_wrapper offset_d(pd()->src_pd(1));
    const memory_desc_wrapper dst_d(pd()->dst_pd());
    const memory_desc_wrapper weights_d(pd()->weights_pd(0));
    const memory_desc_wrapper bias_d(pd()->weights_pd(1));

    const bool with_groups = pd()->with_groups();

    const int G = pd()->G();
    const int MB = pd()->MB();
    const int OH = pd()->OH();
    const int OW = pd()->OW();
    const int IH = pd()->IH();
    const int IW = pd()->IW();

    const int OC = pd()->OC() / G;
    const int IC = pd()->IC() / G;
    const int KH = pd()->KH();
    const int KW = pd()->KW();

    const int KSH = pd()->KSH();
    const int KSW = pd()->KSW();

    const int KDH = pd()->KDH();
    const int KDW = pd()->KDW();

    const int padT = pd()->padT();
    const int padL = pd()->padL();

    const int DG = pd()->defGroup();

    const int ndims = pd()->ndims();

    const int channel_per_deformable_group = pd()->IC() / DG;

    auto ker = [=](int g, int mb, int oc, int oh, int ow) {
        acc_data_t d = 0;
        const int h_in = oh * KSH - padT;
        const int w_in = ow * KSW - padL;

        for (int ic = 0; ic < IC; ic++) {
            const float *data_im_ptr = src + src_d.off(mb, g * IC + ic, h_in, w_in);
            const int deformable_group_index = ic / channel_per_deformable_group;
            const float *data_offset_ptr = offset + offset_d.off(mb, deformable_group_index * 2 * KH * KW, 0, 0);
            for (int kh = 0; kh < KH; kh++) {
                for (int kw = 0; kw < KW; kw++) {
                    const size_t data_offset_h_index = offset_d.off(0, 2 * (kh * KW + kw), oh, ow);
                    const size_t data_offset_w_index = offset_d.off(0, 2 * (kh * KW + kw) + 1, oh, ow);
                    const float offset_h = data_offset_ptr[data_offset_h_index];
                    const float offset_w = data_offset_ptr[data_offset_w_index];
                    float val = 0.0f;
                    const float h_im = h_in + kh * (KDH + 1) + offset_h;
                    const float w_im = w_in + kw * (KDW + 1) + offset_w;

                    if (h_im >= 0 && w_im >= 0 && h_im < IH && w_im < IW) {
                        float map_h = kh * (KDH + 1) + offset_h;
                        float map_w = kw * (KDW + 1) + offset_w;
                        const int cur_height = IH - h_in;
                        const int cur_width = IW - w_in;
                        int h_low = static_cast<int>(floor(map_h));
                        int w_low = static_cast<int>(floor(map_w));
                        int h_high;
                        int w_high;
                        if (h_low >= cur_height - 1) {
                            h_high = h_low = cur_height - 1;
                            map_h = static_cast<float>(h_low);
                        } else {
                            h_high = h_low + 1;
                        }

                        if (w_low >= cur_width - 1) {
                            w_high = w_low = cur_width - 1;
                            map_w = static_cast<float>(w_low);
                        } else {
                            w_high = w_low + 1;
                        }

                        float lh = map_h - h_low;
                        float lw = map_w - w_low;
                        float hh = 1 - lh, hw = 1 - lw;

                        float v1 = data_im_ptr[src_d.off(0, 0, h_low, w_low)];
                        float v2 = data_im_ptr[src_d.off(0, 0, h_low, w_high)];
                        float v3 = data_im_ptr[src_d.off(0, 0, h_high, w_low)];
                        float v4 = data_im_ptr[src_d.off(0, 0, h_high, w_high)];
                        float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

                        val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
                    }
                    d += val * (with_groups ? weights[weights_d.off(g, oc, ic, kh, kw)]
                                            : weights[weights_d.off(oc, ic, kh, kw)]);
                }
            }
        }

        return d;
    };

    parallel_nd(G, MB, OC, OH, OW,
    [&](int g, int mb, int oc, int oh, int ow) {
        float a_fp = ker(g, mb, oc, oh, ow);

        if (bias)
            a_fp += get_bias(bias, bias_d.off(g * OC + oc),
                             pd()->desc()->bias_desc.data_type);

        if (ndims == 4)
            dst[dst_d.off(mb, g * OC + oc, oh, ow)] = saturate<dst_data_t>(a_fp);
        else
            assert(false);
    });
}
}
}
}
