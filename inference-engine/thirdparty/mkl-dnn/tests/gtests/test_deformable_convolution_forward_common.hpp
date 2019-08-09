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

#ifndef TEST_DEFORMABLE_CONVOLUTION_FORWARD_COMMON_H
#define TEST_DEFORMABLE_CONVOLUTION_FORWARD_COMMON_H

#include "mkldnn_test_common.hpp"
#include "gtest/gtest.h"

#include "mkldnn.hpp"
#include <stdint.h>

#include <math.h>
#include "nstl.hpp"

namespace mkldnn {

    void compute_ref_def_conv_fwd(const test_deformable_convolution_sizes_t &c,
        const memory::desc &src_d,
        const memory::desc &offset_d,
        const memory::desc &weights_d,
        const memory::desc &bias_d,
        const memory::desc &dst_d,
        const memory &src,
        const memory &offsets,
        const memory &weights,
        const memory &bias,
        const memory &dst)
{
    const bool w_bias = bias_d.data.format != memory::format::format_undef;

    float* src_data = (float*)src.get_data_handle();
    float* offsets_data = (float*)offsets.get_data_handle();
    float* weights_data = (float*)weights.get_data_handle();
    float* bias_data = w_bias ? (float*)bias.get_data_handle() : nullptr;
    float* dst_data = (float *)dst.get_data_handle();

    size_t padded_ic = src_d.data.layout_desc.blocking.padding_dims[1];
    size_t padded_oc = dst_d.data.layout_desc.blocking.padding_dims[1];

    size_t padded_ic_w = weights_d.data.layout_desc.blocking.padding_dims[1];
    size_t padded_oc_w = weights_d.data.layout_desc.blocking.padding_dims[0];

    mkldnn::impl::parallel_nd(c.mb, c.ng, c.oc / c.ng, c.oh, c.ow,
        [&](int n, int g, int oc, int oh, int ow) {
            float a = 0;
            for (int ic = 0; ic < c.ic / c.ng; ic++) {
                for (int kh = 0; kh < c.kh; kh++) {
                    for (int kw = 0; kw < c.kw; kw++) {
                        size_t off_idx_w = (size_t)(size_t)n * c.dg * 2 * c.kh * c.kw * c.oh * c.ow +
                                                   (ic / (c.ic / c.dg)) * 2 * c.kh * c.kw * c.oh * c.ow +
                                                   ((2 * (kh * c.kw + kw) + 1) * c.oh + oh) * c.ow + ow;
                        off_idx_w = map_index(offset_d, off_idx_w);

                        size_t off_idx_h = (size_t)n * c.dg * 2 * c.kh * c.kw * c.oh * c.ow +
                                                   (ic / (c.ic / c.dg)) * 2 * c.kh * c.kw * c.oh * c.ow +
                                                   ((2 * (kh * c.kw + kw)) * c.oh + oh) * c.ow + ow;
                        off_idx_h = map_index(offset_d, off_idx_h);

                        float offset_w = offsets_data[off_idx_w];
                        float offset_h = offsets_data[off_idx_h];

                        int iw_in = ow * c.strw - c.padw;
                        int ih_in = oh * c.strh - c.padh;

                        float map_w = kw * (c.dilw + 1) + offset_w;
                        float map_h = kh * (c.dilh + 1) + offset_h;

                        float iw_im = iw_in + map_w;
                        float ih_im = ih_in + map_h;

                        float s_val = 0;
                        if (ih_im >= 0 && iw_im >= 0 && ih_im < c.ih && iw_im < c.iw) {

                            int cur_width = c.iw - iw_in;
                            int cur_height = c.ih - ih_in;

                            int h_low = static_cast<int>(floor(map_h));
                            int w_low = static_cast<int>(floor(map_w));

                            int h_high;
                            if (h_low >= cur_height - 1) {
                                h_high = h_low = cur_height - 1;
                                map_h = static_cast<float>(h_low);
                            } else {
                                h_high = h_low + 1;
                            }


                            int w_high;
                            if (w_low >= cur_width - 1) {
                                w_high = w_low = cur_width - 1;
                                map_w = static_cast<float>(w_low);
                            } else {
                                w_high = w_low + 1;
                            }

                            float lh = map_h - h_low;
                            float lw = map_w - w_low;
                            float hh = 1 - lh, hw = 1 - lw;

                            size_t iidx = n * padded_ic * c.ih * c.iw
                                          + g * padded_ic / c.ng * c.ih * c.iw
                                          + ic * c.ih * c.iw + ih_in * c.iw + iw_in;

                            float v1 = src_data[map_index(src_d, iidx + h_low * c.iw + w_low)];
                            float v2 = src_data[map_index(src_d, iidx + h_low * c.iw + w_high)];
                            float v3 = src_data[map_index(src_d, iidx + h_high * c.iw + w_low)];
                            float v4 = src_data[map_index(src_d, iidx + h_high * c.iw + w_high)];
                            float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

                            s_val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
                        }

                        size_t widx = g * padded_oc_w / c.ng * padded_ic_w
                            / c.ng * c.kh * c.kw
                            + oc * padded_ic_w / c.ng * c.kh * c.kw
                            + ic * c.kh * c.kw + kh * c.kw + kw;

                        size_t widx_ = map_index(weights_d, widx);
                        float w_val = weights_data[widx_];
                        a += s_val * w_val;
                    }
                }
            }

            a += (bias_data ? bias_data[map_index(bias_d, g * c.oc / c.ng + oc)] : 0);

            size_t oidx = n * padded_oc * c.oh * c.ow
                     + g * padded_oc / c.ng * c.oh * c.ow
                     + oc * c.oh * c.ow + oh * c.ow + ow;
            dst_data[map_index(dst_d, oidx)] = a;
        }
    );
}

template <typename data_t_src, typename data_t_off, typename data_t_wei,
          typename data_t_acc, typename data_t_dst>
class deformable_convolution_forward_test
        : public ::testing::TestWithParam<test_deformable_convolution_params_t> {
protected:
    virtual void SetUp() {
        auto p = ::testing::TestWithParam<test_deformable_convolution_params_t>::GetParam();
        ASSERT_TRUE(p.engine_kind == engine::kind::cpu);
        ASSERT_EQ(p.aalgorithm, algorithm::deformable_convolution_direct);
        auto eng = engine(p.engine_kind, 0);

        memory::data_type data_type_src = data_traits<data_t_src>::data_type;
        memory::data_type data_type_off = data_traits<data_t_off>::data_type;
        memory::data_type data_type_wei = data_traits<data_t_wei>::data_type;
        memory::data_type data_type_dst = data_traits<data_t_dst>::data_type;

        test_deformable_convolution_sizes_t dcs = p.sizes;

        auto aprop_kind = prop_kind::forward;
        bool with_bias = p.formats.bias_format != memory::format::format_undef;

        auto dc_src_desc = create_md({ dcs.mb, dcs.ic, dcs.ih, dcs.iw }, data_type_src, p.formats.src_format);
        auto dc_offsets_desc = create_md({ dcs.mb, 2 * dcs.dg * dcs.kh * dcs.kw, dcs.oh, dcs.ow }, data_type_off, p.formats.offsets_format);
        auto dc_weights_desc = dcs.ng > 1 ? create_md({ dcs.ng, dcs.oc / dcs.ng, dcs.ic / dcs.ng, dcs.kh, dcs.kw }, data_type_wei, p.formats.weights_format)
                                          : create_md({ dcs.oc, dcs.ic, dcs.kh, dcs.kw }, data_type_wei,p.formats.weights_format);
        auto dc_dst_desc = create_md({ dcs.mb, dcs.oc, dcs.oh, dcs.ow }, data_type_dst, p.formats.dst_format);
        auto dc_bias_desc = with_bias ? create_md({ dcs.oc }, data_type_dst, p.formats.bias_format)
                                      : create_md({}, data_type_dst, p.formats.bias_format);

        auto dc_src = test_memory(dc_src_desc, eng);
        auto dc_offsets = test_memory(dc_offsets_desc, eng);
        auto dc_weights = test_memory(dc_weights_desc, eng);
        auto dc_bias = test_memory(dc_bias_desc, eng);
        auto dc_dst = test_memory(dc_dst_desc, eng);

        std::vector<data_t_dst> ref_dst_data(dc_dst.get_size());

        fill_data<data_t_dst>(dc_dst.get_size() / sizeof(data_t_dst),
                              (data_t_dst *)dc_dst.get().get_data_handle());
        fill_data<data_t_src>(dc_src.get_size() / sizeof(data_t_src),
                              (data_t_src *)dc_src.get().get_data_handle(), (data_t_off)0, (data_t_off)impl::nstl::min(dcs.ih, dcs.iw));
        fill_data<data_t_off>(dc_offsets.get_size() / sizeof(data_t_wei),
                              (data_t_wei *)dc_offsets.get().get_data_handle());
        fill_data<data_t_wei>(dc_weights.get_size() / sizeof(data_t_wei),
                              (data_t_wei *)dc_weights.get().get_data_handle());
        if (with_bias) {
            fill_data<data_t_dst>(dc_bias.get_size() / sizeof(data_t_dst),
                                  (data_t_dst *)dc_bias.get().get_data_handle());
        }
        check_zero_tail<data_t_src>(1, dc_src.get());
        check_zero_tail<data_t_off>(1, dc_offsets.get());
        check_zero_tail<data_t_wei>(1, dc_weights.get());
        check_zero_tail<data_t_dst>(1, dc_dst.get());

        std::vector<ptrdiff_t> padR = {
                right_padding(dcs.ih, dcs.oh, dcs.kh, dcs.padh, dcs.strh, dcs.dilh),
                right_padding(dcs.iw, dcs.ow, dcs.kw, dcs.padw, dcs.strw, dcs.dilw)
        };

        std::vector<memory::desc> dc_src_descs;
        dc_src_descs.push_back(dc_src_desc);
        dc_src_descs.push_back(dc_offsets_desc);

        auto def_conv_desc = with_bias ? deformable_convolution_forward::desc(aprop_kind, p.aalgorithm,
                                                                              dc_src_descs, dc_weights_desc, dc_bias_desc, dc_dst_desc,
                                                                              { dcs.strh, dcs.strw }, { dcs.dilh, dcs.dilw },
                                                                              { dcs.padh, dcs.padw }, padR, padding_kind::zero, dcs.dg)
                                       : deformable_convolution_forward::desc(aprop_kind, p.aalgorithm,
                                                                              dc_src_descs, dc_weights_desc, dc_dst_desc,
                                                                              { dcs.strh, dcs.strw }, { dcs.dilh, dcs.dilw },
                                                                              { dcs.padh, dcs.padw }, padR, padding_kind::zero, dcs.dg);

        auto def_conv_primitive_desc = deformable_convolution_forward::primitive_desc(def_conv_desc, eng);

        std::vector<primitive::at> dc_srcs;
        dc_srcs.emplace_back(dc_src.get());
        dc_srcs.emplace_back(dc_offsets.get());

        auto conv = with_bias ? deformable_convolution_forward(def_conv_primitive_desc, dc_srcs, dc_weights.get(), dc_bias.get(), dc_dst.get())
                              : deformable_convolution_forward(def_conv_primitive_desc, dc_srcs, dc_weights.get(), dc_dst.get());

        std::vector<primitive> pipeline;
        pipeline.push_back(conv);
        auto s = stream(stream::kind::lazy);
        s.submit(pipeline).wait();

        auto ref_memory = memory(memory::primitive_desc(dc_dst_desc, eng), &ref_dst_data[0]);
        compute_ref_def_conv_fwd(dcs, dc_src_desc, dc_offsets_desc, dc_weights_desc, dc_bias_desc, dc_dst_desc,
                                 dc_src.get(), dc_offsets.get(), dc_weights.get(), dc_bias.get(), ref_memory);
        check_zero_tail<data_t_dst>(1, ref_memory);

        compare_data<data_t_dst>(ref_memory, dc_dst.get(), 1e-2f);
        check_zero_tail<data_t_dst>(0, dc_dst.get());
    }
};

}
#endif
