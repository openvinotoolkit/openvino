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

#include "mkldnn_test_common.hpp"
#include "gtest/gtest.h"

#include "mkldnn.hpp"

namespace mkldnn {

template <typename data_t_src, typename data_t_wei,
          typename data_t_acc, typename data_t_dst>
void compute_ref_conv_fwd(const mkldnn_convolution_desc_t &conv_desc,
        const memory &src, const memory &weights, const memory &bias, const memory &dst,
        bool with_relu, float eltwise_alpha)
{
    int MB = conv_desc.src_desc.dims[0];
    int G = conv_desc.weights_desc.ndims == 5 ? conv_desc.weights_desc.dims[0] : 1;
    int IC = conv_desc.src_desc.dims[1];
    int IH = conv_desc.src_desc.dims[2];
    int IW = conv_desc.src_desc.dims[3];
    int OC = conv_desc.dst_desc.dims[1];
    int OH = conv_desc.dst_desc.dims[2];
    int OW = conv_desc.dst_desc.dims[3];

    int KH = G > 1 ? conv_desc.weights_desc.dims[3] : conv_desc.weights_desc.dims[2];
    int KW = G > 1 ? conv_desc.weights_desc.dims[4] : conv_desc.weights_desc.dims[3];
    int SH = conv_desc.strides[0];
    int SW = conv_desc.strides[1];
    int PH = conv_desc.padding[0][0];
    int PW = conv_desc.padding[1][0];
    int DH = conv_desc.dilates[0];
    int DW = conv_desc.dilates[1];

    data_t_src *src_data = (data_t_src *)src.get_data_handle();
    data_t_wei *weights_data = (data_t_wei *)weights.get_data_handle();
    data_t_dst *bias_data = (data_t_dst *)bias.get_data_handle();
    data_t_dst *dst_data = (data_t_dst *)dst.get_data_handle();

    const memory::desc src_d = src.get_primitive_desc().desc();
    const memory::desc weights_d = weights.get_primitive_desc().desc();
//    const memory::desc bias_d = bias.get_primitive_desc().desc();
    const memory::desc dst_d = dst.get_primitive_desc().desc();

//    #pragma omp parallel for collapse(5) schedule(static)
    for (int n = 0; n < MB; n++) {
        for (int g = 0; g < G; g++) {
            for (int oc = 0; oc < OC / G; oc++) {
                for (int oh = 0; oh < OH; oh++) {
                    for (int ow = 0; ow < OW; ow++) {
                        int oidx = n * OC * OH * OW
                                + g * OC / G * OH * OW
                                + oc * OH * OW + oh * OW + ow;

                        data_t_acc a = (data_t_acc)0;

                        for (int ic = 0; ic < IC / G; ic++) {
                            for (int kh = 0; kh < KH; kh++) {
                                for (int kw = 0; kw < KW; kw++) {
                                    int iw = ow * SW
                                          - PW + kw * (1 + DW);
                                    int ih = oh * SH
                                          - PH + kh * (1 + DH);
                                    if (iw < 0 || iw >= IW) continue;
                                    if (ih < 0 || ih >= IH) continue;
                                    int iidx = n * IC * IH * IW
                                            + g * IC / G * IH * IW
                                            + ic * IH * IW + ih * IW + iw;
                                    int widx = g * OC / G * IC
                                                    / G * KH * KW
                                            + oc * IC / G * KH * KW
                                            + ic * KH * KW + kh * KW  + kw;

                                    a += src_data[map_index(src_d, iidx)]
                                            * weights_data[map_index(
                                                      weights_d, widx)];


                                }
                            }
                        }

                        float a_fp = (float)a;

//                        a_fp *= scales[G > 1 ? g : oc];
                        a_fp += bias_data[G > 1 ? g : oc];

                        if (with_relu) {
                            a_fp = (a_fp > 0) ? a_fp : eltwise_alpha * a_fp;
                        }

                        dst_data[map_index(dst_d, oidx)] = (data_t_dst)a_fp;
                    }
                }
            }
        }
    }
}

template <typename data_t_src, typename data_t_wei,
          typename data_t_acc, typename data_t_dst>
class convolution_dw_conv_test
    : public ::testing::TestWithParam<test_convolution_dw_conv_params_t> {
protected:
    void SetUp() {
        auto p = ::testing::TestWithParam<test_convolution_dw_conv_params_t>::GetParam();
        ASSERT_TRUE(p.engine_kind == engine::kind::cpu);
        ASSERT_EQ(p.aalgorithm, convolution_direct);
        auto eng = engine(p.engine_kind, 0);

        memory::data_type data_type_src = data_traits<data_t_src>::data_type;
        memory::data_type data_type_dst = data_traits<data_t_dst>::data_type;
        memory::data_type data_type_wei = data_traits<data_t_wei>::data_type;
        memory::data_type data_type_bia = data_traits<data_t_wei>::data_type;

        test_convolution_dw_conv_sizes_t cd = p.sizes;

        int conv1_oh = (cd.ih - ((cd.conv1_kh - 1) + 1) + 2 * cd.conv1_padh) / cd.conv1_strh + 1;
        int conv1_ow = (cd.iw - ((cd.conv1_kw - 1) + 1) + 2 * cd.conv1_padw) / cd.conv1_strw + 1;

        int conv2_oh = (conv1_oh - ((cd.conv2_kh - 1) + 1) + 2 * cd.conv2_padh) / cd.conv2_strh + 1;
        int conv2_ow = (conv1_ow - ((cd.conv2_kw - 1) + 1) + 2 * cd.conv2_padw) / cd.conv2_strw + 1;

        std::vector<int> conv1_padR = { cd.conv1_padh, cd.conv1_padw };
        conv1_padR[0] += conv2_oh - conv1_oh;
        conv1_padR[1] += conv2_ow - conv1_ow;

        test_convolution_dw_conv_formats_t f = p.formats;

        auto conv1_src_desc = create_md({ cd.mb, cd.ic, cd.ih, cd.iw }, data_type_src, f.src_format);
        auto conv1_weights_desc = create_md({ cd.conv1_oc, cd.ic, cd.conv1_kh, cd.conv1_kw }, data_type_wei, f.conv1_weights_format);
        auto conv1_bias_desc = create_md({ cd.conv1_oc }, data_type_bia, f.conv1_bias_format);
        auto conv1_dst_desc = create_md({ cd.mb, cd.conv1_oc, conv2_oh, conv2_ow }, data_type_dst, f.dst_format);

        auto conv1_desc = convolution_forward::desc(prop_kind::forward_scoring, p.aalgorithm,
                conv1_src_desc, conv1_weights_desc, conv1_bias_desc, conv1_dst_desc,
                { cd.conv1_strh, cd.conv1_strw }, { 0, 0 },
                { cd.conv1_padh, cd.conv1_padw }, conv1_padR, padding_kind::zero);

        auto conv2_src_desc = create_md({ cd.mb, cd.conv1_oc, conv1_oh, conv1_ow }, data_type_src, f.src_format);
        auto conv2_weights_desc = create_md({ cd.conv2_oc, 1, 1, cd.conv2_kh, cd.conv2_kw }, data_type_wei, f.conv2_weights_format);
        auto conv2_bias_desc = create_md({ cd.conv2_oc }, data_type_bia, f.conv2_bias_format);
        auto conv2_dst_desc = create_md({ cd.mb, cd.conv2_oc, conv2_oh, conv2_ow }, data_type_dst, f.dst_format);

        auto conv2_desc = convolution_forward::desc(prop_kind::forward_scoring,
                p.aalgorithm, conv2_src_desc, conv2_weights_desc, conv2_bias_desc, conv2_dst_desc,
                { cd.conv2_strh, cd.conv2_strw }, { 0, 0 },
                { cd.conv2_padh, cd.conv2_padw }, { cd.conv2_padh, cd.conv2_padw }, padding_kind::zero);

        auto conv1_src = memory({conv1_src_desc, eng});
        auto conv1_weights = memory({conv1_weights_desc, eng});
        auto conv1_bias = memory({conv1_bias_desc, eng});
        auto conv2_weights = memory({conv2_weights_desc, eng});
        auto conv2_bias = memory({conv2_bias_desc, eng});
        auto conv2_dst = memory({conv2_dst_desc, eng});

        fill_data<data_t_src>(conv1_src.get_primitive_desc().get_size()
                / sizeof(data_t_src), (data_t_src *)conv1_src.get_data_handle(), 1., true);
        fill_data<data_t_wei>(
                conv1_weights.get_primitive_desc().get_size()
                / sizeof(data_t_wei),(data_t_wei *)conv1_weights.get_data_handle(), 1., true);
        fill_data<data_t_wei>(
                conv1_bias.get_primitive_desc().get_size()
                / sizeof(data_t_wei),(data_t_wei *)conv1_bias.get_data_handle(), 1., true);
        fill_data<data_t_wei>(
                conv2_weights.get_primitive_desc().get_size()
                / sizeof(data_t_wei),(data_t_wei *)conv2_weights.get_data_handle(), 1., true);
        fill_data<data_t_wei>(
                conv2_bias.get_primitive_desc().get_size()
                / sizeof(data_t_wei),(data_t_wei *)conv2_bias.get_data_handle(), 1., true);

        mkldnn::post_ops conv1_post_ops;
        conv1_post_ops.append_eltwise(1.0, mkldnn::algorithm::eltwise_relu, 0.0f, 0.0f);
        conv1_post_ops.append_dw_conv(conv1_oh, conv1_ow, cd.conv2_kh, cd.conv2_kw, cd.conv2_strh, cd.conv2_strw,
                                      static_cast<const float*>(conv2_weights.get_data_handle()),
                                      static_cast<const float*>(conv2_bias.get_data_handle()));
        conv1_post_ops.append_eltwise(1.0, mkldnn::algorithm::eltwise_relu, 0.0f, 0.0f);
        mkldnn::primitive_attr conv1_attr;
        conv1_attr.set_post_ops(conv1_post_ops);

        auto conv1_primitive_desc = convolution_forward::primitive_desc(conv1_desc, conv1_attr, eng);
        auto conv1 = convolution_forward(conv1_primitive_desc, conv1_src, conv1_weights, conv1_bias, conv2_dst);

        std::vector<primitive> pipeline;
        pipeline.push_back(conv1);
        stream(stream::kind::lazy).submit(pipeline).wait();

        auto conv1_dst_desc_ref = create_md({ cd.mb, cd.conv1_oc, conv1_oh, conv1_ow }, data_type_dst, f.dst_format);
        auto conv1_desc_ref = convolution_forward::desc(prop_kind::forward_scoring, p.aalgorithm,
                conv1_src_desc, conv1_weights_desc, conv1_bias_desc, conv1_dst_desc_ref,
                { cd.conv1_strh, cd.conv1_strw }, { 0, 0 },
                { cd.conv1_padh, cd.conv1_padw }, { cd.conv1_padh, cd.conv1_padw }, padding_kind::zero);

        auto conv1_dst_ref = memory({conv1_dst_desc_ref, eng});
        auto conv2_dst_ref = memory({conv2_dst_desc, eng});
        compute_ref_conv_fwd<data_t_src, data_t_wei, data_t_acc, data_t_dst>(conv1_desc_ref.data, conv1_src, conv1_weights, conv1_bias, conv1_dst_ref, true, 0.0f);
        compute_ref_conv_fwd<data_t_dst, data_t_wei, data_t_acc, data_t_dst>(conv2_desc.data, conv1_dst_ref, conv2_weights, conv2_bias, conv2_dst_ref, true, 0.0f);

        compare_data<data_t_dst>(conv2_dst_ref, conv2_dst);
    }
};

}
