// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>
#include "mkldnn_graph.h"

#include "test_graph.hpp"

#include "single_layer_common.hpp"
#include <mkldnn_extension_utils.h>
#include "tests_common.hpp"
#include <cpp/ie_cnn_net_reader.h>
#include <ie_plugin_config.hpp>

using namespace ::testing;
using namespace std;
using namespace mkldnn;

struct conv_params {
    size_t krn_w;
    size_t krn_h;
    size_t str_w;
    size_t str_h;
    size_t pad_w;
    size_t pad_h;
    size_t out_c;
    size_t grp_c;
};

struct dw_conv_fusing_test_params {
    struct {
        size_t n;
        size_t c;
        size_t h;
        size_t w;
    } in;

    conv_params conv1;
    conv_params conv2;
};

template <typename data_t>
void ref_conv(const InferenceEngine::TBlob<data_t> &src, const data_t *weights, const size_t weightsSize,
              InferenceEngine::TBlob<data_t> &dst, conv_params prm, float negative_slope) {
    size_t KW = prm.krn_w;
    size_t KH = prm.krn_h;
    size_t GC = prm.grp_c;

    size_t IC = src.getTensorDesc().getDims()[1];
    size_t IH = src.getTensorDesc().getDims()[2];
    size_t IW = src.getTensorDesc().getDims()[3];

    size_t OW = (IW + 2 * prm.pad_w - prm.krn_w) / prm.str_w + 1;
    size_t OH = (IH + 2 * prm.pad_h - prm.krn_h) / prm.str_h + 1;
    size_t OC = prm.out_c;

    const data_t *src_data = src.readOnly();
    const data_t *weights_data = weights;
    const data_t *bias_data = weights_data + KW * KH * OC * IC / GC;
    data_t *dst_data = dst.data();

    IE_ASSERT(KW * KH * OC * IC / GC + OC == weightsSize);

    for (uint32_t g = 0; g < GC; g++) {
        for (uint32_t oc = 0; oc < OC / GC; oc++) {
            for (uint32_t oh = 0; oh < OH; oh++) {
                for (uint32_t ow = 0; ow < OW; ow++) {
                    size_t oidx = g * OC / GC * OH * OW
                                  + oc * OH * OW + oh * OW + ow;
                    dst_data[oidx] = bias_data[g * OC / GC + oc];

                    for (size_t ic = 0; ic < IC / GC; ic++) {
                        for (size_t kh = 0; kh < KH; kh++) {
                            for (size_t kw = 0; kw < KW; kw++) {
                                int32_t iw = ow * prm.str_w - prm.pad_w + kw;
                                int32_t ih = oh * prm.str_h - prm.pad_h + kh;
                                if (iw < 0 || iw >= (int32_t)IW || ih < 0
                                    || ih >= (int32_t)IH)
                                    continue;
                                size_t iidx = g * IC / GC * IH * IW
                                              + ic * IH * IW + ih * IW + iw;
                                size_t widx = g * OC / GC * IC / GC * KH * KW
                                              + oc * IC / GC * KH * KW
                                              + ic * KH * KW + kh * KW + kw;

                                dst_data[oidx] += src_data[iidx] * weights_data[widx];
                            }
                        }
                    }

                    if (dst_data[oidx] < 0)
                        dst_data[oidx] *= negative_slope;
                }
            }
        }
    }
}

class MKLDNNGraphDWConvFusingTests: public TestsCommon,
                                    public WithParamInterface<dw_conv_fusing_test_params> {
    std::string model_t = R"V0G0N(
<Net Name="Convolution_Only" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </output>
        </layer>
        <layer name="conv1" id="1" type="Convolution" precision="FP32">
            <convolution stride-x="_C1_SW_" stride-y="_C1_SH_"
                         pad-x="_C1_PW_"    pad-y="_C1_PH_"
                         kernel-x="_C1_KW_" kernel-y="_C1_KH_"
                         output="_C1_OC_"   group="_C1_GC_"/>

            <weights offset="0" size="_C1_S1_" />
            <biases offset="_C1_S1_" size="_C1_S2_" />
            <input>
                <port id="1">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>_IN_</dim>
                    <dim>_C1_OC_</dim>
                    <dim>_C1_OH_</dim>
                    <dim>_C1_OW_</dim>
                </port>
            </output>
        </layer>
        <layer name="relu1" id="2" type="ReLU" precision="FP32">
            <data negative_slope="0"/>
            <input>
                <port id="3">
                    <dim>_IN_</dim>
                    <dim>_C1_OC_</dim>
                    <dim>_C1_OH_</dim>
                    <dim>_C1_OW_</dim>
                </port>
            </input>
            <output>
                <port id="4">
                    <dim>_IN_</dim>
                    <dim>_C1_OC_</dim>
                    <dim>_C1_OH_</dim>
                    <dim>_C1_OW_</dim>
                </port>
            </output>
        </layer>
        <layer name="conv2" id="3" type="Convolution" precision="FP32">
            <convolution stride-x="_C2_SW_" stride-y="_C2_SH_"
                         pad-x="_C2_PW_"    pad-y="_C2_PH_"
                         kernel-x="_C2_KW_" kernel-y="_C2_KH_"
                         output="_C2_OC_"   group="_C2_GC_"/>

            <weights offset="_C2_S0_" size="_C2_S1_" />
            <biases offset="_C2_S2_" size="_C2_S3_" />
            <input>
                <port id="5">
                    <dim>_IN_</dim>
                    <dim>_C1_OC_</dim>
                    <dim>_C1_OH_</dim>
                    <dim>_C1_OW_</dim>
                </port>
            </input>
            <output>
                <port id="6">
                    <dim>_IN_</dim>
                    <dim>_C2_OC_</dim>
                    <dim>_C2_OH_</dim>
                    <dim>_C2_OW_</dim>
                </port>
            </output>
        </layer>
        <layer name="relu2" id="4" type="ReLU" precision="FP32">
            <data negative_slope="0"/>
            <input>
                <port id="7">
                    <dim>_IN_</dim>
                    <dim>_C2_OC_</dim>
                    <dim>_C2_OH_</dim>
                    <dim>_C2_OW_</dim>
                </port>
            </input>
            <output>
                <port id="8">
                    <dim>_IN_</dim>
                    <dim>_C2_OC_</dim>
                    <dim>_C2_OH_</dim>
                    <dim>_C2_OW_</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="3"/>
        <edge from-layer="2" from-port="4" to-layer="3" to-port="5"/>
        <edge from-layer="3" from-port="6" to-layer="4" to-port="7"/>
    </edges>
</Net>
)V0G0N";

    std::string getModel(dw_conv_fusing_test_params p) {
        std::string model = model_t;
        REPLACE_WITH_NUM(model, "_IW_", p.in.w);
        REPLACE_WITH_NUM(model, "_IH_", p.in.h);
        REPLACE_WITH_NUM(model, "_IC_", p.in.c);
        REPLACE_WITH_NUM(model, "_IN_", p.in.n);

        REPLACE_WITH_NUM(model, "_C1_KW_", p.conv1.krn_w);
        REPLACE_WITH_NUM(model, "_C1_KH_", p.conv1.krn_h);
        REPLACE_WITH_NUM(model, "_C1_SW_", p.conv1.str_w);
        REPLACE_WITH_NUM(model, "_C1_SH_", p.conv1.str_h);
        REPLACE_WITH_NUM(model, "_C1_PW_", p.conv1.pad_w);
        REPLACE_WITH_NUM(model, "_C1_PH_", p.conv1.pad_h);
        REPLACE_WITH_NUM(model, "_C1_GC_", p.conv1.grp_c);
        REPLACE_WITH_NUM(model, "_C1_OC_", p.conv1.out_c);
        size_t c1_oh = (p.in.h + 2 * p.conv1.pad_h - p.conv1.krn_h) / p.conv1.str_h + 1;
        size_t c1_ow = (p.in.w + 2 * p.conv1.pad_w - p.conv1.krn_w) / p.conv1.str_w + 1;
        REPLACE_WITH_NUM(model, "_C1_OH_", c1_oh);
        REPLACE_WITH_NUM(model, "_C1_OW_", c1_ow);

        size_t conv1_w_data_size = (p.conv1.krn_w * p.conv1.krn_h * p.conv1.out_c * p.in.c / p.conv1.grp_c) * sizeof(float);
        size_t conv1_b_data_size = p.conv1.out_c * sizeof(float);
        REPLACE_WITH_NUM(model, "_C1_S1_", conv1_w_data_size);
        REPLACE_WITH_NUM(model, "_C1_S2_", conv1_b_data_size);

        REPLACE_WITH_NUM(model, "_C2_KW_", p.conv2.krn_w);
        REPLACE_WITH_NUM(model, "_C2_KH_", p.conv2.krn_h);
        REPLACE_WITH_NUM(model, "_C2_SW_", p.conv2.str_w);
        REPLACE_WITH_NUM(model, "_C2_SH_", p.conv2.str_h);
        REPLACE_WITH_NUM(model, "_C2_PW_", p.conv2.pad_w);
        REPLACE_WITH_NUM(model, "_C2_PH_", p.conv2.pad_h);
        REPLACE_WITH_NUM(model, "_C2_GC_", p.conv2.grp_c);
        REPLACE_WITH_NUM(model, "_C2_OC_", p.conv2.out_c);
        REPLACE_WITH_NUM(model, "_C2_OH_", (c1_oh + 2 * p.conv2.pad_h - p.conv2.krn_h) / p.conv2.str_h + 1);
        REPLACE_WITH_NUM(model, "_C2_OW_", (c1_ow + 2 * p.conv2.pad_w - p.conv2.krn_w) / p.conv2.str_w + 1);

        size_t conv2_w_data_size = (p.conv2.krn_w * p.conv2.krn_h * p.conv2.out_c * p.conv1.out_c / p.conv2.grp_c) * sizeof(float);
        size_t conv2_b_data_size = p.conv2.out_c * sizeof(float);
        REPLACE_WITH_NUM(model, "_C2_S0_", conv1_w_data_size + conv1_b_data_size);
        REPLACE_WITH_NUM(model, "_C2_S1_", conv2_w_data_size);
        REPLACE_WITH_NUM(model, "_C2_S2_", conv1_w_data_size + conv1_b_data_size + conv2_w_data_size);
        REPLACE_WITH_NUM(model, "_C2_S3_", conv2_b_data_size);
        return model;
    }

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            dw_conv_fusing_test_params p = ::testing::WithParamInterface<dw_conv_fusing_test_params>::GetParam();
            std::string model = getModel(p);

            InferenceEngine::CNNNetReader net_reader;
            ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

            size_t conv1_w_size = p.conv1.krn_w * p.conv1.krn_h * p.conv1.out_c * p.in.c / p.conv1.grp_c + p.conv1.out_c; // conv1 weights + biases
            size_t conv2_w_size = p.conv2.krn_w * p.conv2.krn_h * p.conv2.out_c * p.conv1.out_c / p.conv2.grp_c + p.conv2.out_c; // conv2 weights + biases

            InferenceEngine::TBlob<uint8_t> *weights = new InferenceEngine::TBlob<uint8_t>({ InferenceEngine::Precision::U8, 
                {(conv1_w_size+conv2_w_size) * sizeof(float)}, InferenceEngine::C });
            weights->allocate();
            fill_data((float *) weights->buffer(), weights->size() / sizeof(float), 1);
            InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr = InferenceEngine::TBlob<uint8_t>::Ptr(weights);

            net_reader.SetWeights(weights_ptr);

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(net_reader.getNetwork());

            InferenceEngine::SizeVector dims_src = {p.in.n, p.in.c, p.in.h, p.in.w};

            InferenceEngine::Blob::Ptr src = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, dims_src, InferenceEngine::NCHW});
            src->allocate();
            fill_data(src->buffer(), src->size());

            auto * srcPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(src.get());

            if (srcPtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";

            InferenceEngine::BlobMap srcs;
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in1", src));

            InferenceEngine::OutputsDataMap out;
            out = net_reader.getNetwork().getOutputsInfo();
            InferenceEngine::BlobMap outputBlobs;

            std::pair<std::string, InferenceEngine::DataPtr> item = *out.begin();

            InferenceEngine::TBlob<float>::Ptr output;
            output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
            output->allocate();
            outputBlobs[item.first] = output;

            graph.Infer(srcs, outputBlobs);

            size_t c1_oh = (p.in.h + 2 * p.conv1.pad_h - p.conv1.krn_h) / p.conv1.str_h + 1;
            size_t c1_ow = (p.in.w + 2 * p.conv1.pad_w - p.conv1.krn_w) / p.conv1.str_w + 1;
            InferenceEngine::TBlob<float> conv1_dst_ref(InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32, {p.in.n, p.conv1.out_c, c1_oh, c1_ow}, InferenceEngine::NCHW));
            conv1_dst_ref.allocate();

            size_t c2_oh = (c1_oh + 2 * p.conv2.pad_h - p.conv2.krn_h) / p.conv2.str_h + 1;
            size_t c2_ow = (c1_ow + 2 * p.conv2.pad_w - p.conv2.krn_w) / p.conv2.str_w + 1;
            InferenceEngine::TBlob<float> conv2_dst_ref(InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32, {p.in.n, p.conv2.out_c, c2_oh, c2_ow}, InferenceEngine::NCHW));
            conv2_dst_ref.allocate();

            ref_conv(*srcPtr, (const float *)weights->buffer(), conv1_w_size, conv1_dst_ref, p.conv1, 0.0f);
            ref_conv(conv1_dst_ref, (const float *)weights->buffer() + conv1_w_size, conv2_w_size, conv2_dst_ref, p.conv2, 0.0f);


            compare(*output, conv2_dst_ref);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNGraphDWConvFusingTests, TestsDwConvFusing) {}

INSTANTIATE_TEST_CASE_P(
        TestsDwConvFusing, MKLDNNGraphDWConvFusingTests,
        ::testing::Values(
                dw_conv_fusing_test_params{{1, 32, 160, 320}, {1, 1, 1, 1, 0, 0, 24, 1}, {3, 3, 1, 1, 1, 1, 24, 24}}
        ));
