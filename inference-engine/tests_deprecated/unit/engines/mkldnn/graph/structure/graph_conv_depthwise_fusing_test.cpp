// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_graph.hpp"

#include "common_test_utils/data_utils.hpp"
#include "single_layer_common.hpp"
#include "tests_common.hpp"
#include <ie_core.hpp>

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

struct conv_depthwise_fusing_test_params {
    struct {
        size_t n;
        size_t c;
        size_t h;
        size_t w;
    } in;

    conv_params conv;
    algorithm depthwise_alg;
    bool isBroadcast;
};

template <typename data_t>
void ref_conv_depthwise(const InferenceEngine::TBlob<data_t> &src, const data_t *weights,
              InferenceEngine::TBlob<data_t> &dst, conv_depthwise_fusing_test_params& prm) {
    size_t KW = prm.conv.krn_w;
    size_t KH = prm.conv.krn_h;
    size_t GC = prm.conv.grp_c;

    size_t IC = src.getTensorDesc().getDims()[1];
    size_t IH = src.getTensorDesc().getDims()[2];
    size_t IW = src.getTensorDesc().getDims()[3];

    size_t OW = (IW + 2 * prm.conv.pad_w - prm.conv.krn_w) / prm.conv.str_w + 1;
    size_t OH = (IH + 2 * prm.conv.pad_h - prm.conv.krn_h) / prm.conv.str_h + 1;
    size_t OC = prm.conv.out_c;

    const data_t *src_data = src.readOnly();
    const data_t *weights_data = weights;
    const data_t *bias_data = weights_data + KW * KH * OC * IC / GC;
    data_t *dst_data = dst.data();

    const data_t *d_weights_data = bias_data + OC;
    const data_t *d_bias_data = (prm.isBroadcast) ? d_weights_data + 1 : d_weights_data + OC;

    for (uint32_t g = 0; g < GC; g++) {
        for (uint32_t oc = 0; oc < OC / GC; oc++) {
            for (uint32_t oh = 0; oh < OH; oh++) {
                for (uint32_t ow = 0; ow < OW; ow++) {
                    size_t bidx = g * OC / GC + oc;
                    size_t oidx = g * OC / GC * OH * OW
                                  + oc * OH * OW + oh * OW + ow;
                    dst_data[oidx] = bias_data[bidx];

                    for (size_t ic = 0; ic < IC / GC; ic++) {
                        for (size_t kh = 0; kh < KH; kh++) {
                            for (size_t kw = 0; kw < KW; kw++) {
                                int32_t iw = ow * prm.conv.str_w - prm.conv.pad_w + kw;
                                int32_t ih = oh * prm.conv.str_h - prm.conv.pad_h + kh;
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


                    switch(prm.depthwise_alg) {
                        case depthwise_scale_shift:
                            dst_data[oidx] = d_weights_data[prm.isBroadcast ? 0 : bidx] * dst_data[oidx] + d_bias_data[prm.isBroadcast ? 0 : bidx];
                            break;
                        case depthwise_prelu:
                            dst_data[oidx] = dst_data[oidx] >= 0 ? dst_data[oidx] : d_weights_data[prm.isBroadcast ? 0 : bidx] * dst_data[oidx];
                            break;
                        default:
                            assert("Unsupported depthwise algorithm");
                    }
                }
            }
        }
    }
}

class MKLDNNGraphConvDepthwiseFusingTests: public TestsCommon,
                                    public WithParamInterface<conv_depthwise_fusing_test_params> {
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
        <layer name="conv" id="1" type="Convolution" precision="FP32">
            <convolution stride-x="_C_SW_" stride-y="_C_SH_"
                         pad-x="_C_PW_"    pad-y="_C_PH_"
                         kernel-x="_C_KW_" kernel-y="_C_KH_"
                         output="_C_OC_"   group="_C_GC_"/>

            <weights offset="0" size="_C_S1_" />
            <biases offset="_C_S1_" size="_C_S2_" />
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
                    <dim>_C_OC_</dim>
                    <dim>_C_OH_</dim>
                    <dim>_C_OW_</dim>
                </port>
            </output>
        </layer>
        <layer name="depthwise" id="2" type="_LT_" precision="FP32">
            <data _P_NAME_="_P_VAL_"  PrimitivesPriority="_IMPLS_"/>
            <weights offset="_D_S0_" size="_D_S1_" />
            <biases offset="_D_S2_" size="_D_S3_" />

            <input>
                <port id="3">
                    <dim>_IN_</dim>
                    <dim>_C_OC_</dim>
                    <dim>_C_OH_</dim>
                    <dim>_C_OW_</dim>
                </port>
            </input>
            <output>
                <port id="4">
                    <dim>_IN_</dim>
                    <dim>_C_OC_</dim>
                    <dim>_C_OH_</dim>
                    <dim>_C_OW_</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="3"/>
    </edges>
</Net>
)V0G0N";

    std::string getModel(conv_depthwise_fusing_test_params p) {
        std::string model = model_t;
        REPLACE_WITH_NUM(model, "_IW_", p.in.w);
        REPLACE_WITH_NUM(model, "_IH_", p.in.h);
        REPLACE_WITH_NUM(model, "_IC_", p.in.c);
        REPLACE_WITH_NUM(model, "_IN_", p.in.n);

        REPLACE_WITH_NUM(model, "_C_KW_", p.conv.krn_w);
        REPLACE_WITH_NUM(model, "_C_KH_", p.conv.krn_h);
        REPLACE_WITH_NUM(model, "_C_SW_", p.conv.str_w);
        REPLACE_WITH_NUM(model, "_C_SH_", p.conv.str_h);
        REPLACE_WITH_NUM(model, "_C_PW_", p.conv.pad_w);
        REPLACE_WITH_NUM(model, "_C_PH_", p.conv.pad_h);
        REPLACE_WITH_NUM(model, "_C_GC_", p.conv.grp_c);
        REPLACE_WITH_NUM(model, "_C_OC_", p.conv.out_c);
        size_t c_oh = (p.in.h + 2 * p.conv.pad_h - p.conv.krn_h) / p.conv.str_h + 1;
        size_t c_ow = (p.in.w + 2 * p.conv.pad_w - p.conv.krn_w) / p.conv.str_w + 1;
        REPLACE_WITH_NUM(model, "_C_OH_", c_oh);
        REPLACE_WITH_NUM(model, "_C_OW_", c_ow);

        size_t conv_w_data_size = (p.conv.krn_w * p.conv.krn_h * p.conv.out_c * p.in.c / p.conv.grp_c) * sizeof(float);
        size_t conv_b_data_size = p.conv.out_c * sizeof(float);
        REPLACE_WITH_NUM(model, "_C_S1_", conv_w_data_size);
        REPLACE_WITH_NUM(model, "_C_S2_", conv_b_data_size);

        if (p.depthwise_alg == depthwise_scale_shift) {
            REPLACE_WITH_STR(model, "_LT_", "ScaleShift");
            REPLACE_WITH_STR(model, "_P_NAME_", "broadcast");
            REPLACE_WITH_NUM(model, "_P_VAL_", p.isBroadcast ? 1 : 0);

        }
        else if (p.depthwise_alg == depthwise_prelu) {
            REPLACE_WITH_STR(model, "_LT_", "PReLU");
            REPLACE_WITH_STR(model, "_P_NAME_", "channel_shared");
            REPLACE_WITH_NUM(model, "_P_VAL_", p.isBroadcast ? 1 : 0);
        }

        size_t array_size =  p.isBroadcast ? 1 : p.conv.out_c;
        size_t depthwise_w_data_size = array_size * sizeof(float);
        size_t depthwise_b_data_size = array_size * sizeof(float);
        REPLACE_WITH_NUM(model, "_D_S0_", conv_w_data_size + conv_b_data_size);
        REPLACE_WITH_NUM(model, "_D_S1_", depthwise_w_data_size);
        REPLACE_WITH_NUM(model, "_D_S2_", conv_w_data_size + conv_b_data_size + depthwise_w_data_size);
        REPLACE_WITH_NUM(model, "_D_S3_", depthwise_b_data_size);

        return model;
    }

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            conv_depthwise_fusing_test_params p = ::testing::WithParamInterface<conv_depthwise_fusing_test_params>::GetParam();
            std::string model = getModel(p);

            size_t conv_w_size = p.conv.krn_w * p.conv.krn_h * p.conv.out_c * p.in.c / p.conv.grp_c + p.conv.out_c; // conv weights + biases

            size_t array_size =  p.isBroadcast ? 1 : p.conv.out_c;
            size_t depthwise_w_size = array_size + array_size; // depthwise weights + biases

            InferenceEngine::TBlob<uint8_t> *weights = new InferenceEngine::TBlob<uint8_t>({ InferenceEngine::Precision::U8, 
                {(conv_w_size+depthwise_w_size) * sizeof(float)}, InferenceEngine::C });
            weights->allocate();
            CommonTestUtils::fill_data_sine((float *) weights->buffer(), weights->size() / sizeof(float), 5, 10, 0.5);
            InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr = InferenceEngine::TBlob<uint8_t>::Ptr(weights);

            InferenceEngine::Core core;
            InferenceEngine::CNNNetwork network;
            ASSERT_NO_THROW(network = core.ReadNetwork(model, weights_ptr));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(network);

            auto& nodes = graph.getNodes();
            nodes = graph.getNodes();
            if (p.in.c == 3) {
                ASSERT_EQ(nodes.size(), 3);
                ASSERT_EQ(nodes[0].get()->getType(), MKLDNNPlugin::Type::Input);
                ASSERT_EQ(nodes[1].get()->getType(), MKLDNNPlugin::Type::Convolution);
                ASSERT_TRUE(nodes[1].get()->isFusedWith(MKLDNNPlugin::Type::Eltwise));
                ASSERT_EQ(nodes[2].get()->getType(), MKLDNNPlugin::Type::Output);
            } else {
                ASSERT_EQ(nodes.size(), 5);
                ASSERT_EQ(nodes[0].get()->getType(), MKLDNNPlugin::Type::Input);
                ASSERT_EQ(nodes[1].get()->getType(), MKLDNNPlugin::Type::Reorder);
                ASSERT_EQ(nodes[2].get()->getType(), MKLDNNPlugin::Type::Convolution);
                ASSERT_TRUE(nodes[2].get()->isFusedWith(MKLDNNPlugin::Type::Eltwise));
                ASSERT_EQ(nodes[3].get()->getType(), MKLDNNPlugin::Type::Reorder);
                ASSERT_EQ(nodes[4].get()->getType(), MKLDNNPlugin::Type::Output);
            }

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
            out = network.getOutputsInfo();
            InferenceEngine::BlobMap outputBlobs;

            std::pair<std::string, InferenceEngine::DataPtr> item = *out.begin();

            InferenceEngine::TBlob<float>::Ptr output;
            output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
            output->allocate();
            outputBlobs[item.first] = output;

            graph.Infer(srcs, outputBlobs);

            size_t c1_oh = (p.in.h + 2 * p.conv.pad_h - p.conv.krn_h) / p.conv.str_h + 1;
            size_t c1_ow = (p.in.w + 2 * p.conv.pad_w - p.conv.krn_w) / p.conv.str_w + 1;
            InferenceEngine::TBlob<float> dst_ref(InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32, {c1_ow, c1_oh, p.conv.out_c, p.in.n}, InferenceEngine::NCHW));
            dst_ref.allocate();

            ref_conv_depthwise(*srcPtr, (const float *)weights->buffer(), dst_ref, p);

            compare(*output, dst_ref);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNGraphConvDepthwiseFusingTests, TestsConvDepthwiseFusing) {}

INSTANTIATE_TEST_CASE_P(
        TestsConvDepthwiseFusing, MKLDNNGraphConvDepthwiseFusingTests,
        ::testing::Values(
                conv_depthwise_fusing_test_params{{1, 64, 5, 5}, {1, 1, 1, 1, 0, 0, 48, 1}, depthwise_scale_shift, false},
                conv_depthwise_fusing_test_params{{1, 64, 5, 5}, {1, 1, 1, 1, 0, 0, 48, 1}, depthwise_prelu, false},
                conv_depthwise_fusing_test_params{{1, 64, 5, 5}, {1, 1, 1, 1, 0, 0, 48, 1}, depthwise_scale_shift, true},
                conv_depthwise_fusing_test_params{{1, 64, 5, 5}, {1, 1, 1, 1, 0, 0, 48, 1}, depthwise_prelu, true},
                conv_depthwise_fusing_test_params{{1, 48, 9, 9}, {3, 3, 1, 1, 1, 1, 64, 1}, depthwise_scale_shift, false},
                conv_depthwise_fusing_test_params{{1, 48, 9, 9}, {3, 3, 1, 1, 1, 1, 64, 1}, depthwise_prelu, false},
                conv_depthwise_fusing_test_params{{1, 48, 9, 9}, {3, 3, 1, 1, 1, 1, 64, 1}, depthwise_scale_shift, true},
                conv_depthwise_fusing_test_params{{1, 48, 9, 9}, {3, 3, 1, 1, 1, 1, 64, 1}, depthwise_prelu, true},
                conv_depthwise_fusing_test_params{{1, 48, 11, 11}, {3, 3, 1, 1, 1, 1, 48, 48}, depthwise_scale_shift, false},
                conv_depthwise_fusing_test_params{{1, 48, 11, 11}, {3, 3, 1, 1, 1, 1, 48, 48}, depthwise_prelu, false},
                conv_depthwise_fusing_test_params{{1, 48, 11, 11}, {3, 3, 1, 1, 1, 1, 48, 48}, depthwise_scale_shift, true},
                conv_depthwise_fusing_test_params{{1, 48, 11, 11}, {3, 3, 1, 1, 1, 1, 48, 48}, depthwise_prelu, true},
                conv_depthwise_fusing_test_params{{1, 3, 11, 11}, {3, 3, 1, 1, 1, 1, 3, 3}, depthwise_scale_shift, false},
                conv_depthwise_fusing_test_params{{1, 3, 11, 11}, {3, 3, 1, 1, 1, 1, 3, 3}, depthwise_prelu, false},
                conv_depthwise_fusing_test_params{{1, 3, 11, 11}, {3, 3, 1, 1, 1, 1, 3, 3}, depthwise_scale_shift, true},
                conv_depthwise_fusing_test_params{{1, 3, 11, 11}, {3, 3, 1, 1, 1, 1, 3, 3}, depthwise_prelu, true}
        ));
