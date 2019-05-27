// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>

#include <cnn_network_int8_normalizer.hpp>
#include "tests_common.hpp"
#include "ir_gen_helper.hpp"

using namespace ::testing;
using namespace single_layer_tests;

struct conv_conv_eltwise_params {
    // Formats: NCHW, NCDHW
    std::vector<size_t> in;

    conv_common_params conv;
    eltwise_common_params eltwise;
};

class NormalizationConvConvEltwiseTests: public TestsCommon,
                                    public WithParamInterface<conv_conv_eltwise_params> {
    std::string layers_t = R"V0G0N(
        <layer id="1" name="conv_1" precision="FP32" type="Convolution">
            <data group="_GC_" kernel="_K_" output="_OC_" pads_begin="_PB_" pads_end="_PE_" strides="_KS_"/>
            <input>
                <port id="0">
                    __INP_DIMS__
                </port>
            </input>
            <output>
                <port id="1">
                    __CONV_OUT_DIMS__
                </port>
            </output>
            <blobs>
                <weights offset="0" size="1"/>
                <biases offset="1" size="2"/>
            </blobs>
        </layer>
        <layer id="2" name="conv_2" precision="FP32" type="Convolution">
            <data group="_GC_" kernel="_K_" output="_OC_" pads_begin="_PB_" pads_end="_PE_" strides="_KS_"/>
            <input>
                <port id="0">
                    __INP_DIMS__
                </port>
            </input>
            <output>
                <port id="1">
                    __CONV_OUT_DIMS__
                </port>
            </output>
            <blobs>
                <weights offset="3" size="4"/>
                <biases offset="4" size="5"/>
            </blobs>
        </layer>
        <layer id="3" name="eltwise_block" precision="FP32" type="Eltwise">
            <data coeff="" operation="sum"/>
            <input>
                <port id="0">
                    __CONV_OUT_DIMS__
                </port>
                <port id="1">
                    __CONV_OUT_DIMS__
                </port>
            </input>
            <output>
                <port id="2">
                    __CONV_OUT_DIMS__
                </port>
            </output>
        </layer>
)V0G0N";

    std::string edges_t = R"V0G0N(
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="3" to-port="0"/>
        <edge from-layer="2" from-port="1" to-layer="3" to-port="1"/>
)V0G0N";

    std::string getModel(conv_conv_eltwise_params p) {
        std::string model = layers_t;
        
        std::string s_dims;
        for (auto& dim : p.in) {
            s_dims += "\n                    <dim>";
            s_dims += std::to_string(dim) + "</dim>";
        }
	REPLACE_WITH_STR(model, "__INP_DIMS__", s_dims);

        s_dims = "\n                    <dim>";
        s_dims += std::to_string(p.in[0]) + "</dim>";
        s_dims += "\n                    <dim>";
        s_dims += std::to_string(p.conv.out_c) + "</dim>";
        int k_len = p.conv.kernel.size();
        for (size_t i = 2; i < p.in.size(); i++) {
            size_t inx = k_len - i + 1;
            size_t dim = (p.in[i] + 2lu * p.conv.pads_begin[inx] - p.conv.kernel[inx]) / p.conv.stride[inx] + 1lu;
            s_dims += "\n                    <dim>";
            s_dims += std::to_string(dim) + "</dim>";
        }
	REPLACE_WITH_STR(model, "__CONV_OUT_DIMS__", s_dims);

        REPLACE_WITH_NUM_VECTOR_REVERSE(model, "_K_", p.conv.kernel);
        REPLACE_WITH_NUM_VECTOR_REVERSE(model, "_KS_", p.conv.stride);
        REPLACE_WITH_NUM_VECTOR_REVERSE(model, "_PB_", p.conv.pads_begin);
        REPLACE_WITH_NUM_VECTOR_REVERSE(model, "_PE_", p.conv.pads_end);
        REPLACE_WITH_NUM(model, "_GC_", p.conv.group);
        REPLACE_WITH_NUM(model, "_OC_", p.conv.out_c);

        model = IRTemplateGenerator::getIRTemplate("Deconvolution_Concat", p.in, "FP32", model, edges_t);

        return model;
    }

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            conv_conv_eltwise_params p = ::testing::WithParamInterface<conv_conv_eltwise_params>::GetParam();
            std::string model = getModel(p);

            InferenceEngine::CNNNetReader net_reader;
            ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

            auto network = net_reader.getNetwork();

            int maxSign = 0x7F;
            int maxUnsign = 0xFF;

            InferenceEngine::details::CNNStatisticHelper statHelper(network, {}, maxSign, maxUnsign);
            auto conv_1 = network.getLayerByName("conv_1");
            auto conv_2 = network.getLayerByName("conv_2");
            auto eltwise = network.getLayerByName("eltwise_block");

            ASSERT_EQ(eltwise, statHelper.getLatestInFuse(conv_1));
            ASSERT_EQ(conv_2, statHelper.getLatestInFuse(conv_2));
            ASSERT_EQ(eltwise, statHelper.getLatestInFuse(eltwise));
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(NormalizationConvConvEltwiseTests, TestsConvConvEltwise) {}

INSTANTIATE_TEST_CASE_P(
        TestsConvConvEltwise, NormalizationConvConvEltwiseTests,
        ::testing::Values(
                conv_conv_eltwise_params{{1, 16, 4, 4}, 
                                     { {1, 1}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, "", 1, 32, true },
                                     {"sum", {}} },
                conv_conv_eltwise_params{{1, 16, 4, 4, 4},
                                     { {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}, {1, 1, 1}, "", 1, 32, true },
                                     {"sum", {}} }
        ));
