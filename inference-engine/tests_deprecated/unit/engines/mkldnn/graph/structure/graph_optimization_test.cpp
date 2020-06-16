// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "../test_graph.hpp"

#include "single_layer_common.hpp"
#include <mkldnn_extension_mngr.h>
#include "tests_common.hpp"
#include <ie_core.hpp>


using namespace ::testing;
using namespace std;
using namespace mkldnn;

class MKLDNNGraphOptimizationTests: public TestsCommon {};

TEST_F(MKLDNNGraphOptimizationTests, TestNoFuseConvSumWithOneInput) {
    std::string model = R"V0G0N(
<net name="AlexNet" version="2" batch="1">
    <layers>
        <layer name="data" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>5</dim>
                    <dim>5</dim>
                </port>
            </output>
        </layer>
        <layer name="conv1" type="Convolution" precision="FP32" id="1">
            <convolution_data stride-x="1" stride-y="1" pad-x="0" pad-y="0" kernel-x="1" kernel-y="1" output="3" group="1"/>
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>5</dim>
                    <dim>5</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>5</dim>
                    <dim>5</dim>
                </port>
            </output>
            <weights offset="0" size="36"/>
            <biases offset="36" size="12"/>
        </layer>
        <layer name="res2a" type="Eltwise" precision="FP32" id="2">
            <elementwise_data operation="sum"/>
            <input>
                <port id="3">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>5</dim>
                    <dim>5</dim>
                </port>
                <port id="4">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>5</dim>
                    <dim>5</dim>
                </port>
            </input>
            <output>
                <port id="5">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>5</dim>
                    <dim>5</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="3"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="4"/>
    </edges>
</net>

)V0G0N";

    InferenceEngine::TBlob<uint8_t> *weights = new InferenceEngine::TBlob<uint8_t>({ InferenceEngine::Precision::U8, {48}, InferenceEngine::C });
    weights->allocate();
    float * data = weights->buffer();

    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));
    InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr = InferenceEngine::TBlob<uint8_t>::Ptr(weights);

    InferenceEngine::Core ie;
    InferenceEngine::CNNNetwork network;
    ASSERT_NO_THROW(network = ie.ReadNetwork(model, weights_ptr));

    MKLDNNGraphTestClass graph;
    ASSERT_NO_THROW(graph.CreateGraph(network));

    bool fused = true;
    auto& nodes = graph.getNodes();
    for (auto &node : nodes) {
        if (node->getType() == MKLDNNPlugin::Convolution) {
            fused = false;
        }
    }
    ASSERT_FALSE(fused);
}

TEST_F(MKLDNNGraphOptimizationTests, DISABLED_TestNoCrashForFuseConvSumAndInput) {
    std::string model = R"V0G0N(
<net name="AlexNet" version="2" batch="1">
    <layers>
        <layer name="data" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>5</dim>
                    <dim>5</dim>
                </port>
            </output>
        </layer>
        <layer name="conv1" type="Convolution" precision="FP32" id="1">
            <convolution_data stride-x="1" stride-y="1" pad-x="0" pad-y="0" kernel-x="1" kernel-y="1" output="3" group="1"/>
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>5</dim>
                    <dim>5</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>5</dim>
                    <dim>5</dim>
                </port>
            </output>
            <weights offset="0" size="36"/>
            <biases offset="36" size="12"/>
        </layer>
        <layer name="relu1" type="ReLU" precision="FP32" id="2">
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>5</dim>
                    <dim>5</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>5</dim>
                    <dim>5</dim>
                </port>
            </output>
        </layer>
        <layer name="res2a" type="Eltwise" precision="FP32" id="3">
            <elementwise_data operation="sum"/>
            <input>
                <port id="3">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>5</dim>
                    <dim>5</dim>
                </port>
                <port id="4">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>5</dim>
                    <dim>5</dim>
                </port>
            </input>
            <output>
                <port id="5">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>5</dim>
                    <dim>5</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="3" to-port="3"/>
        <edge from-layer="2" from-port="2" to-layer="3" to-port="4"/>
    </edges>
</net>

)V0G0N";

    InferenceEngine::TBlob<uint8_t> *weights = new InferenceEngine::TBlob<uint8_t>({ InferenceEngine::Precision::U8, {48}, InferenceEngine::C });
    weights->allocate();
    float * data = weights->buffer();

    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));
    InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr = InferenceEngine::TBlob<uint8_t>::Ptr(weights);

    InferenceEngine::Core ie;
    InferenceEngine::CNNNetwork network;
    ASSERT_NO_THROW(ie.ReadNetwork(model, weights_ptr));

    MKLDNNGraphTestClass graph;
    ASSERT_NO_THROW(graph.CreateGraph(network));

    bool fused = false;
    auto& nodes = graph.getNodes();
    for (auto &node : nodes) {
        if (node->isFusedWith(MKLDNNPlugin::Eltwise)) {
            fused = true;
        }
    }
    ASSERT_TRUE(fused);
}
