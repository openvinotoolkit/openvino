// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "mkldnn_exec_network.h"

#include <mkldnn_extension_utils.h>
#include "tests_common.hpp"
#include "../test_graph.hpp"
#include <ie_ir_reader.hpp>

// to fix compilation in Debug mode
IE_SUPPRESS_DEPRECATED_START
#include <ie_builders.hpp>
IE_SUPPRESS_DEPRECATED_END

using namespace ::testing;
using namespace std;
using namespace mkldnn;

class MKLDNNGraphStructureTests: public TestsCommon {};

TEST_F(MKLDNNGraphStructureTests, TestNoRedundantReorders) {
    std::string model = R"V0G0N(
<net name="PVANET" version="2" batch="1">
    <layers>
        <layer name="data" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>544</dim>
                    <dim>992</dim>
                </port>
            </output>
        </layer>
        <layer name="conv1_1_conv" type="Convolution" precision="FP32" id="2">
            <convolution_data stride-x="2" stride-y="2" pad-x="3" pad-y="3" kernel-x="7" kernel-y="7" output="16" group="1"/>
            <input>
                <port id="2">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>544</dim>
                    <dim>992</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>16</dim>
                    <dim>272</dim>
                    <dim>496</dim>
                </port>
            </output>
            <weights offset="0" size="9408"/>
            <biases offset="9408" size="64"/>
        </layer>
        <layer name="conv1_1_neg" type="Power" precision="FP32" id="3">
            <power_data power="1" scale="-1" shift="0"/>
            <input>
                <port id="4">
                    <dim>1</dim>
                    <dim>16</dim>
                    <dim>272</dim>
                    <dim>496</dim>
                </port>
            </input>
            <output>
                <port id="5">
                    <dim>1</dim>
                    <dim>16</dim>
                    <dim>272</dim>
                    <dim>496</dim>
                </port>
            </output>
        </layer>
        <layer name="conv1_1_concat" type="Concat" precision="FP32" id="4">
            <concat_data axis="1"/>
            <input>
                <port id="6">
                    <dim>1</dim>
                    <dim>16</dim>
                    <dim>272</dim>
                    <dim>496</dim>
                </port>
                <port id="7">
                    <dim>1</dim>
                    <dim>16</dim>
                    <dim>272</dim>
                    <dim>496</dim>
                </port>
            </input>
            <output>
                <port id="8">
                    <dim>1</dim>
                    <dim>32</dim>
                    <dim>272</dim>
                    <dim>496</dim>
                </port>
            </output>
        </layer>
        <layer name="conv1_1_scale" type="ScaleShift" precision="FP32" id="5">
            <input>
                <port id="9">
                    <dim>1</dim>
                    <dim>32</dim>
                    <dim>272</dim>
                    <dim>496</dim>
                </port>
            </input>
            <output>
                <port id="10">
                    <dim>1</dim>
                    <dim>32</dim>
                    <dim>272</dim>
                    <dim>496</dim>
                </port>
            </output>
            <weights offset="9472" size="128"/>
            <biases offset="9600" size="128"/>
        </layer>
        <layer name="conv1_1_relu" type="ReLU" precision="FP32" id="6">
            <data negative_slope="0" engine="caffe.ReLUParameter.DEFAULT"/>
            <input>
                <port id="11">
                    <dim>1</dim>
                    <dim>32</dim>
                    <dim>272</dim>
                    <dim>496</dim>
                </port>
            </input>
            <output>
                <port id="12">
                    <dim>1</dim>
                    <dim>32</dim>
                    <dim>272</dim>
                    <dim>496</dim>
                </port>
            </output>
        </layer>
        <layer name="pool1" type="Pooling" precision="FP32" id="7">
            <pooling_data kernel-x="3" kernel-y="3" pad-x="0" pad-y="0" stride-x="2" stride-y="2" rounding-type="ceil" pool-method="max"/>
            <input>
                <port id="13">
                    <dim>1</dim>
                    <dim>32</dim>
                    <dim>272</dim>
                    <dim>496</dim>
                </port>
            </input>
            <output>
                <port id="14">
                    <dim>1</dim>
                    <dim>32</dim>
                    <dim>136</dim>
                    <dim>248</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="2"/>
        <edge from-layer="2" from-port="3" to-layer="3" to-port="4"/>
        <edge from-layer="2" from-port="3" to-layer="4" to-port="6"/>
        <edge from-layer="3" from-port="5" to-layer="4" to-port="7"/>
        <edge from-layer="4" from-port="8" to-layer="5" to-port="9"/>
        <edge from-layer="5" from-port="10" to-layer="6" to-port="11"/>
        <edge from-layer="6" from-port="12" to-layer="7" to-port="13"/>
    </edges>
</net>)V0G0N";



    InferenceEngine::CNNNetReader net_reader;
    ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

    InferenceEngine::TBlob<uint8_t> *weights = new InferenceEngine::TBlob<uint8_t>({ InferenceEngine::Precision::U8, {9728}, InferenceEngine::C });
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));
    InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr = InferenceEngine::TBlob<uint8_t>::Ptr(weights);

    net_reader.SetWeights(weights_ptr);


    MKLDNNGraphTestClass graph;
    graph.CreateGraph(net_reader.getNetwork());

    size_t reorders_num = 0;
    auto& nodes = graph.getNodes();
    for (auto &node : nodes) {
        if (node->getType() == MKLDNNPlugin::Reorder) {
            reorders_num++;
            ASSERT_EQ(MKLDNNPlugin::Output, node->getChildEdgeAt(0)->getChild()->getType());
        }
    }
    ASSERT_EQ(reorders_num, 1);
}

TEST_F(MKLDNNGraphStructureTests, TestRedundantReorderBeforeConvWithC_3) {
    std::string model = R"V0G0N(
<net name="net" version="2" batch="1">
    <layers>
        <layer name="data" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>320</dim>
                    <dim>544</dim>
                </port>
            </output>
        </layer>
        <layer name="data_norm_bn" type="BatchNormalization" precision="FP32" id="1">
            <batch_norm_data epsilon="9.9999997473787516e-06"/>
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>320</dim>
                    <dim>544</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>320</dim>
                    <dim>544</dim>
                </port>
            </output>
            <biases offset="0" size="12"/>
            <weights offset="12" size="12"/>
        </layer>
        <layer name="data_norm_scale" type="ScaleShift" precision="FP32" id="2">
            <input>
                <port id="3">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>320</dim>
                    <dim>544</dim>
                </port>
            </input>
            <output>
                <port id="4">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>320</dim>
                    <dim>544</dim>
                </port>
            </output>
            <weights offset="24" size="12"/>
            <biases offset="36" size="12"/>
        </layer>
        <layer name="init_conv" type="Convolution" precision="FP32" id="3">
            <convolution_data stride-x="2" stride-y="2" pad-x="3" pad-y="3" kernel-x="7" kernel-y="7" output="64" group="1"/>
            <input>
                <port id="5">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>320</dim>
                    <dim>544</dim>
                </port>
            </input>
            <output>
                <port id="6">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>160</dim>
                    <dim>272</dim>
                </port>
            </output>
            <weights offset="48" size="37632"/>
            <biases offset="37680" size="256"/>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="3"/>
        <edge from-layer="2" from-port="4" to-layer="3" to-port="5"/>
    </edges>
</net>
)V0G0N";

    InferenceEngine::CNNNetReader net_reader;
    ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

    InferenceEngine::TBlob<uint8_t> *weights = new InferenceEngine::TBlob<uint8_t>({ InferenceEngine::Precision::U8, {37936}, InferenceEngine::C });
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));
    InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr = InferenceEngine::TBlob<uint8_t>::Ptr(weights);

    net_reader.SetWeights(weights_ptr);


    MKLDNNGraphTestClass graph;
    graph.CreateGraph(net_reader.getNetwork());

    size_t reorders_num = 0;
    auto& nodes = graph.getNodes();
    for (auto &node : nodes) {
        if (node->getType() == MKLDNNPlugin::Reorder) {
            reorders_num++;
            if (node->getChildEdgeAt(0)->getChild()->getName() == "init_conv"){
                ASSERT_EQ(MKLDNNPlugin::Convolution, node->getChildEdgeAt(0)->getChild()->getType());
                ASSERT_EQ(InferenceEngine::Layout::NCHW,
                          node->getChildEdgeAt(0)->getBlob()->getTensorDesc().getLayout());
            }
        }
    }
    ASSERT_EQ(reorders_num, 3);
}

TEST_F(MKLDNNGraphStructureTests, TestNoRedundantReordersBeforeConcat) {
    std::string model = R"V0G0N(
<net name="net" version="2" batch="1">
    <layers>
        <layer name="data" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>7</dim>
                    <dim>7</dim>
                </port>
            </output>
        </layer>
        <layer name="conv1_1_conv" type="Convolution" precision="FP32" id="2">
            <convolution_data stride-x="2" stride-y="2" pad-x="3" pad-y="3" kernel-x="7" kernel-y="7" output="4" group="1"/>
            <input>
                <port id="2">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>7</dim>
                    <dim>7</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>4</dim>
                    <dim>4</dim>
                    <dim>4</dim>
                </port>
            </output>
            <weights offset="0" size="2352"/>
            <biases offset="2352" size="16"/>
        </layer>
        <layer name="conv1_1_neg" type="Power" precision="FP32" id="3">
            <power_data power="1" scale="-1" shift="0"/>
            <input>
                <port id="4">
                    <dim>1</dim>
                    <dim>4</dim>
                    <dim>4</dim>
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="5">
                    <dim>1</dim>
                    <dim>4</dim>
                    <dim>4</dim>
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer name="conv1_1_concat" type="Concat" precision="FP32" id="4">
            <concat_data axis="1"/>
            <input>
                <port id="6">
                    <dim>1</dim>
                    <dim>4</dim>
                    <dim>4</dim>
                    <dim>4</dim>
                </port>
                <port id="7">
                    <dim>1</dim>
                    <dim>4</dim>
                    <dim>4</dim>
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="8">
                    <dim>1</dim>
                    <dim>8</dim>
                    <dim>4</dim>
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer name="conv1_1_scale" type="ScaleShift" precision="FP32" id="5">
            <input>
                <port id="9">
                    <dim>1</dim>
                    <dim>8</dim>
                    <dim>4</dim>
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="10">
                    <dim>1</dim>
                    <dim>8</dim>
                    <dim>4</dim>
                    <dim>4</dim>
                </port>
            </output>
            <weights offset="2368" size="32"/>
            <biases offset="2400" size="32"/>
        </layer>
        <layer name="conv1_1_relu" type="ReLU" precision="FP32" id="6">
            <input>
                <port id="11">
                    <dim>1</dim>
                    <dim>8</dim>
                    <dim>4</dim>
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="12">
                    <dim>1</dim>
                    <dim>8</dim>
                    <dim>4</dim>
                    <dim>4</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="2"/>
        <edge from-layer="2" from-port="3" to-layer="3" to-port="4"/>
        <edge from-layer="2" from-port="3" to-layer="4" to-port="6"/>
        <edge from-layer="3" from-port="5" to-layer="4" to-port="7"/>
        <edge from-layer="4" from-port="8" to-layer="5" to-port="9"/>
        <edge from-layer="5" from-port="10" to-layer="6" to-port="11"/>
    </edges>
</net>
)V0G0N";

    InferenceEngine::CNNNetReader net_reader;
    ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

    InferenceEngine::TBlob<uint8_t> *weights = new InferenceEngine::TBlob<uint8_t>({ InferenceEngine::Precision::U8, {2432}, InferenceEngine::C });
    weights->allocate();
    float * data = weights->buffer();

    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));
    size_t idx = 592; // Convolution weights
    size_t size = 8; // Scale and shift sizes
    for (size_t i = 0; i < size; i++, idx++) {
        data[idx] = 1.f;
    }
    for (size_t i = 0; i < size; i++, idx++) {
        data[idx] = 0.f;
    }

    InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr = InferenceEngine::TBlob<uint8_t>::Ptr(weights);

    net_reader.SetWeights(weights_ptr);

    MKLDNNGraphTestClass graph;
    graph.CreateGraph(net_reader.getNetwork());

    size_t reorders_num = 0;
    auto& nodes = graph.getNodes();
    for (auto &node : nodes) {
        if (node->getType() == MKLDNNPlugin::Reorder && node->getChildEdgeAt(0)->getChild()->getType() != MKLDNNPlugin::Output) {
            reorders_num++;
        }
    }
    ASSERT_EQ(reorders_num, 2);
    InferenceEngine::TensorDesc desc(InferenceEngine::Precision::FP32, {1, 3, 7, 7}, InferenceEngine::NCHW);
    InferenceEngine::Blob::Ptr src = InferenceEngine::make_shared_blob<float>(desc);
    src->allocate();
    data = src->buffer().as<float *>();
    for (size_t i = 0; i < src->size(); i++) {
        data[i] = (i % 2) ? 1 : -1;
    }

    InferenceEngine::BlobMap srcs;
    srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("data", src));

    InferenceEngine::OutputsDataMap out = net_reader.getNetwork().getOutputsInfo();

    InferenceEngine::BlobMap outputBlobs;
    std::pair<std::string, InferenceEngine::DataPtr> item = *out.begin();

    InferenceEngine::TBlob<float>::Ptr output;
    output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
    output->allocate();
    outputBlobs[item.first] = output;

    graph.Infer(srcs, outputBlobs);

    std::vector<float> refDst = {0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.040f, 0.000f,
                                 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f,
                                 0.119f, 0.000f, 0.000f, 1.889f, 0.000f, 0.000f, 0.000f, 1.138f, 0.647f, 0.000f, 0.348f,
                                 0.000f, 1.711f, 1.311f, 0.000f, 0.000f, 3.045f, 1.203f, 0.000f, 0.927f, 2.041f, 0.000f,
                                 0.564f, 1.415f, 1.524f, 0.000f, 1.812f, 0.486f, 0.103f, 1.606f, 0.999f, 0.000f, 1.145f,
                                 2.158f, 0.712f, 0.000f, 0.009f, 0.756f, 0.000f, 0.000f, 0.008f, 0.243f,

                                 0.381f, 0.363f, 1.846f, 0.804f, 1.372f, 1.113f, 2.453f, 1.609f, 0.557f, 0.000f, 3.020f,
                                 1.422f, 0.481f, 0.221f, 1.137f, 0.401f, 1.475f, 0.301f, 0.862f, 2.052f, 2.680f, 0.284f,
                                 0.000f, 2.389f, 0.917f, 0.000f, 0.358f, 1.989f, 0.355f, 0.000f, 0.000f, 0.570f, 0.000f,
                                 0.761f, 0.000f, 0.000f, 0.652f, 0.910f, 0.000f, 0.000f, 0.226f, 0.000f, 0.000f, 0.323f,
                                 0.000f, 0.000f, 0.000f, 0.108f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.866f, 0.000f,
                                 0.000f, 0.000f, 0.759f, 0.000f, 0.000f, 0.029f, 1.186f, 0.000f, 0.000f};
    InferenceEngine::TBlob<float>::Ptr dstOut = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc(), refDst.data());

    compare(*output, *dstOut);

    // Compare for batch2
    net_reader.getNetwork().setBatchSize(2);
    graph.CreateGraph(net_reader.getNetwork());
    desc = InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32, {2, 3, 7, 7}, InferenceEngine::NCHW);

    InferenceEngine::Blob::Ptr srcBatch = InferenceEngine::make_shared_blob<float>(desc);
    srcBatch->allocate();
    data = srcBatch->buffer().as<float *>();
    float *originData = src->buffer().as<float *>();
    for(size_t b = 0; b < 2; b++) {
        for (size_t i = 0; i < src->size(); i++) {
            data[srcBatch->getTensorDesc().offset(b*src->size() + i)] = originData[src->getTensorDesc().offset(i)];
        }
    }

    srcs.clear();
    srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("data", srcBatch));
    out = net_reader.getNetwork().getOutputsInfo();

    outputBlobs.clear();
    item = *out.begin();
    output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
    output->allocate();
    outputBlobs[item.first] = output;

    graph.Infer(srcs, outputBlobs);
    dstOut = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
    dstOut->allocate();
    data = dstOut->buffer().as<float *>();
    for(size_t b = 0; b < 2; b++) {
        for (size_t i = 0; i < refDst.size(); i++) {
            data[dstOut->getTensorDesc().offset(b*refDst.size() + i)] = refDst[i];
        }
    }

    compare(*output, *dstOut);
}

TEST_F(MKLDNNGraphStructureTests, TestNoRedundantReordersBeforeDWConvolution) {
    std::string model = R"V0G0N(
<net name="net" version="2" batch="1">
    <layers>
        <layer name="input" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>2</dim>
                    <dim>3</dim>
                    <dim>5</dim>
                    <dim>5</dim>
                </port>
            </output>
        </layer>
        <layer name="conv2_1_1" type="Convolution" precision="FP32" id="1">
            <convolution_data stride-x="1" stride-y="1" pad-x="0" pad-y="0" kernel-x="1" kernel-y="1" output="4" group="1"/>
            <input>
                <port id="1">
                    <dim>2</dim>
                    <dim>3</dim>
                    <dim>5</dim>
                    <dim>5</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>2</dim>
                    <dim>4</dim>
                    <dim>5</dim>
                    <dim>5</dim>
                </port>
            </output>
            <weights offset="0" size="48"/>
            <biases offset="48" size="16"/>
        </layer>
        <layer name="conv2_1_1_relu" type="ReLU" precision="FP32" id="2">
            <data negative_slope="0" engine="caffe.ReLUParameter.DEFAULT"/>
            <input>
                <port id="3">
                    <dim>2</dim>
                    <dim>4</dim>
                    <dim>5</dim>
                    <dim>5</dim>
                </port>
            </input>
            <output>
                <port id="4">
                    <dim>2</dim>
                    <dim>4</dim>
                    <dim>5</dim>
                    <dim>5</dim>
                </port>
            </output>
        </layer>
        <layer name="conv2_1_2_orig" type="Convolution" precision="FP32" id="3">
            <convolution_data stride-x="1" stride-y="1" pad-x="1" pad-y="1" kernel-x="3" kernel-y="3" output="4" group="4"/>
            <input>
                <port id="5">
                    <dim>2</dim>
                    <dim>4</dim>
                    <dim>5</dim>
                    <dim>5</dim>
                </port>
            </input>
            <output>
                <port id="6">
                    <dim>2</dim>
                    <dim>4</dim>
                    <dim>5</dim>
                    <dim>5</dim>
                </port>
            </output>
            <weights offset="64" size="144"/>
            <biases offset="208" size="16"/>
        </layer>
        <layer name="conv2_1_2_neg" type="Power" precision="FP32" id="4">
            <power_data power="1" scale="-1" shift="0"/>
            <input>
                <port id="7">
                    <dim>2</dim>
                    <dim>4</dim>
                    <dim>5</dim>
                    <dim>5</dim>
                </port>
            </input>
            <output>
                <port id="8">
                    <dim>2</dim>
                    <dim>4</dim>
                    <dim>5</dim>
                    <dim>5</dim>
                </port>
            </output>
        </layer>
        <layer name="conv2_1_2" type="Concat" precision="FP32" id="5">
            <concat_data axis="1"/>
            <input>
                <port id="9">
                    <dim>2</dim>
                    <dim>4</dim>
                    <dim>5</dim>
                    <dim>5</dim>
                </port>
                <port id="10">
                    <dim>2</dim>
                    <dim>4</dim>
                    <dim>5</dim>
                    <dim>5</dim>
                </port>
            </input>
            <output>
                <port id="11">
                    <dim>2</dim>
                    <dim>8</dim>
                    <dim>5</dim>
                    <dim>5</dim>
                </port>
            </output>
        </layer>
        <layer name="conv2_1_2_scale" type="ScaleShift" precision="FP32" id="6">
            <input>
                <port id="12">
                    <dim>2</dim>
                    <dim>8</dim>
                    <dim>5</dim>
                    <dim>5</dim>
                </port>
            </input>
            <output>
                <port id="13">
                    <dim>2</dim>
                    <dim>8</dim>
                    <dim>5</dim>
                    <dim>5</dim>
                </port>
            </output>
            <weights offset="224" size="32"/>
            <biases offset="256" size="32"/>
        </layer>
        <layer name="conv2_1_2_relu" type="ReLU" precision="FP32" id="7">
            <data negative_slope="0" engine="caffe.ReLUParameter.DEFAULT"/>
            <input>
                <port id="14">
                    <dim>2</dim>
                    <dim>8</dim>
                    <dim>5</dim>
                    <dim>5</dim>
                </port>
            </input>
            <output>
                <port id="15">
                    <dim>2</dim>
                    <dim>8</dim>
                    <dim>5</dim>
                    <dim>5</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="3"/>
        <edge from-layer="2" from-port="4" to-layer="3" to-port="5"/>
        <edge from-layer="3" from-port="6" to-layer="4" to-port="7"/>
        <edge from-layer="3" from-port="6" to-layer="5" to-port="9"/>
        <edge from-layer="4" from-port="8" to-layer="5" to-port="10"/>
        <edge from-layer="5" from-port="11" to-layer="6" to-port="12"/>
        <edge from-layer="6" from-port="13" to-layer="7" to-port="14"/>
    </edges>
</net>
)V0G0N";

    InferenceEngine::CNNNetReader net_reader;
    ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

    InferenceEngine::TBlob<uint8_t> *weights = new InferenceEngine::TBlob<uint8_t>({ InferenceEngine::Precision::U8, {288}, InferenceEngine::C });
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));

    InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr = InferenceEngine::TBlob<uint8_t>::Ptr(weights);

    net_reader.SetWeights(weights_ptr);

    MKLDNNGraphTestClass graph;
    graph.CreateGraph(net_reader.getNetwork());

    size_t reorders_num = 0;
    auto& nodes = graph.getNodes();
    for (auto &node : nodes) {
        if (node->getType() == MKLDNNPlugin::Reorder) {
            reorders_num++;
        }
    }
    ASSERT_EQ(reorders_num, 2);
    InferenceEngine::TensorDesc desc(InferenceEngine::Precision::FP32, {2, 3, 5, 5}, InferenceEngine::NCHW);
    InferenceEngine::Blob::Ptr src = InferenceEngine::make_shared_blob<float>(desc);
    src->allocate();
    auto *data = src->buffer().as<float *>();
    size_t sizeB1 = src->size() / 2;
    fill_data(data, sizeB1);
    for (size_t i = 0; i < sizeB1; i++) {
        data[sizeB1 + i] = data[i];
    }

    InferenceEngine::BlobMap srcs;
    srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("input", src));

    InferenceEngine::OutputsDataMap out = net_reader.getNetwork().getOutputsInfo();

    InferenceEngine::BlobMap outputBlobs;
    std::pair<std::string, InferenceEngine::DataPtr> item = *out.begin();

    InferenceEngine::TBlob<float>::Ptr output;
    output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
    output->allocate();
    outputBlobs[item.first] = output;

    graph.Infer(srcs, outputBlobs);

    std::vector<float> refDst = {0.920f, 0.920f, 0.920f, 0.920f, 0.920f, 0.920f, 0.920f, 0.920f, 0.920f, 0.920f, 0.920f,
                                 0.920f, 0.920f, 0.920f, 0.920f, 0.920f, 0.920f, 0.920f, 0.920f, 0.920f, 0.920f, 0.920f,
                                 0.920f, 0.920f, 0.920f, 0.827f, 0.827f, 0.827f, 0.827f, 0.827f, 0.827f, 0.827f, 0.827f,
                                 0.827f, 0.827f, 0.827f, 0.827f, 0.827f, 0.827f, 0.827f, 0.827f, 0.827f, 0.827f, 0.827f,
                                 0.827f, 0.827f, 0.827f, 0.827f, 0.827f, 0.827f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f,
                                 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f,
                                 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f,
                                 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f,
                                 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f,
                                 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f,
                                 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f,
                                 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f,
                                 0.000f, 0.185f, 0.176f, 0.000f, 0.000f, 0.000f, 0.215f, 0.000f, 0.957f, 1.092f, 0.000f,
                                 0.000f, 0.213f, 0.020f, 1.391f, 2.359f, 0.583f, 0.000f, 0.000f, 0.138f, 0.043f, 0.000f,
                                 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.720f, 0.000f, 0.000f, 0.000f,
                                 0.000f, 0.069f, 0.188f, 0.046f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.045f,
                                 0.041f, 0.000f, 0.000f, 0.056f, 0.000f, 0.000f, 0.086f, 0.025f, 0.000f, 0.000f, 0.000f,
                                 0.000f, 0.012f, 0.056f, 0.000f, 0.060f, 0.055f, 0.000f, 0.000f, 0.037f, 0.000f, 0.000f,
                                 0.000f, 0.000f,

                                 0.920f, 0.920f, 0.920f, 0.920f, 0.920f, 0.920f, 0.920f, 0.920f, 0.920f, 0.920f, 0.920f,
                                 0.920f, 0.920f, 0.920f, 0.920f, 0.920f, 0.920f, 0.920f, 0.920f, 0.920f, 0.920f, 0.920f,
                                 0.920f, 0.920f, 0.920f, 0.827f, 0.827f, 0.827f, 0.827f, 0.827f, 0.827f, 0.827f, 0.827f,
                                 0.827f, 0.827f, 0.827f, 0.827f, 0.827f, 0.827f, 0.827f, 0.827f, 0.827f, 0.827f, 0.827f,
                                 0.827f, 0.827f, 0.827f, 0.827f, 0.827f, 0.827f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f,
                                 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f,
                                 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f,
                                 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f,
                                 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f,
                                 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f,
                                 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f,
                                 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f,
                                 0.000f, 0.185f, 0.176f, 0.000f, 0.000f, 0.000f, 0.215f, 0.000f, 0.957f, 1.092f, 0.000f,
                                 0.000f, 0.213f, 0.020f, 1.391f, 2.359f, 0.583f, 0.000f, 0.000f, 0.138f, 0.043f, 0.000f,
                                 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.720f, 0.000f, 0.000f, 0.000f,
                                 0.000f, 0.069f, 0.188f, 0.046f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.045f,
                                 0.041f, 0.000f, 0.000f, 0.056f, 0.000f, 0.000f, 0.086f, 0.025f, 0.000f, 0.000f, 0.000f,
                                 0.000f, 0.012f, 0.056f, 0.000f, 0.060f, 0.055f, 0.000f, 0.000f, 0.037f, 0.000f, 0.000f,
                                 0.000f, 0.000f};
    InferenceEngine::TBlob<float>::Ptr dstOut = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc(), refDst.data());

    compare(*output, *dstOut);
}

// TODO change hardcoded reference to dynamically generated
TEST_F(MKLDNNGraphStructureTests, DISABLED_TestNoRedundantReordersBeforeDWDeconvolution) {
    std::string model = R"V0G0N(
<net name="deconv" version="2" batch="1">
    <layers>
        <layer name="input" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer name="conv1" type="Convolution" precision="FP32" id="1">
            <convolution_data stride-x="1" stride-y="1" pad-x="0" pad-y="0" kernel-x="1" kernel-y="1" output="12" group="1"/>
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>12</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                </port>
            </output>
            <weights offset="0" size="144"/>
        </layer>
        <layer name="deconv1" type="Deconvolution" precision="FP32" id="2">
            <deconvolution_data stride-x="2" stride-y="2" pad-x="1" pad-y="1" kernel-x="4" kernel-y="4" output="12" group="12"/>
            <input>
                <port id="3">
                    <dim>1</dim>
                    <dim>12</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                </port>
            </input>
            <output>
                <port id="4">
                    <dim>1</dim>
                    <dim>12</dim>
                    <dim>4</dim>
                    <dim>4</dim>
                </port>
            </output>
            <weights offset="144" size="768"/>
            <biases offset="912" size="48"/>
        </layer>
        <layer name="deconv2" type="Deconvolution" precision="FP32" id="3">
            <deconvolution_data stride-x="1" stride-y="1" pad-x="1" pad-y="1" kernel-x="2" kernel-y="2" output="24" group="1"/>
            <input>
                <port id="5">
                    <dim>1</dim>
                    <dim>12</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                </port>
            </input>
            <output>
                <port id="6">
                    <dim>1</dim>
                    <dim>24</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
            <weights offset="960" size="4608"/>
            <biases offset="5568" size="96"/>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="3"/>
        <edge from-layer="1" from-port="2" to-layer="3" to-port="5"/>
    </edges>
</net>
)V0G0N";

    InferenceEngine::CNNNetReader net_reader;
    ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

    InferenceEngine::TBlob<uint8_t> *weights = new InferenceEngine::TBlob<uint8_t>({ InferenceEngine::Precision::U8, {5664}, InferenceEngine::C });
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));

    InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr = InferenceEngine::TBlob<uint8_t>::Ptr(weights);

    net_reader.SetWeights(weights_ptr);

    MKLDNNGraphTestClass graph;
    graph.CreateGraph(net_reader.getNetwork());

    size_t reorders_num = 0;
    auto& nodes = graph.getNodes();
    for (auto &node : nodes) {
        if (node->getType() == MKLDNNPlugin::Reorder) {
            reorders_num++;
            ASSERT_EQ(MKLDNNPlugin::Output, node->getChildEdgeAt(0)->getChild()->getType());
        }
    }
    ASSERT_EQ(reorders_num, 2);
    InferenceEngine::TensorDesc desc(InferenceEngine::Precision::FP32, {1, 3, 2, 2}, InferenceEngine::NCHW);
    InferenceEngine::Blob::Ptr src = InferenceEngine::make_shared_blob<float>(desc);
    src->allocate();
    fill_data(src->buffer(), src->size());

    InferenceEngine::BlobMap srcs;
    srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("input", src));

    InferenceEngine::OutputsDataMap out = net_reader.getNetwork().getOutputsInfo();

    InferenceEngine::BlobMap outputBlobs;
    InferenceEngine::DataPtr item = out["deconv1"];
    InferenceEngine::TBlob<float>::Ptr output1;
    output1 = InferenceEngine::make_shared_blob<float>(item->getTensorDesc());
    output1->allocate();
    outputBlobs["deconv1"] = output1;

    item = out["deconv2"];
    InferenceEngine::TBlob<float>::Ptr output2;
    output2 = InferenceEngine::make_shared_blob<float>(item->getTensorDesc());
    output2->allocate();
    outputBlobs["deconv2"] = output2;

    graph.Infer(srcs, outputBlobs);

    std::vector<float> refDst1 = {-0.042f, -0.563f, -0.150f, 0.396f, 0.224f, 0.229f, -0.335f, -0.390f, -0.213f, 0.959f, 0.520f, -0.507f,
                                  -0.200f, -0.202f, 0.441f, 0.499f, 0.000f, 0.000f, 0.000f, 0.000f, 0.363f, 0.141f, -0.497f, -0.332f, -0.311f,
                                  0.423f, 0.693f, -0.012f, -0.328f, -0.106f, 0.518f, 0.353f, 0.000f, 0.000f, 0.000f, 0.000f, 0.050f, -0.352f,
                                  -0.045f, 0.000f, -0.303f, 0.605f, 0.754f, -0.143f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.012f, 0.298f, 0.000f,
                                  -0.066f, -0.303f, -0.318f, -0.054f, 0.322f, 0.002f, 0.050f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f,
                                  0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f,
                                  0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f,
                                  0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.328f, -0.162f, -0.765f, -0.221f, 0.422f, 0.715f, 0.726f, 0.375f,
                                  0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, -0.744f, -0.038f, -0.109f, 0.000f, 0.583f, 0.892f,
                                  0.039f, -0.356f, 0.000f, 0.000f, 0.000f, 0.000f, -0.514f, 0.320f, 0.193f, 0.000f, -0.785f, -0.508f, 0.160f, -0.104f,
                                  0.473f, 0.214f, 0.129f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, -0.299f, 0.784f, 0.953f, -0.163f, -1.160f, -0.547f,
                                  0.401f, -0.066f, 0.275f, -0.172f, -0.683f, -0.188f, 0.384f, -0.149f, 0.151f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f,
                                  0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f,
                                  0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f};
    InferenceEngine::TBlob<float>::Ptr dstOut1 = InferenceEngine::make_shared_blob<float>(out["deconv1"]->getTensorDesc(), refDst1.data());

    std::vector<float> refDst2 = {-0.814f, -0.337f, -1.081f, 1.139f, -0.197f, 1.547f, -0.778f, -2.467f, 1.409f, -1.472f, 2.827f, 0.663f,
                                  -0.645f, 0.105f, -1.873f, -0.272f, 1.071f, 2.706f, -1.705f, 0.602f, -1.956f, 0.734f, 2.325f, -2.147f};
    InferenceEngine::TBlob<float>::Ptr dstOut2 = InferenceEngine::make_shared_blob<float>(out["deconv2"]->getTensorDesc(), refDst2.data());

    compare(*output1, *dstOut1);
    compare(*output2, *dstOut2);
}

TEST_F(MKLDNNGraphStructureTests, TestSeveralOutputToNextLayer) {
    std::string model = R"V0G0N(
<?xml version="1.0" ?>
<net batch="1" name="model" version="2">
	<layers>
		<layer id="0" name="data" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="Slice1" precision="FP32" type="Slice">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>1</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>1</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
				<port id="3">
					<dim>1</dim>
					<dim>1</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="5" name="Concat2" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>1</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>1</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>3</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
		<edge from-layer="1" from-port="1" to-layer="5" to-port="0"/>
		<edge from-layer="1" from-port="2" to-layer="5" to-port="1"/>
		<edge from-layer="1" from-port="3" to-layer="5" to-port="2"/>
	</edges>
</net>
)V0G0N";

    InferenceEngine::CNNNetReader net_reader;
    ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

    MKLDNNGraphTestClass graph;
    graph.CreateGraph(net_reader.getNetwork());

    size_t reorders_num = 0;
    auto& nodes = graph.getNodes();
    for (auto &node : nodes) {
        if (node->getType() == MKLDNNPlugin::Reorder) {
            reorders_num++;
        }
    }
    ASSERT_EQ(reorders_num, 3);
    InferenceEngine::TensorDesc desc(InferenceEngine::Precision::FP32, {1, 3, 2, 2}, InferenceEngine::NCHW);
    InferenceEngine::Blob::Ptr src = InferenceEngine::make_shared_blob<float>(desc);
    src->allocate();
    fill_data(src->buffer(), src->size());

    InferenceEngine::BlobMap srcs;
    srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("data", src));

    InferenceEngine::OutputsDataMap out = net_reader.getNetwork().getOutputsInfo();

    InferenceEngine::BlobMap outputBlobs;
    std::pair<std::string, InferenceEngine::DataPtr> item = *out.begin();

    InferenceEngine::TBlob<float>::Ptr output;
    output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
    output->allocate();
    outputBlobs[item.first] = output;

    graph.Infer(srcs, outputBlobs);

    compare(*output, *src);
}


TEST_F(MKLDNNGraphStructureTests, TestOutputAfterInplacePlusConcat) {
    std::string model = R"V0G0N(
<?xml version="1.0" ?>
<net batch="1" name="model" version="2">
	<layers>
		<layer id="0" name="data" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="Slice1" precision="FP32" type="Slice">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>1</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>1</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
				<port id="3">
					<dim>1</dim>
					<dim>1</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="Concat2" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>1</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>1</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>3</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="Reshape3" precision="FP32" type="Reshape">
			<data axis="0" dim="1,12" num_axes="-1"/>
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>12</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
		<edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>
		<edge from-layer="1" from-port="2" to-layer="2" to-port="1"/>
		<edge from-layer="1" from-port="3" to-layer="2" to-port="2"/>
		<edge from-layer="2" from-port="3" to-layer="3" to-port="0"/>
	</edges>
</net>
)V0G0N";

    InferenceEngine::CNNNetReader net_reader;
    ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));
    MKLDNNPlugin::MKLDNNExecNetwork::Ptr execNetwork(new MKLDNNPlugin::MKLDNNExecNetwork(net_reader.getNetwork(), {}, {}));
    InferenceEngine::InputsDataMap _networkInputs = net_reader.getNetwork().getInputsInfo();
    InferenceEngine::OutputsDataMap _networkOutputs = net_reader.getNetwork().getOutputsInfo();
    execNetwork->setNetworkInputs(_networkInputs);
    execNetwork->setNetworkOutputs(_networkOutputs);
    InferenceEngine::IInferRequest::Ptr inferRequest;
    execNetwork->CreateInferRequest(inferRequest);

    InferenceEngine::TensorDesc desc(InferenceEngine::Precision::FP32, {1, 3, 2, 2}, InferenceEngine::NCHW);
    InferenceEngine::Blob::Ptr src = InferenceEngine::make_shared_blob<float>(desc);
    src->allocate();
    fill_data(src->buffer(), src->size());

    InferenceEngine::ResponseDesc resp;

    InferenceEngine::StatusCode sts = inferRequest->SetBlob("data", src, &resp);
    ASSERT_EQ(InferenceEngine::OK, sts) << resp.msg;

    InferenceEngine::OutputsDataMap out = net_reader.getNetwork().getOutputsInfo();

    std::pair<std::string, InferenceEngine::DataPtr> item = *out.begin();

    InferenceEngine::TBlob<float>::Ptr output;
    output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
    output->allocate();

    sts = inferRequest->SetBlob(item.first.c_str(), output, &resp);
    ASSERT_EQ(InferenceEngine::OK, sts) << resp.msg;

    sts = inferRequest->Infer(&resp);
    ASSERT_EQ(InferenceEngine::OK, sts) << resp.msg;

    compare(*output, *src);
}

TEST_F(MKLDNNGraphStructureTests, TestResnetPart) {
    std::string modelB = R"V0G0N(
<net name="ResNet-152" version="2" batch="1">
    <layers>
        <layer name="input" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>224</dim>
                    <dim>224</dim>
                </port>
            </output>
        </layer>
        <layer name="conv1" type="Convolution" precision="FP32" id="1">
            <convolution_data stride-x="2" stride-y="2" pad-x="3" pad-y="3" kernel-x="7" kernel-y="7" output="64" group="1"/>
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>224</dim>
                    <dim>224</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
            <weights offset="0" size="37632"/>
            <biases offset="37632" size="256"/>
        </layer>
        <layer name="conv1_relu" type="ReLU" precision="FP32" id="4">
            <input>
                <port id="7">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
            <output>
                <port id="8">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer name="pool1" type="Pooling" precision="FP32" id="5">
            <pooling_data kernel-x="3" kernel-y="3" pad-x="0" pad-y="0" stride-x="2" stride-y="2" rounding-type="ceil" pool-method="max"/>
            <input>
                <port id="9">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
            <output>
                <port id="10">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>56</dim>
                    <dim>56</dim>
                </port>
            </output>
        </layer>
        <layer name="res2a_branch2a" type="Convolution" precision="FP32" id="9">
            <convolution_data stride-x="1" stride-y="1" pad-x="0" pad-y="0" kernel-x="1" kernel-y="1" output="64" group="1"/>
            <input>
                <port id="17">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>56</dim>
                    <dim>56</dim>
                </port>
            </input>
            <output>
                <port id="18">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>56</dim>
                    <dim>56</dim>
                </port>
            </output>
            <weights offset="37888" size="16384"/>
            <biases offset="54272" size="256"/>
        </layer>
        <layer name="res2a_branch2a_relu" type="ReLU" precision="FP32" id="12">
            <input>
                <port id="23">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>56</dim>
                    <dim>56</dim>
                </port>
            </input>
            <output>
                <port id="24">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>56</dim>
                    <dim>56</dim>
                </port>
            </output>
        </layer>
        <layer name="res2a_branch2b" type="Convolution" precision="FP32" id="13">
            <convolution_data stride-x="1" stride-y="1" pad-x="1" pad-y="1" kernel-x="3" kernel-y="3" output="64" group="1"/>
            <input>
                <port id="25">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>56</dim>
                    <dim>56</dim>
                </port>
            </input>
            <output>
                <port id="26">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>56</dim>
                    <dim>56</dim>
                </port>
            </output>
            <weights offset="54528" size="147456"/>
            <biases offset="201984" size="256"/>
        </layer>
        <layer name="res2a_branch2b_relu" type="ReLU" precision="FP32" id="16">
            <input>
                <port id="31">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>56</dim>
                    <dim>56</dim>
                </port>
            </input>
            <output>
                <port id="32">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>56</dim>
                    <dim>56</dim>
                </port>
            </output>
        </layer>
        <layer name="res2a_branch2c" type="Convolution" precision="FP32" id="17">
            <convolution_data stride-x="1" stride-y="1" pad-x="0" pad-y="0" kernel-x="1" kernel-y="1" output="256" group="1"/>
            <input>
                <port id="33">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>56</dim>
                    <dim>56</dim>
                </port>
            </input>
            <output>
                <port id="34">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>56</dim>
                    <dim>56</dim>
                </port>
            </output>
            <weights offset="202240" size="65536"/>
            <biases offset="267776" size="1024"/>
        </layer>
        <layer name="res2a_branch1" type="Convolution" precision="FP32" id="6">
            <convolution_data stride-x="1" stride-y="1" pad-x="0" pad-y="0" kernel-x="1" kernel-y="1" output="256" group="1"/>
            <input>
                <port id="11">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>56</dim>
                    <dim>56</dim>
                </port>
            </input>
            <output>
                <port id="12">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>56</dim>
                    <dim>56</dim>
                </port>
            </output>
            <weights offset="268800" size="65536"/>
            <biases offset="334336" size="1024"/>
        </layer>
        <layer name="res2a" type="Eltwise" precision="FP32" id="20">
            <elementwise_data operation="sum"/>
            <input>
                <port id="39">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>56</dim>
                    <dim>56</dim>
                </port>
                <port id="40">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>56</dim>
                    <dim>56</dim>
                </port>
            </input>
            <output>
                <port id="41">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>56</dim>
                    <dim>56</dim>
                </port>
            </output>
        </layer>
        <layer name="res2a_relu" type="ReLU" precision="FP32" id="21">
            <input>
                <port id="42">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>56</dim>
                    <dim>56</dim>
                </port>
            </input>
            <output>
                <port id="43">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>56</dim>
                    <dim>56</dim>
                </port>
            </output>
        </layer>
        <layer name="res2b_branch2a" type="Convolution" precision="FP32" id="22">
            <convolution_data stride-x="1" stride-y="1" pad-x="0" pad-y="0" kernel-x="1" kernel-y="1" output="64" group="1"/>
            <input>
                <port id="44">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>56</dim>
                    <dim>56</dim>
                </port>
            </input>
            <output>
                <port id="45">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>56</dim>
                    <dim>56</dim>
                </port>
            </output>
            <weights offset="335360" size="65536"/>
            <biases offset="400896" size="256"/>
        </layer>
        <layer name="res2b_branch2a_relu" type="ReLU" precision="FP32" id="25">
            <input>
                <port id="50">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>56</dim>
                    <dim>56</dim>
                </port>
            </input>
            <output>
                <port id="51">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>56</dim>
                    <dim>56</dim>
                </port>
            </output>
        </layer>
        <layer name="res2b_branch2b" type="Convolution" precision="FP32" id="26">
            <convolution_data stride-x="1" stride-y="1" pad-x="1" pad-y="1" kernel-x="3" kernel-y="3" output="64" group="1"/>
            <input>
                <port id="52">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>56</dim>
                    <dim>56</dim>
                </port>
            </input>
            <output>
                <port id="53">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>56</dim>
                    <dim>56</dim>
                </port>
            </output>
            <weights offset="401152" size="147456"/>
            <biases offset="548608" size="256"/>
        </layer> )V0G0N";
        std::string modelE =R"V0G0N(
        <layer name="res2b_branch2b_relu" type="ReLU" precision="FP32" id="29">
            <input>
                <port id="58">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>56</dim>
                    <dim>56</dim>
                </port>
            </input>
            <output>
                <port id="59">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>56</dim>
                    <dim>56</dim>
                </port>
            </output>
        </layer>
        <layer name="res2b_branch2c" type="Convolution" precision="FP32" id="30">
            <convolution_data stride-x="1" stride-y="1" pad-x="0" pad-y="0" kernel-x="1" kernel-y="1" output="256" group="1"/>
            <input>
                <port id="60">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>56</dim>
                    <dim>56</dim>
                </port>
            </input>
            <output>
                <port id="61">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>56</dim>
                    <dim>56</dim>
                </port>
            </output>
            <weights offset="548864" size="65536"/>
            <biases offset="614400" size="1024"/>
        </layer>
        <layer name="res2b" type="Eltwise" precision="FP32" id="33">
            <elementwise_data operation="sum"/>
            <input>
                <port id="66">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>56</dim>
                    <dim>56</dim>
                </port>
                <port id="67">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>56</dim>
                    <dim>56</dim>
                </port>
            </input>
            <output>
                <port id="68">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>56</dim>
                    <dim>56</dim>
                </port>
            </output>
        </layer>
        <layer name="res2b_relu" type="ReLU" precision="FP32" id="34">
            <input>
                <port id="69">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>56</dim>
                    <dim>56</dim>
                </port>
            </input>
            <output>
                <port id="70">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>56</dim>
                    <dim>56</dim>
                </port>
            </output>
        </layer>
        <layer name="pool5" type="Pooling" precision="FP32" id="668">
            <pooling_data kernel-x="56" kernel-y="56" pad-x="0" pad-y="0" stride-x="1" stride-y="1" rounding-type="ceil" pool-method="avg"/>
            <input>
                <port id="1385">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>56</dim>
                    <dim>56</dim>
                </port>
            </input>
            <output>
                <port id="1386">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer name="fc1000" type="FullyConnected" precision="FP32" id="669">
            <fc_data out-size="1000"/>
            <input>
                <port id="1387">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="1388">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </output>
            <weights offset="615424" size="1024000"/>
            <biases offset="1639424" size="4000"/>
        </layer>
        <layer name="prob" type="SoftMax" precision="FP32" id="670">
            <input>
                <port id="1389">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </input>
            <output>
                <port id="1390">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="4" to-port="7"/>
        <edge from-layer="4" from-port="8" to-layer="5" to-port="9"/>
        <edge from-layer="5" from-port="10" to-layer="6" to-port="11"/>
        <edge from-layer="5" from-port="10" to-layer="9" to-port="17"/>
        <edge from-layer="12" from-port="24" to-layer="13" to-port="25"/>
        <edge from-layer="16" from-port="32" to-layer="17" to-port="33"/>
        <edge from-layer="6" from-port="12" to-layer="20" to-port="39"/>
        <edge from-layer="9" from-port="18" to-layer="12" to-port="23"/>
        <edge from-layer="13" from-port="26" to-layer="16" to-port="31"/>
        <edge from-layer="17" from-port="34" to-layer="20" to-port="40"/>
        <edge from-layer="20" from-port="41" to-layer="21" to-port="42"/>
        <edge from-layer="21" from-port="43" to-layer="22" to-port="44"/>
        <edge from-layer="25" from-port="51" to-layer="26" to-port="52"/>
        <edge from-layer="29" from-port="59" to-layer="30" to-port="60"/>
        <edge from-layer="21" from-port="43" to-layer="33" to-port="66"/>
        <edge from-layer="22" from-port="45" to-layer="25" to-port="50"/>
        <edge from-layer="26" from-port="53" to-layer="29" to-port="58"/>
        <edge from-layer="30" from-port="61" to-layer="33" to-port="67"/>
        <edge from-layer="33" from-port="68" to-layer="34" to-port="69"/>
        <edge from-layer="34" from-port="70" to-layer="668" to-port="1385"/>
        <edge from-layer="668" from-port="1386" to-layer="669" to-port="1387"/>
        <edge from-layer="669" from-port="1388" to-layer="670" to-port="1389"/>
    </edges>
    <pre-process reference-layer-name="input" mean-precision="FP16">
        <channel id="0">
            <mean value="104.00698793"/>
        </channel>
        <channel id="1">
            <mean value="116.66876762"/>
        </channel>
        <channel id="2">
            <mean value="122.67891434"/>
        </channel>
    </pre-process>
</net>
)V0G0N";

    std::string model = modelB + modelE;
    InferenceEngine::CNNNetReader net_reader;
    ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

    InferenceEngine::TBlob<uint8_t> *weights = new InferenceEngine::TBlob<uint8_t>({ InferenceEngine::Precision::U8, {1643424}, InferenceEngine::C });
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));
    InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr = InferenceEngine::TBlob<uint8_t>::Ptr(weights);

    net_reader.SetWeights(weights_ptr);

    MKLDNNPlugin::MKLDNNExecNetwork::Ptr execNetwork(new MKLDNNPlugin::MKLDNNExecNetwork(net_reader.getNetwork(), {}, {}));
    InferenceEngine::InputsDataMap _networkInputs = net_reader.getNetwork().getInputsInfo();
    InferenceEngine::OutputsDataMap _networkOutputs = net_reader.getNetwork().getOutputsInfo();
    execNetwork->setNetworkInputs(_networkInputs);
    execNetwork->setNetworkOutputs(_networkOutputs);
    InferenceEngine::IInferRequest::Ptr inferRequest;
    execNetwork->CreateInferRequest(inferRequest);

    InferenceEngine::TensorDesc desc(InferenceEngine::Precision::FP32, {1, 3, 224, 224}, InferenceEngine::NCHW);
    InferenceEngine::Blob::Ptr src = InferenceEngine::make_shared_blob<float>(desc);
    src->allocate();
    fill_data(src->buffer(), src->size());

    InferenceEngine::ResponseDesc resp;

    InferenceEngine::StatusCode sts = inferRequest->SetBlob("input", src, &resp);
    ASSERT_EQ(InferenceEngine::OK, sts) << resp.msg;

    InferenceEngine::OutputsDataMap out = net_reader.getNetwork().getOutputsInfo();

    std::pair<std::string, InferenceEngine::DataPtr> item = *out.begin();

    InferenceEngine::TBlob<float>::Ptr output;
    output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
    output->allocate();

    sts = inferRequest->SetBlob(item.first.c_str(), output, &resp);
    ASSERT_EQ(InferenceEngine::OK, sts) << resp.msg;

    sts = inferRequest->Infer(&resp);
    ASSERT_EQ(InferenceEngine::OK, sts) << resp.msg;
}

TEST_F(MKLDNNGraphStructureTests, TestConcatAfterConcat) {
    std::string model = R"V0G0N(
<net batch="1" name="model" version="2">
	<layers>
		<layer id="0" name="data" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>20</dim>
					<dim>20</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="data2" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>4</dim>
					<dim>20</dim>
					<dim>20</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="data3" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>2</dim>
					<dim>20</dim>
					<dim>20</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="Concat1" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>20</dim>
					<dim>20</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>2</dim>
					<dim>20</dim>
					<dim>20</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>5</dim>
					<dim>20</dim>
					<dim>20</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="Concat2" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>4</dim>
					<dim>20</dim>
					<dim>20</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>5</dim>
					<dim>20</dim>
					<dim>20</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>9</dim>
					<dim>20</dim>
					<dim>20</dim>
				</port>
			</output>
		</layer>
        <layer name="pool" type="Pooling" precision="FP32" id="5">
            <pooling_data kernel-x="20" kernel-y="20" pad-x="0" pad-y="0" stride-x="1" stride-y="1" rounding-type="ceil" pool-method="avg"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>9</dim>
                    <dim>20</dim>
                    <dim>20</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>9</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
		<edge from-layer="2" from-port="0" to-layer="3" to-port="1"/>
		<edge from-layer="1" from-port="0" to-layer="4" to-port="0"/>
		<edge from-layer="3" from-port="2" to-layer="4" to-port="1"/>
		<edge from-layer="4" from-port="2" to-layer="5" to-port="0"/>
	</edges>
</net>
)V0G0N";

    InferenceEngine::CNNNetReader net_reader;
    ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));
    MKLDNNPlugin::MKLDNNExecNetwork::Ptr execNetwork(new MKLDNNPlugin::MKLDNNExecNetwork(net_reader.getNetwork(), {}, {}));
    InferenceEngine::InputsDataMap _networkInputs = net_reader.getNetwork().getInputsInfo();
    InferenceEngine::OutputsDataMap _networkOutputs = net_reader.getNetwork().getOutputsInfo();
    execNetwork->setNetworkInputs(_networkInputs);
    execNetwork->setNetworkOutputs(_networkOutputs);
    InferenceEngine::IInferRequest::Ptr inferRequest;
    execNetwork->CreateInferRequest(inferRequest);

    InferenceEngine::TensorDesc desc1(InferenceEngine::Precision::FP32, {1, 3, 20, 20}, InferenceEngine::NCHW);
    InferenceEngine::Blob::Ptr src1 = InferenceEngine::make_shared_blob<float>(desc1);
    src1->allocate();
    fill_data(src1->buffer(), src1->size());

    InferenceEngine::TensorDesc desc2(InferenceEngine::Precision::FP32, {1, 4, 20, 20}, InferenceEngine::NCHW);
    InferenceEngine::Blob::Ptr src2 = InferenceEngine::make_shared_blob<float>(desc2);
    src2->allocate();
    fill_data(src2->buffer(), src2->size());

    InferenceEngine::TensorDesc desc3(InferenceEngine::Precision::FP32, {1, 2, 20, 20}, InferenceEngine::NCHW);
    InferenceEngine::Blob::Ptr src3 = InferenceEngine::make_shared_blob<float>(desc3);
    src3->allocate();
    fill_data(src3->buffer(), src3->size());

    InferenceEngine::ResponseDesc resp;

    InferenceEngine::StatusCode sts = inferRequest->SetBlob("data1", src1, &resp);
    sts = inferRequest->SetBlob("data2", src2, &resp);
    sts = inferRequest->SetBlob("data3", src3, &resp);
    ASSERT_EQ(InferenceEngine::OK, sts) << resp.msg;

    InferenceEngine::OutputsDataMap out = net_reader.getNetwork().getOutputsInfo();

    std::pair<std::string, InferenceEngine::DataPtr> item = *out.begin();

    InferenceEngine::TBlob<float>::Ptr output;
    output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
    output->allocate();

    sts = inferRequest->SetBlob(item.first.c_str(), output, &resp);
    ASSERT_EQ(InferenceEngine::OK, sts) << resp.msg;

    sts = inferRequest->Infer(&resp);
    ASSERT_EQ(InferenceEngine::OK, sts) << resp.msg;

//    compare(*output, *src);
}

TEST_F(MKLDNNGraphStructureTests, Test2ConcatFromConcat) {
    std::string model = R"V0G0N(
<net batch="1" name="model" version="2">
	<layers>
		<layer id="0" name="data1" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="data2" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>4</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="data3" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>2</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="data4" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>1</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="Concat0" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>2</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>5</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="5" name="Concat1" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>5</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>4</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>9</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="6" name="Concat2" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>5</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>1</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>6</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="4" to-port="0"/>
		<edge from-layer="2" from-port="0" to-layer="4" to-port="1"/>
		<edge from-layer="1" from-port="0" to-layer="5" to-port="1"/>
		<edge from-layer="4" from-port="2" to-layer="5" to-port="0"/>
		<edge from-layer="3" from-port="0" to-layer="6" to-port="1"/>
		<edge from-layer="4" from-port="2" to-layer="6" to-port="0"/>
	</edges>
</net>
)V0G0N";

    InferenceEngine::CNNNetReader net_reader;
    ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));
    MKLDNNPlugin::MKLDNNExecNetwork::Ptr execNetwork(new MKLDNNPlugin::MKLDNNExecNetwork(net_reader.getNetwork(), {}, {}));
    InferenceEngine::InputsDataMap _networkInputs = net_reader.getNetwork().getInputsInfo();
    InferenceEngine::OutputsDataMap _networkOutputs = net_reader.getNetwork().getOutputsInfo();
    execNetwork->setNetworkInputs(_networkInputs);
    execNetwork->setNetworkOutputs(_networkOutputs);
    InferenceEngine::IInferRequest::Ptr inferRequest;
    execNetwork->CreateInferRequest(inferRequest);

    InferenceEngine::TensorDesc desc1(InferenceEngine::Precision::FP32, {1, 3, 2, 2}, InferenceEngine::NCHW);
    InferenceEngine::Blob::Ptr src1 = InferenceEngine::make_shared_blob<float>(desc1);
    src1->allocate();
    fill_data(src1->buffer(), src1->size());

    InferenceEngine::TensorDesc desc2(InferenceEngine::Precision::FP32, {1, 4, 2, 2}, InferenceEngine::NCHW);
    InferenceEngine::Blob::Ptr src2 = InferenceEngine::make_shared_blob<float>(desc2);
    src2->allocate();
    fill_data(src2->buffer(), src2->size());

    InferenceEngine::TensorDesc desc3(InferenceEngine::Precision::FP32, {1, 2, 2, 2}, InferenceEngine::NCHW);
    InferenceEngine::Blob::Ptr src3 = InferenceEngine::make_shared_blob<float>(desc3);
    src3->allocate();
    fill_data(src3->buffer(), src3->size());

    InferenceEngine::TensorDesc desc4(InferenceEngine::Precision::FP32, {1, 1, 2, 2}, InferenceEngine::NCHW);
    InferenceEngine::Blob::Ptr src4 = InferenceEngine::make_shared_blob<float>(desc4);
    src4->allocate();
    fill_data(src4->buffer(), src4->size());

    InferenceEngine::ResponseDesc resp;

    InferenceEngine::StatusCode sts = inferRequest->SetBlob("data1", src1, &resp);
    ASSERT_EQ(InferenceEngine::OK, sts) << resp.msg;
    sts = inferRequest->SetBlob("data2", src2, &resp);
    ASSERT_EQ(InferenceEngine::OK, sts) << resp.msg;
    sts = inferRequest->SetBlob("data3", src3, &resp);
    ASSERT_EQ(InferenceEngine::OK, sts) << resp.msg;
    sts = inferRequest->SetBlob("data4", src4, &resp);
    ASSERT_EQ(InferenceEngine::OK, sts) << resp.msg;

    InferenceEngine::OutputsDataMap out = net_reader.getNetwork().getOutputsInfo();

    std::vector<InferenceEngine::TBlob<float>::Ptr> outputs;
    std::vector<InferenceEngine::TBlob<float>::Ptr> refOutputs;
    for (const auto& it : out) {
        InferenceEngine::TBlob<float>::Ptr output;
        output = InferenceEngine::make_shared_blob<float>(it.second->getTensorDesc());
        output->allocate();
        outputs.push_back(output);

        InferenceEngine::TBlob<float>::Ptr refOutput;
        refOutput = InferenceEngine::make_shared_blob<float>(it.second->getTensorDesc());
        refOutput->allocate();

        float * refData = refOutput->buffer().as<float *>();
        size_t ref_idx = 0;
        if (it.first == "Concat1") {
            float *srcData = src1->buffer().as<float *>();
            for (size_t i = 0; i < src1->size(); i++, ref_idx++) {
                refData[ref_idx] = srcData[i];
            }
            srcData = src3->buffer().as<float *>();
            for (size_t i = 0; i < src3->size(); i++, ref_idx++) {
                refData[ref_idx] = srcData[i];
            }
            srcData = src2->buffer().as<float *>();
            for (size_t i = 0; i < src2->size(); i++, ref_idx++) {
                refData[ref_idx] = srcData[i];
            }


        } else if (it.first == "Concat2") {
            float *srcData = src1->buffer().as<float *>();
            for (size_t i = 0; i < src1->size(); i++, ref_idx++) {
                refData[ref_idx] = srcData[i];
            }
            srcData = src3->buffer().as<float *>();
            for (size_t i = 0; i < src3->size(); i++, ref_idx++) {
                refData[ref_idx] = srcData[i];
            }
            srcData = src4->buffer().as<float *>();
            for (size_t i = 0; i < src4->size(); i++, ref_idx++) {
                refData[ref_idx] = srcData[i];
            }

        }
        refOutputs.push_back(refOutput);

        sts = inferRequest->SetBlob(it.first.c_str(), output, &resp);
        ASSERT_EQ(InferenceEngine::OK, sts) << resp.msg;
    }

    sts = inferRequest->Infer(&resp);
    ASSERT_EQ(InferenceEngine::OK, sts) << resp.msg;

    for (size_t i = 0; i < outputs.size(); i++) {
        compare(*outputs[i], *refOutputs[i]);
    }
}

TEST_F(MKLDNNGraphStructureTests, TestResultsAfterGroupedConvWithStrides) {
    std::string model = R"V0G0N(
<net name="net" version="2" batch="1">
    <layers>
        <layer name="data" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>24</dim>
                    <dim>80</dim>
                    <dim>80</dim>
                </port>
            </output>
        </layer>
        <layer name="conv1_1_conv" type="Convolution" precision="FP32" id="2">
            <data dilation-x="1" dilation-y="1" group="6" kernel-x="3" kernel-y="3" output="24" pad-x="1" pad-y="1" stride="1,1,1,1" stride-x="1" stride-y="1"/>
            <input>
                <port id="2">
                    <dim>1</dim>
                    <dim>24</dim>
                    <dim>80</dim>
                    <dim>80</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>24</dim>
                    <dim>80</dim>
                    <dim>80</dim>
                </port>
            </output>
            <weights offset="0" size="3456"/>
            <biases offset="3456" size="96"/>
        </layer>
        <layer name="conv1_1_neg" type="Power" precision="FP32" id="3">
            <power_data power="1" scale="-1" shift="0"/>
            <input>
                <port id="4">
                    <dim>1</dim>
                    <dim>24</dim>
                    <dim>80</dim>
                    <dim>80</dim>
                </port>
            </input>
            <output>
                <port id="5">
                    <dim>1</dim>
                    <dim>24</dim>
                    <dim>80</dim>
                    <dim>80</dim>
                </port>
            </output>
        </layer>
        <layer name="conv1_1_concat" type="Concat" precision="FP32" id="4">
            <concat_data axis="1"/>
            <input>
                <port id="6">
                    <dim>1</dim>
                    <dim>24</dim>
                    <dim>80</dim>
                    <dim>80</dim>
                </port>
                <port id="7">
                    <dim>1</dim>
                    <dim>24</dim>
                    <dim>80</dim>
                    <dim>80</dim>
                </port>
            </input>
            <output>
                <port id="8">
                    <dim>1</dim>
                    <dim>48</dim>
                    <dim>80</dim>
                    <dim>80</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="2"/>
        <edge from-layer="2" from-port="3" to-layer="3" to-port="4"/>
        <edge from-layer="2" from-port="3" to-layer="4" to-port="6"/>
        <edge from-layer="3" from-port="5" to-layer="4" to-port="7"/>
    </edges>
</net>
)V0G0N";

    InferenceEngine::CNNNetReader net_reader;
    ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

    InferenceEngine::TBlob<uint8_t> *weights = new InferenceEngine::TBlob<uint8_t>({ InferenceEngine::Precision::U8, {3552}, InferenceEngine::C });
    weights->allocate();
    float * data = weights->buffer();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));
    InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr = InferenceEngine::TBlob<uint8_t>::Ptr(weights);

    net_reader.SetWeights(weights_ptr);

    MKLDNNGraphTestClass graph;
    graph.CreateGraph(net_reader.getNetwork());

    InferenceEngine::TensorDesc desc(InferenceEngine::Precision::FP32, {1, 24, 80, 80}, InferenceEngine::NCHW);
    InferenceEngine::Blob::Ptr src = InferenceEngine::make_shared_blob<float>(desc);
    src->allocate();
    fill_data((float *) src->buffer(), src->size());

    InferenceEngine::BlobMap srcs;
    srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("data", src));

    InferenceEngine::OutputsDataMap out = net_reader.getNetwork().getOutputsInfo();

    InferenceEngine::BlobMap outputBlobs;
    std::pair<std::string, InferenceEngine::DataPtr> item = *out.begin();

    InferenceEngine::TBlob<float>::Ptr refOutput;
    refOutput = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
    refOutput->allocate();
    outputBlobs[item.first] = refOutput;

    graph.Infer(srcs, outputBlobs);

    // Compare for batch2
    net_reader.getNetwork().setBatchSize(2);
    graph.CreateGraph(net_reader.getNetwork());
    desc = InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32, {2, 24, 80, 80}, InferenceEngine::NCHW);

    InferenceEngine::Blob::Ptr srcBatch = InferenceEngine::make_shared_blob<float>(desc);
    srcBatch->allocate();
    data = srcBatch->buffer().as<float *>();
    float *originData = src->buffer().as<float *>();
    for(size_t b = 0; b < 2; b++) {
        for (size_t i = 0; i < src->size(); i++) {
            data[srcBatch->getTensorDesc().offset(b*src->size() + i)] = originData[src->getTensorDesc().offset(i)];
        }
    }

    srcs.clear();
    srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("data", srcBatch));
    out = net_reader.getNetwork().getOutputsInfo();

    outputBlobs.clear();
    item = *out.begin();

    InferenceEngine::TBlob<float>::Ptr output;
    output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
    output->allocate();
    outputBlobs[item.first] = output;

    graph.Infer(srcs, outputBlobs);
    InferenceEngine::TBlob<float>::Ptr dstOut = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
    dstOut->allocate();
    data = dstOut->buffer().as<float *>();
    originData = refOutput->buffer().as<float *>();
    for(size_t b = 0; b < 2; b++) {
        for (size_t i = 0; i < refOutput->size(); i++) {
            data[dstOut->getTensorDesc().offset(b*refOutput->size() + i)] = originData[refOutput->getTensorDesc().offset(i)];
        }
    }

    compare(*output, *dstOut);
}

TEST_F(MKLDNNGraphStructureTests, TestLoadTopologyWithConstLayer) {
    std::string model = R"V0G0N(
<net batch="1" name="model" version="2">
	<layers>
		<layer id="0" name="data" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>20</dim>
					<dim>20</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="data1" precision="FP32" type="Const">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>4</dim>
					<dim>20</dim>
					<dim>20</dim>
				</port>
			</output>
			<blobs>
				<custom offset="0" size="6400"/>
			</blobs>
		</layer>
		<layer id="3" name="Concat1" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>20</dim>
					<dim>20</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>4</dim>
					<dim>20</dim>
					<dim>20</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>7</dim>
					<dim>20</dim>
					<dim>20</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
		<edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>
	</edges>
</net>
)V0G0N";

    InferenceEngine::CNNNetReader net_reader;
    ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

    InferenceEngine::TBlob<uint8_t> *weights = new InferenceEngine::TBlob<uint8_t>({ InferenceEngine::Precision::U8, {6400}, InferenceEngine::C });
    weights->allocate();
    float * data = weights->buffer();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));
    InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr = InferenceEngine::TBlob<uint8_t>::Ptr(weights);

    net_reader.SetWeights(weights_ptr);
    MKLDNNPlugin::MKLDNNExecNetwork::Ptr execNetwork(new MKLDNNPlugin::MKLDNNExecNetwork(net_reader.getNetwork(), {}, {}));
    InferenceEngine::InputsDataMap _networkInputs = net_reader.getNetwork().getInputsInfo();
    InferenceEngine::OutputsDataMap _networkOutputs = net_reader.getNetwork().getOutputsInfo();
    execNetwork->setNetworkInputs(_networkInputs);
    execNetwork->setNetworkOutputs(_networkOutputs);
    InferenceEngine::IInferRequest::Ptr inferRequest;
    execNetwork->CreateInferRequest(inferRequest);

    InferenceEngine::TensorDesc desc1(InferenceEngine::Precision::FP32, {1, 3, 20, 20}, InferenceEngine::NCHW);
    InferenceEngine::Blob::Ptr src1 = InferenceEngine::make_shared_blob<float>(desc1);
    src1->allocate();
    fill_data(src1->buffer(), src1->size());

    InferenceEngine::ResponseDesc resp;

    InferenceEngine::StatusCode sts = inferRequest->SetBlob("data", src1, &resp);
    ASSERT_EQ(InferenceEngine::OK, sts) << resp.msg;

    InferenceEngine::OutputsDataMap out = net_reader.getNetwork().getOutputsInfo();

    std::pair<std::string, InferenceEngine::DataPtr> item = *out.begin();

    InferenceEngine::TBlob<float>::Ptr output;
    output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
    output->allocate();

    sts = inferRequest->SetBlob(item.first.c_str(), output, &resp);
    ASSERT_EQ(InferenceEngine::OK, sts) << resp.msg;

    sts = inferRequest->Infer(&resp);
    ASSERT_EQ(InferenceEngine::OK, sts) << resp.msg;
}

TEST_F(MKLDNNGraphStructureTests, TestLoadTopologyWithEltwiseBeforeConcat) {
    std::string model = R"V0G0N(
<net batch="1" name="model" version="2">
	<layers>
		<layer id="0" name="data" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>20</dim>
					<dim>20</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="data1" precision="FP32" type="Const">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>20</dim>
					<dim>20</dim>
				</port>
			</output>
			<blobs>
				<custom offset="0" size="4800"/>
			</blobs>
		</layer>
		<layer id="2" name="data2" precision="FP32" type="Const">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>1</dim>
					<dim>20</dim>
					<dim>20</dim>
				</port>
			</output>
			<blobs>
				<custom offset="4800" size="1600"/>
			</blobs>
		</layer>
        <layer name="Eltwise1" type="Eltwise" id="3" precision="FP32">
			<data operation="sum" />
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>20</dim>
					<dim>20</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>3</dim>
					<dim>20</dim>
					<dim>20</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>3</dim>
					<dim>20</dim>
					<dim>20</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="Concat1" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1</dim>
					<dim>20</dim>
					<dim>20</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>3</dim>
					<dim>20</dim>
					<dim>20</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>4</dim>
					<dim>20</dim>
					<dim>20</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
		<edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>
		<edge from-layer="3" from-port="2" to-layer="4" to-port="1"/>
		<edge from-layer="2" from-port="0" to-layer="4" to-port="0"/>
	</edges>
</net>
)V0G0N";

    InferenceEngine::CNNNetReader net_reader;
    ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

    InferenceEngine::TBlob<uint8_t> *weights = new InferenceEngine::TBlob<uint8_t>({ InferenceEngine::Precision::U8, {6400}, InferenceEngine::C });
    weights->allocate();
    float * data = weights->buffer();
    for (size_t i = 0; i < 1200; i++) {
        data[i] = 3;
    }
    for (size_t i = 1200; i < 1600; i++) {
        data[i] = 4;
    }
    InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr = InferenceEngine::TBlob<uint8_t>::Ptr(weights);

    net_reader.SetWeights(weights_ptr);
    MKLDNNPlugin::MKLDNNExecNetwork::Ptr execNetwork(new MKLDNNPlugin::MKLDNNExecNetwork(net_reader.getNetwork(), {}, {}));
    InferenceEngine::InputsDataMap _networkInputs = net_reader.getNetwork().getInputsInfo();
    InferenceEngine::OutputsDataMap _networkOutputs = net_reader.getNetwork().getOutputsInfo();
    execNetwork->setNetworkInputs(_networkInputs);
    execNetwork->setNetworkOutputs(_networkOutputs);
    InferenceEngine::IInferRequest::Ptr inferRequest;
    execNetwork->CreateInferRequest(inferRequest);

    InferenceEngine::TensorDesc desc1(InferenceEngine::Precision::FP32, {1, 3, 20, 20}, InferenceEngine::NCHW);
    InferenceEngine::Blob::Ptr src1 = InferenceEngine::make_shared_blob<float>(desc1);
    src1->allocate();
    data = src1->buffer();
    for (size_t i = 0; i < 1200; i++) {
        data[i] = 1;
    }

    InferenceEngine::ResponseDesc resp;

    InferenceEngine::StatusCode sts = inferRequest->SetBlob("data", src1, &resp);
    ASSERT_EQ(InferenceEngine::OK, sts) << resp.msg;

    InferenceEngine::OutputsDataMap out = net_reader.getNetwork().getOutputsInfo();

    std::pair<std::string, InferenceEngine::DataPtr> item = *out.begin();

    InferenceEngine::TBlob<float>::Ptr output;
    output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
    output->allocate();

    sts = inferRequest->SetBlob(item.first.c_str(), output, &resp);
    ASSERT_EQ(InferenceEngine::OK, sts) << resp.msg;

    sts = inferRequest->Infer(&resp);
    ASSERT_EQ(InferenceEngine::OK, sts) << resp.msg;

    auto *res_ptr = output->buffer().as<float*>();
    size_t res_size = output->size();

    for (size_t i = 0; i < res_size; i++) {
        ASSERT_NEAR(res_ptr[i], 4, 0.01f);
    }

    MKLDNNGraphTestClass graph;
    graph.CreateGraph(net_reader.getNetwork());

    size_t reorders_num = 0;
    auto& nodes = graph.getNodes();
    for (auto &node : nodes) {
        if (node->getType() == MKLDNNPlugin::Reorder) {
            reorders_num++;
            ASSERT_EQ(MKLDNNPlugin::Input, node->getParentEdgeAt(0)->getParent()->getType());
            ASSERT_EQ(MKLDNNPlugin::Eltwise, node->getChildEdgeAt(0)->getChild()->getType());
        }
    }
    ASSERT_EQ(reorders_num, 0);
}
TEST_F(MKLDNNGraphStructureTests, TestNoRedundantReordersRmnet_SSSSD) {
    std::string model = R"V0G0N(
<?xml version="1.0" ?>
<net batch="1" name="model" version="2">
	<layers>
		<layer id="26" name="data" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>320</dim>
					<dim>544</dim>
				</port>
			</output>
		</layer>
		<layer id="9" name="Mul_115/Fused_Mul_157/FusedScaleShift_204" precision="FP32" type="ScaleShift">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>320</dim>
					<dim>544</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>3</dim>
					<dim>320</dim>
					<dim>544</dim>
				</port>
			</output>
			<blobs>
				<weights offset="7180" size="12"/>
				<biases offset="2528" size="12"/>
			</blobs>
		</layer>
		<layer id="51" name="init_block1/dim_inc/conv" precision="FP32" type="Convolution">
			<data dilation-x="1" dilation-y="1" group="1" kernel-x="3" kernel-y="3" output="32" pad-x="1" pad-y="1" stride="1,1,2,2" stride-x="2" stride-y="2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>320</dim>
					<dim>544</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>32</dim>
					<dim>160</dim>
					<dim>272</dim>
				</port>
			</output>
			<blobs>
				<weights offset="3596" size="3456"/>
				<biases offset="8536" size="128"/>
			</blobs>
		</layer>
		<layer id="43" name="init_block1/dim_inc/fn" precision="FP32" type="ReLU">
			<data engine="caffe.ReLUParameter.DEFAULT" negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>160</dim>
					<dim>272</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
					<dim>160</dim>
					<dim>272</dim>
				</port>
			</output>
		</layer>
		<layer id="11" name="bottleneck1_1/dim_red/conv" precision="FP32" type="Convolution">
			<data dilation-x="1" dilation-y="1" group="1" kernel-x="1" kernel-y="1" output="8" pad-x="0" pad-y="0" stride="1,1,1,1" stride-x="1" stride-y="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>160</dim>
					<dim>272</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>8</dim>
					<dim>160</dim>
					<dim>272</dim>
				</port>
			</output>
			<blobs>
				<weights offset="32" size="1024"/>
				<biases offset="1472" size="32"/>
			</blobs>
		</layer>
		<layer id="22" name="bottleneck1_1/dim_red/fn" precision="FP32" type="ELU">
			<data alpha="1.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>8</dim>
					<dim>160</dim>
					<dim>272</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>8</dim>
					<dim>160</dim>
					<dim>272</dim>
				</port>
			</output>
		</layer>
		<layer id="34" name="bottleneck1_1/inner/dw1/conv" precision="FP32" type="Convolution">
			<data dilation-x="1" dilation-y="1" group="8" kernel-x="3" kernel-y="3" output="8" pad-x="1" pad-y="1" stride="1,1,1,1" stride-x="1" stride-y="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>8</dim>
					<dim>160</dim>
					<dim>272</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>8</dim>
					<dim>160</dim>
					<dim>272</dim>
				</port>
			</output>
			<blobs>
				<weights offset="8248" size="288"/>
				<biases offset="3564" size="32"/>
			</blobs>
		</layer>
		<layer id="39" name="bottleneck1_1/inner/dw1/fn" precision="FP32" type="ELU">
			<data alpha="1.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>8</dim>
					<dim>160</dim>
					<dim>272</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>8</dim>
					<dim>160</dim>
					<dim>272</dim>
				</port>
			</output>
		</layer>
		<layer id="18" name="bottleneck1_1/dim_inc/conv" precision="FP32" type="Convolution">
			<data dilation-x="1" dilation-y="1" group="1" kernel-x="1" kernel-y="1" output="32" pad-x="0" pad-y="0" stride="1,1,1,1" stride-x="1" stride-y="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>8</dim>
					<dim>160</dim>
					<dim>272</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>32</dim>
					<dim>160</dim>
					<dim>272</dim>
				</port>
			</output>
			<blobs>
				<weights offset="2540" size="1024"/>
				<biases offset="7052" size="128"/>
			</blobs>
		</layer>
		<layer id="32" name="bottleneck1_1/add" precision="FP32" type="Eltwise">
			<data coeff="" operation="sum"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>160</dim>
					<dim>272</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
					<dim>160</dim>
					<dim>272</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>32</dim>
					<dim>160</dim>
					<dim>272</dim>
				</port>
			</output>
		</layer>
		<layer id="7" name="bottleneck1_1/fn" precision="FP32" type="ELU">
			<data alpha="1.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>160</dim>
					<dim>272</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
					<dim>160</dim>
					<dim>272</dim>
				</port>
			</output>
		</layer>
		<layer id="29" name="bottleneck1_2/dim_red/conv" precision="FP32" type="Convolution">
			<data dilation-x="1" dilation-y="1" group="1" kernel-x="1" kernel-y="1" output="8" pad-x="0" pad-y="0" stride="1,1,1,1" stride-x="1" stride-y="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>160</dim>
					<dim>272</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>8</dim>
					<dim>160</dim>
					<dim>272</dim>
				</port>
			</output>
			<blobs>
				<weights offset="7192" size="1024"/>
				<biases offset="0" size="32"/>
			</blobs>
		</layer>
		<layer id="45" name="bottleneck1_2/dim_red/fn" precision="FP32" type="ELU">
			<data alpha="1.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>8</dim>
					<dim>160</dim>
					<dim>272</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>8</dim>
					<dim>160</dim>
					<dim>272</dim>
				</port>
			</output>
		</layer>
		<layer id="41" name="bottleneck1_2/inner/dw1/conv" precision="FP32" type="Convolution">
			<data dilation-x="1" dilation-y="1" group="8" kernel-x="3" kernel-y="3" output="8" pad-x="1" pad-y="1" stride="1,1,1,1" stride-x="1" stride-y="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>8</dim>
					<dim>160</dim>
					<dim>272</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>8</dim>
					<dim>160</dim>
					<dim>272</dim>
				</port>
			</output>
			<blobs>
				<weights offset="1184" size="288"/>
				<biases offset="8216" size="32"/>
			</blobs>
		</layer>
		<layer id="25" name="bottleneck1_2/inner/dw1/fn" precision="FP32" type="ELU">
			<data alpha="1.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>8</dim>
					<dim>160</dim>
					<dim>272</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>8</dim>
					<dim>160</dim>
					<dim>272</dim>
				</port>
			</output>
		</layer>
		<layer id="6" name="bottleneck1_2/dim_inc/conv" precision="FP32" type="Convolution">
			<data dilation-x="1" dilation-y="1" group="1" kernel-x="1" kernel-y="1" output="32" pad-x="0" pad-y="0" stride="1,1,1,1" stride-x="1" stride-y="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>8</dim>
					<dim>160</dim>
					<dim>272</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>32</dim>
					<dim>160</dim>
					<dim>272</dim>
				</port>
			</output>
			<blobs>
				<weights offset="1504" size="1024"/>
				<biases offset="1056" size="128"/>
			</blobs>
		</layer>
		<layer id="44" name="bottleneck1_2/add" precision="FP32" type="Eltwise">
			<data coeff="" operation="sum"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>160</dim>
					<dim>272</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
					<dim>160</dim>
					<dim>272</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>32</dim>
					<dim>160</dim>
					<dim>272</dim>
				</port>
			</output>
		</layer>
		<layer id="49" name="bottleneck1_2/fn" precision="FP32" type="ELU">
			<data alpha="1.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>160</dim>
					<dim>272</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>32</dim>
					<dim>160</dim>
					<dim>272</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="26" from-port="0" to-layer="9" to-port="0"/>
		<edge from-layer="9" from-port="3" to-layer="51" to-port="0"/>
		<edge from-layer="51" from-port="3" to-layer="43" to-port="0"/>
		<edge from-layer="43" from-port="1" to-layer="11" to-port="0"/>
		<edge from-layer="11" from-port="3" to-layer="22" to-port="0"/>
		<edge from-layer="22" from-port="1" to-layer="34" to-port="0"/>
		<edge from-layer="34" from-port="3" to-layer="39" to-port="0"/>
		<edge from-layer="39" from-port="1" to-layer="18" to-port="0"/>
		<edge from-layer="43" from-port="1" to-layer="32" to-port="0"/>
		<edge from-layer="18" from-port="3" to-layer="32" to-port="1"/>
		<edge from-layer="32" from-port="2" to-layer="7" to-port="0"/>
		<edge from-layer="7" from-port="1" to-layer="29" to-port="0"/>
		<edge from-layer="29" from-port="3" to-layer="45" to-port="0"/>
		<edge from-layer="45" from-port="1" to-layer="41" to-port="0"/>
		<edge from-layer="41" from-port="3" to-layer="25" to-port="0"/>
		<edge from-layer="25" from-port="1" to-layer="6" to-port="0"/>
		<edge from-layer="7" from-port="1" to-layer="44" to-port="0"/>
		<edge from-layer="6" from-port="3" to-layer="44" to-port="1"/>
		<edge from-layer="44" from-port="2" to-layer="49" to-port="0"/>
	</edges>
</net>
)V0G0N";
    InferenceEngine::CNNNetReader net_reader;
    ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

    InferenceEngine::TBlob<uint8_t> *weights = new InferenceEngine::TBlob<uint8_t>({ InferenceEngine::Precision::U8, {8664}, InferenceEngine::C });
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));
    InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr = InferenceEngine::TBlob<uint8_t>::Ptr(weights);

    net_reader.SetWeights(weights_ptr);

    MKLDNNGraphTestClass graph;
    graph.CreateGraph(net_reader.getNetwork());

    size_t reorders_num = 0;
    auto& nodes = graph.getNodes();
    for (auto &node : nodes) {
        if (node->getType() == MKLDNNPlugin::Reorder) {
            reorders_num++;
            ASSERT_EQ(MKLDNNPlugin::Output, node->getChildEdgeAt(0)->getChild()->getType());
        }
    }

    ASSERT_EQ(reorders_num, 1);
}

TEST_F(MKLDNNGraphStructureTests, TestFailedPartDPN92) {
    std::string model = R"V0G0N(
<net name="net" version="2" batch="1">
    <layers>
        <layer name="data" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>32</dim>
                    <dim>14</dim>
                    <dim>14</dim>
                </port>
            </output>
        </layer>
        <layer name="data2" type="Input" precision="FP32" id="1">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>28</dim>
                    <dim>28</dim>
                </port>
            </output>
        </layer>
        <layer id="132" name="dpn8_match_conv" precision="FP32" type="Convolution">
			<data dilation-x="1" dilation-y="1" group="1" kernel-x="1" kernel-y="1" output="96" pad-x="0" pad-y="0" stride="1,1,2,2" stride-x="2" stride-y="2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>28</dim>
					<dim>28</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>96</dim>
					<dim>14</dim>
					<dim>14</dim>
				</port>
			</output>
			<blobs>
				<weights offset="0" size="24576"/>
			</blobs>
		</layer>
		<layer id="133" name="dpn8_match_conv_Slice" precision="FP32" type="Slice">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>96</dim>
					<dim>14</dim>
					<dim>14</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>14</dim>
					<dim>14</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>32</dim>
					<dim>14</dim>
					<dim>14</dim>
				</port>
			</output>
		</layer>
		<layer id="145" name="dpn8_conv3" precision="FP32" type="Convolution">
			<data dilation-x="1" dilation-y="1" group="1" kernel-x="1" kernel-y="1" output="72" pad-x="0" pad-y="0" stride="1,1,1,1" stride-x="1" stride-y="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>14</dim>
					<dim>14</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>72</dim>
					<dim>14</dim>
					<dim>14</dim>
				</port>
			</output>
			<blobs>
				<weights offset="24576" size="9216"/>
			</blobs>
		</layer>
		<layer id="146" name="dpn8_conv3_Slice" precision="FP32" type="Slice">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>72</dim>
					<dim>14</dim>
					<dim>14</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>14</dim>
					<dim>14</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>8</dim>
					<dim>14</dim>
					<dim>14</dim>
				</port>
			</output>
		</layer>
		<layer id="147" name="dpn8_elewise" precision="FP32" type="Eltwise">
			<data coeff="" operation="sum"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>14</dim>
					<dim>14</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>14</dim>
					<dim>14</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>64</dim>
					<dim>14</dim>
					<dim>14</dim>
				</port>
			</output>
		</layer>
		<layer id="148" name="dpn8_concat" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>32</dim>
					<dim>14</dim>
					<dim>14</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>8</dim>
					<dim>14</dim>
					<dim>14</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>40</dim>
					<dim>14</dim>
					<dim>14</dim>
				</port>
			</output>
		</layer>
		<layer id="149" name="dpn9_concat_input" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>14</dim>
					<dim>14</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>40</dim>
					<dim>14</dim>
					<dim>14</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>104</dim>
					<dim>14</dim>
					<dim>14</dim>
				</port>
			</output>
		</layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="145" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="132" to-port="0"/>
        <edge from-layer="145" from-port="2" to-layer="146" to-port="0"/>
        <edge from-layer="132" from-port="2" to-layer="133" to-port="0"/>
        <edge from-layer="133" from-port="1" to-layer="147" to-port="0"/>
		<edge from-layer="146" from-port="1" to-layer="147" to-port="1"/>
		<edge from-layer="133" from-port="2" to-layer="148" to-port="0"/>
		<edge from-layer="146" from-port="2" to-layer="148" to-port="1"/>
        <edge from-layer="148" from-port="2" to-layer="149" to-port="1"/>
		<edge from-layer="147" from-port="2" to-layer="149" to-port="0"/>
    </edges>
</net>)V0G0N";

    InferenceEngine::CNNNetReader net_reader;
    ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

    InferenceEngine::TBlob<uint8_t> *weights = new InferenceEngine::TBlob<uint8_t>({ InferenceEngine::Precision::U8, {33792}, InferenceEngine::C });
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));

    InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr = InferenceEngine::TBlob<uint8_t>::Ptr(weights);

    net_reader.SetWeights(weights_ptr);

    MKLDNNGraphTestClass graph;
    graph.CreateGraph(net_reader.getNetwork());

    InferenceEngine::TensorDesc desc(InferenceEngine::Precision::FP32, {1, 32, 14, 14}, InferenceEngine::NCHW);
    InferenceEngine::Blob::Ptr src1 = InferenceEngine::make_shared_blob<float>(desc);
    src1->allocate();
    fill_data((float *) src1->buffer(), src1->size());


    desc = InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32, {1, 64, 28, 28}, InferenceEngine::NCHW);
    InferenceEngine::Blob::Ptr src2 = InferenceEngine::make_shared_blob<float>(desc);
    src2->allocate();
    fill_data((float *) src2->buffer(), src2->size());

    InferenceEngine::BlobMap srcs;
    srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("data", src1));
    srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("data2", src2));

    InferenceEngine::OutputsDataMap out = net_reader.getNetwork().getOutputsInfo();

    InferenceEngine::BlobMap outputBlobs;
    std::pair<std::string, InferenceEngine::DataPtr> item = *out.begin();

    InferenceEngine::TBlob<float>::Ptr output;
    output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
    output->allocate();
    outputBlobs[item.first] = output;

    graph.Infer(srcs, outputBlobs);

    std::vector<float> refDst(output->size());
    auto *data = output->buffer().as<float *>();
    for (size_t i = 0; i < output->size(); i++) {
        refDst[i] = data[output->getTensorDesc().offset(i)];
    }

    // Compare for batch2
    net_reader.getNetwork().setBatchSize(2);
    graph.CreateGraph(net_reader.getNetwork());
    desc = InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32, {2, 32, 14, 14}, InferenceEngine::NCHW);

    InferenceEngine::Blob::Ptr src1Batch = InferenceEngine::make_shared_blob<float>(desc);
    src1Batch->allocate();
    data = src1Batch->buffer().as<float *>();
    auto *originData = src1->buffer().as<float *>();
    for(size_t b = 0; b < 2; b++) {
        for (size_t i = 0; i < src1->size(); i++) {
            data[src1Batch->getTensorDesc().offset(b*src1->size() + i)] = originData[src1->getTensorDesc().offset(i)];
        }
    }

    desc = InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32, {2, 64, 28, 28}, InferenceEngine::NCHW);

    InferenceEngine::Blob::Ptr src2Batch = InferenceEngine::make_shared_blob<float>(desc);
    src2Batch->allocate();
    data = src2Batch->buffer().as<float *>();
    originData = src2->buffer().as<float *>();
    for(size_t b = 0; b < 2; b++) {
        for (size_t i = 0; i < src2->size(); i++) {
            data[src2Batch->getTensorDesc().offset(b*src2->size() + i)] = originData[src2->getTensorDesc().offset(i)];
        }
    }

    srcs.clear();
    srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("data", src1Batch));
    srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("data2", src2Batch));
    out = net_reader.getNetwork().getOutputsInfo();

    outputBlobs.clear();
    item = *out.begin();
    output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
    output->allocate();
    outputBlobs[item.first] = output;

    graph.Infer(srcs, outputBlobs);
    InferenceEngine::TBlob<float>::Ptr dstOut = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
    dstOut->allocate();
    data = dstOut->buffer().as<float *>();
    for(size_t b = 0; b < 2; b++) {
        for (size_t i = 0; i < refDst.size(); i++) {
            data[dstOut->getTensorDesc().offset(b*refDst.size() + i)] = refDst[i];
        }
    }

    compare(*output, *dstOut);
}

TEST_F(MKLDNNGraphStructureTests, TestNoRedundantReordersForXceptionTopology) {
    std::string model = R"V0G0N(
<net batch="1" name="xception" version="2">
	<layers>
		<layer id="1" name="input_1" precision="FP32" type="Input">
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>3</dim>
					<dim>299</dim>
					<dim>299</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="block1_conv1" precision="FP32" type="Convolution">
			<data dilation-x="1" dilation-y="1" group="1" kernel-x="3" kernel-y="3" output="32" pad-x="0" pad-y="0" stride-x="2" stride-y="2"/>
			<input>
				<port id="2">
					<dim>1</dim>
					<dim>3</dim>
					<dim>299</dim>
					<dim>299</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>32</dim>
					<dim>149</dim>
					<dim>149</dim>
				</port>
			</output>
			<weights offset="0" size="3456"/>
			<biases offset="3456" size="128"/>
		</layer>
		<layer id="4" name="block1_conv1_act" precision="FP32" type="ReLU">
			<input>
				<port id="6">
					<dim>1</dim>
					<dim>32</dim>
					<dim>149</dim>
					<dim>149</dim>
				</port>
			</input>
			<output>
				<port id="7">
					<dim>1</dim>
					<dim>32</dim>
					<dim>149</dim>
					<dim>149</dim>
				</port>
			</output>
		</layer>
		<layer id="5" name="block1_conv2" precision="FP32" type="Convolution">
			<data dilation-x="1" dilation-y="1" group="1" kernel-x="3" kernel-y="3" output="64" pad-x="0" pad-y="0" stride-x="1" stride-y="1"/>
			<input>
				<port id="8">
					<dim>1</dim>
					<dim>32</dim>
					<dim>149</dim>
					<dim>149</dim>
				</port>
			</input>
			<output>
				<port id="9">
					<dim>1</dim>
					<dim>64</dim>
					<dim>147</dim>
					<dim>147</dim>
				</port>
			</output>
			<weights offset="3584" size="73728"/>
			<biases offset="77312" size="256"/>
		</layer>
		<layer id="7" name="block1_conv2_act" precision="FP32" type="ReLU">
			<input>
				<port id="12">
					<dim>1</dim>
					<dim>64</dim>
					<dim>147</dim>
					<dim>147</dim>
				</port>
			</input>
			<output>
				<port id="13">
					<dim>1</dim>
					<dim>64</dim>
					<dim>147</dim>
					<dim>147</dim>
				</port>
			</output>
		</layer>
		<layer id="136" name="block2_sepconv1_depth" precision="FP32" type="Convolution">
			<data dilation-x="1" dilation-y="1" group="64" kernel-x="3" kernel-y="3" output="64" pad-x="1" pad-y="1" stride-x="1" stride-y="1"/>
			<input>
				<port id="282">
					<dim>1</dim>
					<dim>64</dim>
					<dim>147</dim>
					<dim>147</dim>
				</port>
			</input>
			<output>
				<port id="283">
					<dim>1</dim>
					<dim>64</dim>
					<dim>147</dim>
					<dim>147</dim>
				</port>
			</output>
			<weights offset="77568" size="2304"/>
		</layer>
		<layer id="137" name="block2_sepconv1_point" precision="FP32" type="Convolution">
			<data dilation-x="1" dilation-y="1" group="1" kernel-x="1" kernel-y="1" output="128" pad-x="0" pad-y="0" stride-x="1" stride-y="1"/>
			<input>
				<port id="284">
					<dim>1</dim>
					<dim>64</dim>
					<dim>147</dim>
					<dim>147</dim>
				</port>
			</input>
			<output>
				<port id="285">
					<dim>1</dim>
					<dim>128</dim>
					<dim>147</dim>
					<dim>147</dim>
				</port>
			</output>
			<weights offset="79872" size="32768"/>
			<biases offset="112640" size="512"/>
		</layer>
		<layer id="10" name="block2_sepconv2_act" precision="FP32" type="ReLU">
			<input>
				<port id="19">
					<dim>1</dim>
					<dim>128</dim>
					<dim>147</dim>
					<dim>147</dim>
				</port>
			</input>
			<output>
				<port id="20">
					<dim>1</dim>
					<dim>128</dim>
					<dim>147</dim>
					<dim>147</dim>
				</port>
			</output>
		</layer>
		<layer id="138" name="block2_sepconv2_depth" precision="FP32" type="Convolution">
			<data dilation-x="1" dilation-y="1" group="128" kernel-x="3" kernel-y="3" output="128" pad-x="1" pad-y="1" stride-x="1" stride-y="1"/>
			<input>
				<port id="286">
					<dim>1</dim>
					<dim>128</dim>
					<dim>147</dim>
					<dim>147</dim>
				</port>
			</input>
			<output>
				<port id="287">
					<dim>1</dim>
					<dim>128</dim>
					<dim>147</dim>
					<dim>147</dim>
				</port>
			</output>
			<weights offset="113152" size="4608"/>
		</layer>
		<layer id="139" name="block2_sepconv2_point" precision="FP32" type="Convolution">
			<data dilation-x="1" dilation-y="1" group="1" kernel-x="1" kernel-y="1" output="128" pad-x="0" pad-y="0" stride-x="1" stride-y="1"/>
			<input>
				<port id="288">
					<dim>1</dim>
					<dim>128</dim>
					<dim>147</dim>
					<dim>147</dim>
				</port>
			</input>
			<output>
				<port id="289">
					<dim>1</dim>
					<dim>128</dim>
					<dim>147</dim>
					<dim>147</dim>
				</port>
			</output>
			<weights offset="117760" size="65536"/>
			<biases offset="183296" size="512"/>
		</layer>
		<layer id="13" name="conv2d_1" precision="FP32" type="Convolution">
			<data dilation-x="1" dilation-y="1" group="1" kernel-x="1" kernel-y="1" output="128" pad-x="0" pad-y="0" stride-x="2" stride-y="2"/>
			<input>
				<port id="15">
					<dim>1</dim>
					<dim>64</dim>
					<dim>147</dim>
					<dim>147</dim>
				</port>
			</input>
			<output>
				<port id="26">
					<dim>1</dim>
					<dim>128</dim>
					<dim>74</dim>
					<dim>74</dim>
				</port>
			</output>
			<weights offset="183808" size="32768"/>
			<biases offset="216576" size="512"/>
		</layer>
		<layer id="14" name="block2_pool" precision="FP32" type="Pooling">
			<data kernel-x="3" kernel-y="3" pad-x="1" pad-y="1" pool-method="max" stride-x="2" stride-y="2"/>
			<input>
				<port id="25">
					<dim>1</dim>
					<dim>128</dim>
					<dim>147</dim>
					<dim>147</dim>
				</port>
			</input>
			<output>
				<port id="28">
					<dim>1</dim>
					<dim>128</dim>
					<dim>74</dim>
					<dim>74</dim>
				</port>
			</output>
		</layer>
		<layer id="16" name="add_1" precision="FP32" type="Eltwise">
			<input>
				<port id="29">
					<dim>1</dim>
					<dim>128</dim>
					<dim>74</dim>
					<dim>74</dim>
				</port>
				<port id="31">
					<dim>1</dim>
					<dim>128</dim>
					<dim>74</dim>
					<dim>74</dim>
				</port>
			</input>
			<output>
				<port id="32">
					<dim>1</dim>
					<dim>128</dim>
					<dim>74</dim>
					<dim>74</dim>
				</port>
			</output>
		</layer>
		<layer id="17" name="block3_sepconv1_act" precision="FP32" type="ReLU">
			<input>
				<port id="33">
					<dim>1</dim>
					<dim>128</dim>
					<dim>74</dim>
					<dim>74</dim>
				</port>
			</input>
			<output>
				<port id="35">
					<dim>1</dim>
					<dim>128</dim>
					<dim>74</dim>
					<dim>74</dim>
				</port>
			</output>
		</layer>
		<layer id="140" name="block3_sepconv1_depth" precision="FP32" type="Convolution">
			<data dilation-x="1" dilation-y="1" group="128" kernel-x="3" kernel-y="3" output="128" pad-x="1" pad-y="1" stride-x="1" stride-y="1"/>
			<input>
				<port id="290">
					<dim>1</dim>
					<dim>128</dim>
					<dim>74</dim>
					<dim>74</dim>
				</port>
			</input>
			<output>
				<port id="291">
					<dim>1</dim>
					<dim>128</dim>
					<dim>74</dim>
					<dim>74</dim>
				</port>
			</output>
			<weights offset="217088" size="4608"/>
		</layer>
		<layer id="141" name="block3_sepconv1_point" precision="FP32" type="Convolution">
			<data dilation-x="1" dilation-y="1" group="1" kernel-x="1" kernel-y="1" output="256" pad-x="0" pad-y="0" stride-x="1" stride-y="1"/>
			<input>
				<port id="292">
					<dim>1</dim>
					<dim>128</dim>
					<dim>74</dim>
					<dim>74</dim>
				</port>
			</input>
			<output>
				<port id="293">
					<dim>1</dim>
					<dim>256</dim>
					<dim>74</dim>
					<dim>74</dim>
				</port>
			</output>
			<weights offset="221696" size="131072"/>
			<biases offset="352768" size="1024"/>
		</layer>
		<layer id="20" name="block3_sepconv2_act" precision="FP32" type="ReLU">
			<input>
				<port id="40">
					<dim>1</dim>
					<dim>256</dim>
					<dim>74</dim>
					<dim>74</dim>
				</port>
			</input>
			<output>
				<port id="41">
					<dim>1</dim>
					<dim>256</dim>
					<dim>74</dim>
					<dim>74</dim>
				</port>
			</output>
		</layer>
		<layer id="142" name="block3_sepconv2_depth" precision="FP32" type="Convolution">
			<data dilation-x="1" dilation-y="1" group="256" kernel-x="3" kernel-y="3" output="256" pad-x="1" pad-y="1" stride-x="1" stride-y="1"/>
			<input>
				<port id="294">
					<dim>1</dim>
					<dim>256</dim>
					<dim>74</dim>
					<dim>74</dim>
				</port>
			</input>
			<output>
				<port id="295">
					<dim>1</dim>
					<dim>256</dim>
					<dim>74</dim>
					<dim>74</dim>
				</port>
			</output>
			<weights offset="353792" size="9216"/>
		</layer>
		<layer id="143" name="block3_sepconv2_point" precision="FP32" type="Convolution">
			<data dilation-x="1" dilation-y="1" group="1" kernel-x="1" kernel-y="1" output="256" pad-x="0" pad-y="0" stride-x="1" stride-y="1"/>
			<input>
				<port id="296">
					<dim>1</dim>
					<dim>256</dim>
					<dim>74</dim>
					<dim>74</dim>
				</port>
			</input>
			<output>
				<port id="297">
					<dim>1</dim>
					<dim>256</dim>
					<dim>74</dim>
					<dim>74</dim>
				</port>
			</output>
			<weights offset="363008" size="262144"/>
			<biases offset="625152" size="1024"/>
		</layer>
		<layer id="23" name="conv2d_2" precision="FP32" type="Convolution">
			<data dilation-x="1" dilation-y="1" group="1" kernel-x="1" kernel-y="1" output="256" pad-x="0" pad-y="0" stride-x="2" stride-y="2"/>
			<input>
				<port id="34">
					<dim>1</dim>
					<dim>128</dim>
					<dim>74</dim>
					<dim>74</dim>
				</port>
			</input>
			<output>
				<port id="47">
					<dim>1</dim>
					<dim>256</dim>
					<dim>37</dim>
					<dim>37</dim>
				</port>
			</output>
			<weights offset="626176" size="131072"/>
			<biases offset="757248" size="1024"/>
		</layer>
		<layer id="24" name="block3_pool" precision="FP32" type="Pooling">
			<data kernel-x="3" kernel-y="3" pad-x="0" pad-y="0" pool-method="max" stride-x="2" stride-y="2"/>
			<input>
				<port id="46">
					<dim>1</dim>
					<dim>256</dim>
					<dim>74</dim>
					<dim>74</dim>
				</port>
			</input>
			<output>
				<port id="49">
					<dim>1</dim>
					<dim>256</dim>
					<dim>37</dim>
					<dim>37</dim>
				</port>
			</output>
		</layer>
		<layer id="26" name="add_2" precision="FP32" type="Eltwise">
			<input>
				<port id="50">
					<dim>1</dim>
					<dim>256</dim>
					<dim>37</dim>
					<dim>37</dim>
				</port>
				<port id="52">
					<dim>1</dim>
					<dim>256</dim>
					<dim>37</dim>
					<dim>37</dim>
				</port>
			</input>
			<output>
				<port id="53">
					<dim>1</dim>
					<dim>256</dim>
					<dim>37</dim>
					<dim>37</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="1" from-port="1" to-layer="2" to-port="2"/>
		<edge from-layer="2" from-port="3" to-layer="4" to-port="6"/>
		<edge from-layer="4" from-port="7" to-layer="5" to-port="8"/>
		<edge from-layer="5" from-port="9" to-layer="7" to-port="12"/>
		<edge from-layer="7" from-port="13" to-layer="13" to-port="15"/>
		<edge from-layer="137" from-port="285" to-layer="10" to-port="19"/>
		<edge from-layer="139" from-port="289" to-layer="14" to-port="25"/>
		<edge from-layer="14" from-port="28" to-layer="16" to-port="29"/>
		<edge from-layer="13" from-port="26" to-layer="16" to-port="31"/>
		<edge from-layer="16" from-port="32" to-layer="17" to-port="33"/>
		<edge from-layer="16" from-port="32" to-layer="23" to-port="34"/>
		<edge from-layer="141" from-port="293" to-layer="20" to-port="40"/>
		<edge from-layer="143" from-port="297" to-layer="24" to-port="46"/>
		<edge from-layer="24" from-port="49" to-layer="26" to-port="50"/>
		<edge from-layer="23" from-port="47" to-layer="26" to-port="52"/>
		<edge from-layer="7" from-port="13" to-layer="136" to-port="282"/>
		<edge from-layer="136" from-port="283" to-layer="137" to-port="284"/>
		<edge from-layer="10" from-port="20" to-layer="138" to-port="286"/>
		<edge from-layer="138" from-port="287" to-layer="139" to-port="288"/>
		<edge from-layer="17" from-port="35" to-layer="140" to-port="290"/>
		<edge from-layer="140" from-port="291" to-layer="141" to-port="292"/>
		<edge from-layer="20" from-port="41" to-layer="142" to-port="294"/>
		<edge from-layer="142" from-port="295" to-layer="143" to-port="296"/>
	</edges>
</net>
)V0G0N";



    InferenceEngine::CNNNetReader net_reader;
    ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

    InferenceEngine::TBlob<uint8_t> *weights = new InferenceEngine::TBlob<uint8_t>({ InferenceEngine::Precision::U8, {758272}, InferenceEngine::C });
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));
    InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr = InferenceEngine::TBlob<uint8_t>::Ptr(weights);

    net_reader.SetWeights(weights_ptr);

    MKLDNNGraphTestClass graph;
    graph.CreateGraph(net_reader.getNetwork());

    size_t reorders_num = 0;
    auto& nodes = graph.getNodes();
    for (auto &node : nodes) {
        if (node->getType() == MKLDNNPlugin::Reorder) {
            reorders_num++;
            ASSERT_EQ(MKLDNNPlugin::Output, node->getChildEdgeAt(0)->getChild()->getType());
        }
    }
    ASSERT_EQ(reorders_num, 1);
}

TEST_F(MKLDNNGraphStructureTests, TestNoRedundantReordersForGrayscaleInput) {
    std::string model = R"V0G0N(
<net batch="1" name="xception" version="4">
	<layers>
		<layer id="1" name="data" precision="FP32" type="Input">
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>1</dim>
					<dim>40</dim>
					<dim>40</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="conv1" precision="FP32" type="Convolution">
			<data auto_pad="same_upper" dilations="1,1" group="1" kernel="3,3" output="32" pads_begin="0,0" pads_end="2,2" strides="1,1"/>
			<input>
				<port id="2">
					<dim>1</dim>
					<dim>1</dim>
					<dim>40</dim>
					<dim>40</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>32</dim>
					<dim>40</dim>
					<dim>40</dim>
				</port>
			</output>
			<weights offset="0" size="1152"/>
			<biases offset="1152" size="128"/>
		</layer>
	</layers>
	<edges>
		<edge from-layer="1" from-port="1" to-layer="2" to-port="2"/>
	</edges>
</net>
)V0G0N";



    InferenceEngine::CNNNetReader net_reader;
    ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

    InferenceEngine::TBlob<uint8_t> *weights = new InferenceEngine::TBlob<uint8_t>({ InferenceEngine::Precision::U8, {1280}, InferenceEngine::C });
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));
    InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr = InferenceEngine::TBlob<uint8_t>::Ptr(weights);

    net_reader.SetWeights(weights_ptr);

    MKLDNNGraphTestClass graph;
    graph.CreateGraph(net_reader.getNetwork());

    size_t reorders_num = 0;
    auto& nodes = graph.getNodes();
    for (auto &node : nodes) {
        if (node->getType() == MKLDNNPlugin::Reorder) {
            reorders_num++;
            ASSERT_EQ(MKLDNNPlugin::Output, node->getChildEdgeAt(0)->getChild()->getType());
        }
    }
    ASSERT_EQ(reorders_num, 1);
}

TEST_F(MKLDNNGraphStructureTests, TestFailedPartPlateRecognitionBarrier0001) {
    std::string model = R"V0G0N(
<net name="net" version="2" batch="1">
    <layers>
        <layer name="data" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>128</dim>
                    <dim>1</dim>
                    <dim>88</dim>
                </port>
            </output>
        </layer>
        <layer id="32" name="conv3_w" precision="FP32" type="Convolution">
			<data dilation-x="1" dilation-y="1" group="1" kernel-x="13" kernel-y="1" output="71" pad-x="6" pad-y="0" stride="1,1,1,1" stride-x="1" stride-y="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>88</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>71</dim>
					<dim>1</dim>
					<dim>88</dim>
				</port>
			</output>
			<blobs>
				<weights offset="0" size="472576"/>
				<biases offset="472576" size="284"/>
			</blobs>
		</layer>
		<layer id="33" name="relu_conv3_w" precision="FP32" type="ReLU">
			<data negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>71</dim>
					<dim>1</dim>
					<dim>88</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>71</dim>
					<dim>1</dim>
					<dim>88</dim>
				</port>
			</output>
		</layer>
		<layer id="34" name="pattern" precision="FP32" type="FullyConnected">
			<data out-size="128"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>71</dim>
					<dim>1</dim>
					<dim>88</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>128</dim>
				</port>
			</output>
			<blobs>
				<weights offset="472860" size="3198976"/>
				<biases offset="3671836" size="512"/>
			</blobs>
		</layer>
		<layer id="35" name="reshape" precision="FP32" type="Reshape">
			<data axis="0" dim="-1,128,1,1" num_axes="-1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
		</layer>
		<layer id="36" name="tile" precision="FP32" type="Tile">
			<data axis="3" tiles="88"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>88</dim>
				</port>
			</output>
		</layer>
		<layer id="37" name="concat" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>71</dim>
					<dim>1</dim>
					<dim>88</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>1</dim>
					<dim>88</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>199</dim>
					<dim>1</dim>
					<dim>88</dim>
				</port>
			</output>
		</layer>
    </layers>
    <edges>
		<edge from-layer="0" from-port="0" to-layer="32" to-port="0"/>
		<edge from-layer="32" from-port="3" to-layer="33" to-port="0"/>
		<edge from-layer="33" from-port="1" to-layer="34" to-port="0"/>
		<edge from-layer="34" from-port="3" to-layer="35" to-port="0"/>
		<edge from-layer="35" from-port="1" to-layer="36" to-port="0"/>
		<edge from-layer="33" from-port="1" to-layer="37" to-port="0"/>
		<edge from-layer="36" from-port="1" to-layer="37" to-port="1"/>
    </edges>
</net>)V0G0N";

    InferenceEngine::CNNNetReader net_reader;
    ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

    InferenceEngine::TBlob<uint8_t> *weights = new InferenceEngine::TBlob<uint8_t>({ InferenceEngine::Precision::U8, {3672348}, InferenceEngine::C });
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));

    InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr = InferenceEngine::TBlob<uint8_t>::Ptr(weights);

    net_reader.SetWeights(weights_ptr);

    MKLDNNGraphTestClass graph;
    graph.CreateGraph(net_reader.getNetwork());

    InferenceEngine::TensorDesc desc(InferenceEngine::Precision::FP32, {1, 128, 1, 88}, InferenceEngine::NCHW);
    InferenceEngine::Blob::Ptr src1 = InferenceEngine::make_shared_blob<float>(desc);
    src1->allocate();
    fill_data((float *) src1->buffer(), src1->size());

    InferenceEngine::BlobMap srcs;
    srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("data", src1));

    InferenceEngine::OutputsDataMap out = net_reader.getNetwork().getOutputsInfo();

    InferenceEngine::BlobMap outputBlobs;
    std::pair<std::string, InferenceEngine::DataPtr> item = *out.begin();

    InferenceEngine::TBlob<float>::Ptr output;
    output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
    output->allocate();
    outputBlobs[item.first] = output;

    graph.Infer(srcs, outputBlobs);

    std::vector<float> refDst(output->size());
    auto *data = output->buffer().as<float *>();
    for (size_t i = 0; i < output->size(); i++) {
        refDst[i] = data[output->getTensorDesc().offset(i)];
    }

    // Compare for batch2
    net_reader.getNetwork().setBatchSize(2);
    graph.CreateGraph(net_reader.getNetwork());
    desc = InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32, {2, 128, 1, 88}, InferenceEngine::NCHW);

    InferenceEngine::Blob::Ptr src1Batch = InferenceEngine::make_shared_blob<float>(desc);
    src1Batch->allocate();
    data = src1Batch->buffer().as<float *>();
    auto *originData = src1->buffer().as<float *>();
    for(size_t b = 0; b < 2; b++) {
        for (size_t i = 0; i < src1->size(); i++) {
            data[src1Batch->getTensorDesc().offset(b*src1->size() + i)] = originData[src1->getTensorDesc().offset(i)];
        }
    }

    srcs.clear();
    srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("data", src1Batch));
    out = net_reader.getNetwork().getOutputsInfo();

    outputBlobs.clear();
    item = *out.begin();
    output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
    output->allocate();
    outputBlobs[item.first] = output;

    graph.Infer(srcs, outputBlobs);
    InferenceEngine::TBlob<float>::Ptr dstOut = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
    dstOut->allocate();
    data = dstOut->buffer().as<float *>();
    for(size_t b = 0; b < 2; b++) {
        for (size_t i = 0; i < refDst.size(); i++) {
            data[dstOut->getTensorDesc().offset(b*refDst.size() + i)] = refDst[i];
        }
    }

    compare(*output, *dstOut);
}

TEST_F(MKLDNNGraphStructureTests, TestFailedVNect0001) {
    std::string model = R"V0G0N(
<net name="net" version="2" batch="1">
    <layers>
        <layer name="data" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>23</dim>
					<dim>23</dim>
                </port>
            </output>
        </layer>
 		<layer id="207" name="res5c_branch1a" precision="FP32" type="Deconvolution">
			<data dilation-x="1" dilation-y="1" group="1" kernel-x="4" kernel-y="4" output="63" pad-x="1" pad-y="1" stride="1,1,2,2" stride-x="2" stride-y="2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>23</dim>
					<dim>23</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>63</dim>
					<dim>46</dim>
					<dim>46</dim>
				</port>
			</output>
			<blobs>
				<weights offset="0" size="1032192"/>
			</blobs>
		</layer>
		<layer id="347" name="res5c_branch1a_sqr" precision="FP32" type="Eltwise">
			<data operation="mul"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>63</dim>
					<dim>46</dim>
					<dim>46</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>63</dim>
					<dim>46</dim>
					<dim>46</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>63</dim>
					<dim>46</dim>
					<dim>46</dim>
				</port>
			</output>
		</layer>
		<layer id="236" name="split_res5c_branch1a" precision="FP32" type="Slice">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>63</dim>
					<dim>46</dim>
					<dim>46</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>21</dim>
					<dim>46</dim>
					<dim>46</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>21</dim>
					<dim>46</dim>
					<dim>46</dim>
				</port>
				<port id="3">
					<dim>1</dim>
					<dim>21</dim>
					<dim>46</dim>
					<dim>46</dim>
				</port>
			</output>
		</layer>
    </layers>
    <edges>
		<edge from-layer="0" from-port="0" to-layer="207" to-port="0"/>
		<edge from-layer="207" from-port="2" to-layer="347" to-port="0"/>
		<edge from-layer="207" from-port="2" to-layer="347" to-port="1"/>
		<edge from-layer="207" from-port="2" to-layer="236" to-port="0"/>
    </edges>
</net>)V0G0N";

    InferenceEngine::CNNNetReader net_reader;
    ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

    InferenceEngine::TBlob<uint8_t> *weights = new InferenceEngine::TBlob<uint8_t>({ InferenceEngine::Precision::FP32, { 1032192 }, InferenceEngine::C });
    weights->allocate();
    fill_data((float *)weights->buffer(), weights->size() / sizeof(float));

    InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr = InferenceEngine::TBlob<uint8_t>::Ptr(weights);

    net_reader.SetWeights(weights_ptr);

    MKLDNNGraphTestClass graph;
    ASSERT_NO_THROW(graph.CreateGraph(net_reader.getNetwork()));
}

TEST_F(MKLDNNGraphStructureTests, TestFailedVNect0002) {
    std::string model = R"V0G0N(
<net batch="1" name="vnect" version="2">
	<layers>
		<layer id="1" name="data" precision="FP32" type="Input">
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>46</dim>
					<dim>46</dim>
				</port>
			</output>
		</layer>
		<layer id="32" name="res5c_branch2c" precision="FP32" type="Convolution">
			<data dilation-x="1" dilation-y="1" group="1" kernel-x="1" kernel-y="1" output="84" pad-x="0" pad-y="0" stride="1,1,1,1" stride-x="1" stride-y="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>46</dim>
					<dim>46</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>84</dim>
					<dim>46</dim>
					<dim>46</dim>
				</port>
			</output>
			<blobs>
				<weights offset="0" size="43008"/>
			</blobs>
		</layer>
		<layer id="160" name="slice_heatmaps" precision="FP32" type="Slice">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>84</dim>
					<dim>46</dim>
					<dim>46</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>21</dim>
					<dim>46</dim>
					<dim>46</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>21</dim>
					<dim>46</dim>
					<dim>46</dim>
				</port>
				<port id="3">
					<dim>1</dim>
					<dim>21</dim>
					<dim>46</dim>
					<dim>46</dim>
				</port>
				<port id="4">
					<dim>1</dim>
					<dim>21</dim>
					<dim>46</dim>
					<dim>46</dim>
				</port>
			</output>
		</layer>
    </layers>
    <edges>
		<edge from-layer="1" from-port="1" to-layer="32" to-port="0"/>
		<edge from-layer="32" from-port="2" to-layer="160" to-port="0"/>
    </edges>
</net>
)V0G0N";

    InferenceEngine::CNNNetReader net_reader;
    ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

    InferenceEngine::TBlob<uint8_t> *weights = new InferenceEngine::TBlob<uint8_t>({ InferenceEngine::Precision::FP32, { 43008 }, InferenceEngine::C });
    weights->allocate();
    fill_data((float *)weights->buffer(), weights->size() / sizeof(float));

    InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr = InferenceEngine::TBlob<uint8_t>::Ptr(weights);

    net_reader.SetWeights(weights_ptr);

    MKLDNNGraphTestClass graph;
    graph.CreateGraph(net_reader.getNetwork());

    size_t outputs_num = 0;
    auto& nodes = graph.getNodes();
    for (auto &node : nodes) {
        if ( node->getType() == MKLDNNPlugin::Output &&
             (node->getName() == "out_slice_heatmaps.0" ||
              node->getName() == "out_slice_heatmaps.1" ||
              node->getName() == "out_slice_heatmaps.2" ||
              node->getName() == "out_slice_heatmaps.3" ) ) {
            outputs_num++;
        }
    }
    ASSERT_EQ(outputs_num, 4);
}


TEST_F(MKLDNNGraphStructureTests, TestFailedVNect0003) {
    std::string model = R"V0G0N(
<net name="net" version="2" batch="1">
    <layers>
        <layer name="data1" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>46</dim>
					<dim>46</dim>
                </port>
            </output>
        </layer>
        <layer name="data2" type="Input" precision="FP32" id="1">
            <output>
                <port id="0">
					<dim>1</dim>
					<dim>63</dim>
					<dim>46</dim>
					<dim>46</dim>
                </port>
            </output>
        </layer>
        <layer name="data3" type="Input" precision="FP32" id="2">
            <output>
                <port id="0">
					<dim>1</dim>
					<dim>21</dim>
					<dim>46</dim>
					<dim>46</dim>
                </port>
            </output>
        </layer>
		<layer id="86" name="res5c_branch2a_relu" precision="FP32" type="ReLU">
			<data engine="caffe.ReLUParameter.DEFAULT" negative_slope="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>46</dim>
					<dim>46</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>128</dim>
					<dim>46</dim>
					<dim>46</dim>
				</port>
			</output>
		</layer>
		<layer id="236" name="split_res5c_branch1a" precision="FP32" type="Slice">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>63</dim>
					<dim>46</dim>
					<dim>46</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>21</dim>
					<dim>46</dim>
					<dim>46</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>21</dim>
					<dim>46</dim>
					<dim>46</dim>
				</port>
				<port id="3">
					<dim>1</dim>
					<dim>21</dim>
					<dim>46</dim>
					<dim>46</dim>
				</port>
			</output>
		</layer>
		<layer id="67" name="res5c_bone_length" precision="FP32" type="Power">
			<data power="0.5" scale="1.0" shift="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>21</dim>
					<dim>46</dim>
					<dim>46</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>21</dim>
					<dim>46</dim>
					<dim>46</dim>
				</port>
			</output>
		</layer>
		<layer id="24" name="res5c_branch2a_feat" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>128</dim>
					<dim>46</dim>
					<dim>46</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>21</dim>
					<dim>46</dim>
					<dim>46</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>21</dim>
					<dim>46</dim>
					<dim>46</dim>
				</port>
				<port id="3">
					<dim>1</dim>
					<dim>21</dim>
					<dim>46</dim>
					<dim>46</dim>
				</port>
				<port id="4">
					<dim>1</dim>
					<dim>21</dim>
					<dim>46</dim>
					<dim>46</dim>
				</port>
			</input>
			<output>
				<port id="5">
					<dim>1</dim>
					<dim>212</dim>
					<dim>46</dim>
					<dim>46</dim>
				</port>
			</output>
		</layer>
    </layers>
    <edges>
		<edge from-layer="0" from-port="0" to-layer="86" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="236" to-port="0"/>
        <edge from-layer="2" from-port="0" to-layer="67" to-port="0"/>
		<edge from-layer="86" from-port="1" to-layer="24" to-port="0"/>
		<edge from-layer="236" from-port="1" to-layer="24" to-port="1"/>
		<edge from-layer="236" from-port="2" to-layer="24" to-port="2"/>
		<edge from-layer="236" from-port="3" to-layer="24" to-port="3"/>
		<edge from-layer="67" from-port="1" to-layer="24" to-port="4"/>
    </edges>
</net>)V0G0N";

    InferenceEngine::CNNNetReader net_reader;
    ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

    MKLDNNGraphTestClass graph;
    ASSERT_NO_THROW(graph.CreateGraph(net_reader.getNetwork()));
}

TEST_F(MKLDNNGraphStructureTests, TestConvolutionDWConvolutionSumFusing) {
    std::string model = R"V0G0N(
<net name="net" version="4" batch="1">
    <layers>
        <layer name="data0" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>32</dim>
                    <dim>300</dim>
                    <dim>600</dim>
                </port>
            </output>
        </layer>
        <layer name="data1" type="Input" precision="FP32" id="1">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>48</dim>
                    <dim>150</dim>
                    <dim>300</dim>
                </port>
            </output>
        </layer>
        <layer name="conv0" type="Convolution" precision="FP32" id="2">
			<data auto_pad="same_upper" dilations="1,1" group="1" kernel="1,1" output="48" pads_end="0, 0" pads_begin="150,300" strides="1,1"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>32</dim>
                    <dim>300</dim>
                    <dim>600</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>48</dim>
                    <dim>300</dim>
                    <dim>600</dim>
                </port>
            </output>
            <weights offset="0" size="6144"/>
            <biases offset="6144" size="192"/>
        </layer>
        <layer name="conv1" type="Convolution" precision="FP32" id="3">
			<data auto_pad="same_upper" dilations="1,1" group="48" kernel="3,3" output="48" pads_end="1,1" pads_begin="1,1" strides="2,2"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>48</dim>
                    <dim>300</dim>
                    <dim>600</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>48</dim>
                    <dim>150</dim>
                    <dim>300</dim>
                </port>
            </output>
            <weights offset="6336" size="1728"/>
            <biases offset="7872" size="192"/>
        </layer>
        <layer name="eltwise" type="Eltwise" precision="FP32" id="4">
            <data operation="sum"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>48</dim>
                    <dim>150</dim>
                    <dim>300</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                    <dim>48</dim>
                    <dim>150</dim>
                    <dim>300</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>48</dim>
                    <dim>150</dim>
                    <dim>300</dim>
                </port>
            </output>
        </layer>
        <layer name="relu" type="ReLU" precision="FP32" id="5">
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>48</dim>
                    <dim>150</dim>
                    <dim>300</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>48</dim>
                    <dim>150</dim>
                    <dim>300</dim>
                </port>
            </output>
        </layer>
        <layer name="power" type="Power" precision="FP32" id="6">
            <data power="1" scale="-1" shift="0"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>48</dim>
                    <dim>150</dim>
                    <dim>300</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>48</dim>
                    <dim>150</dim>
                    <dim>300</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="2" from-port="1" to-layer="3" to-port="0"/>
        <edge from-layer="3" from-port="1" to-layer="4" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="4" to-port="1"/>
        <edge from-layer="4" from-port="2" to-layer="5" to-port="0"/>
        <edge from-layer="5" from-port="1" to-layer="6" to-port="0"/>
    </edges>
</net>
)V0G0N";

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(model.data(), model.length());

    InferenceEngine::TBlob<uint8_t> *weights = new InferenceEngine::TBlob<uint8_t>({ InferenceEngine::Precision::U8, {8064}, InferenceEngine::C });
    weights->allocate();
    float * data = weights->buffer();
    memset((float *) weights->buffer(), 0, weights->size());

    InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr = InferenceEngine::TBlob<uint8_t>::Ptr(weights);

    net_reader.SetWeights(weights_ptr);

    MKLDNNGraphTestClass graph;
    graph.CreateGraph(net_reader.getNetwork());

    InferenceEngine::TensorDesc src0_desc(InferenceEngine::Precision::FP32, {1, 32, 300, 600}, InferenceEngine::NCHW);
    InferenceEngine::Blob::Ptr src0 = InferenceEngine::make_shared_blob<float>(src0_desc);
    src0->allocate();
    data = src0->buffer().as<float *>();
    for (size_t i = 0; i < src0->size(); i++) {
        data[i] = 0;
    }

    InferenceEngine::TensorDesc src1_desc(InferenceEngine::Precision::FP32, {1, 48, 150, 300}, InferenceEngine::NCHW);
    InferenceEngine::Blob::Ptr src1 = InferenceEngine::make_shared_blob<float>(src1_desc);
    src1->allocate();
    data = src1->buffer().as<float *>();
    for (size_t i = 0; i < src1->size(); i++) {
        data[i] = i % 10;
    }

    std::vector<float> refDst(src1->size());
    for (size_t i = 0; i < refDst.size(); i++) {
        refDst[i] = -1 * data[i];
    }

    InferenceEngine::BlobMap srcs;
    srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("data0", src0));
    srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("data1", src1));

    InferenceEngine::OutputsDataMap out = net_reader.getNetwork().getOutputsInfo();

    InferenceEngine::BlobMap outputBlobs;
    std::pair<std::string, InferenceEngine::DataPtr> item = *out.begin();

    InferenceEngine::TBlob<float>::Ptr output;
    output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
    output->allocate();
    outputBlobs[item.first] = output;

    graph.Infer(srcs, outputBlobs);

    InferenceEngine::TBlob<float>::Ptr dstOut = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc(), refDst.data());

    compare(*output, *dstOut);
}

TEST_F(MKLDNNGraphStructureTests, TestConstantLayerAsOutput) {
    std::string model = R"V0G0N(
<net batch="1" name="ResNet10_SSD" version="2">
	<layers>
		<layer id="0" name="data" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="Add_" precision="FP32" type="ScaleShift">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>3</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
			</output>
			<blobs>
				<weights offset="0" size="12"/>
				<biases offset="12" size="12"/>
			</blobs>
		</layer>
		<layer id="2" name="Convolution1" precision="FP32" type="Convolution">
			<data dilation-x="1" dilation-y="1" group="1" kernel-x="7" kernel-y="7" output="64" pad-x="3" pad-y="3" stride="1,1,2,2" stride-x="2" stride-y="2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>64</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
			</output>
			<blobs>
				<weights offset="24" size="37632"/>
				<biases offset="37656" size="256"/>
			</blobs>
		</layer>
		<layer id="3" name="x32_priorbox" precision="FP32" type="PriorBoxClustered">
			<data clip="0" flip="0" height="118.25800323486328,105.21199798583984,141.15499877929688,128.63600158691406,174.2689971923828,176.98300170898438" offset="0.5" step="32.0" variance="0.10000000149011612,0.10000000149011612,0.20000000298023224,0.20000000298023224" width="104.06500244140625,130.3560028076172,136.86500549316406,179.89199829101562,181.1739959716797,248.28199768066406"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>5</dim>
					<dim>5</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>3</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>2</dim>
					<dim>600</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
		<edge from-layer="1" from-port="3" to-layer="2" to-port="0"/>
		<edge from-layer="2" from-port="3" to-layer="3" to-port="0"/>
		<edge from-layer="1" from-port="3" to-layer="3" to-port="1"/>
	</edges>
</net>
)V0G0N";

    InferenceEngine::CNNNetReader net_reader;
    ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

    InferenceEngine::TBlob<uint8_t> *weights = new InferenceEngine::TBlob<uint8_t>({ InferenceEngine::Precision::U8, {37912}, InferenceEngine::C });
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));

    InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr = InferenceEngine::TBlob<uint8_t>::Ptr(weights);

    net_reader.SetWeights(weights_ptr);

    MKLDNNGraphTestClass graph;
    graph.CreateGraph(net_reader.getNetwork());

    InferenceEngine::TensorDesc desc(InferenceEngine::Precision::FP32, {1, 3, 10, 10}, InferenceEngine::NCHW);
    InferenceEngine::Blob::Ptr src = InferenceEngine::make_shared_blob<float>(desc);
    src->allocate();
    auto *data = src->buffer().as<float *>();
    size_t sizeB1 = src->size() / 2;
    fill_data(data, sizeB1);
    for (size_t i = 0; i < sizeB1; i++) {
        data[sizeB1 + i] = data[i];
    }

    InferenceEngine::BlobMap srcs;
    srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("data", src));

    InferenceEngine::OutputsDataMap out = net_reader.getNetwork().getOutputsInfo();

    InferenceEngine::BlobMap outputBlobs;
    std::pair<std::string, InferenceEngine::DataPtr> item = *out.begin();

    InferenceEngine::TBlob<float>::Ptr output;
    output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
    output->allocate();
    outputBlobs[item.first] = output;

    graph.Infer(srcs, outputBlobs);

    std::vector<float> refDst = {-3.603f,-4.313f,6.803f,7.513f,-4.918f,-3.661f,8.118f,6.861f,-5.243f,-5.458f,8.443f,8.658f,-7.395f,-4.832f,10.595f,8.032f,
                                 -7.459f,-7.113f,10.659f,10.313f,-10.814f,-7.249f,14.014f,10.449f,-0.403f,-4.313f,10.003f,7.513f,-1.718f,-3.661f,11.318f,6.861f,
                                 -2.043f,-5.458f,11.643f,8.658f,-4.195f,-4.832f,13.795f,8.032f,-4.259f,-7.113f,13.859f,10.313f,-7.614f,-7.249f,17.214f,10.449f,
                                 2.797f,-4.313f,13.203f,7.513f,1.482f,-3.661f,14.518f,6.861f,1.157f,-5.458f,14.843f,8.658f,-0.995f,-4.832f,16.995f,8.032f,
                                 -1.059f,-7.113f,17.059f,10.313f,-4.414f,-7.249f,20.414f,10.449f,5.997f,-4.313f,16.403f,7.513f,4.682f,-3.661f,17.718f,6.861f,
                                 4.357f,-5.458f,18.043f,8.658f,2.205f,-4.832f,20.195f,8.032f,2.141f,-7.113f,20.259f,10.313f,-1.214f,-7.249f,23.614f,10.449f,
                                 9.197f,-4.313f,19.603f,7.513f,7.882f,-3.661f,20.918f,6.861f,7.557f,-5.458f,21.243f,8.658f,5.405f,-4.832f,23.395f,8.032f,5.341f,
                                 -7.113f,23.459f,10.313f,1.986f,-7.249f,26.814f,10.449f,-3.603f,-1.113f,6.803f,10.713f,-4.918f,-0.461f,8.118f,10.061f,-5.243f,-2.258f,
                                 8.443f,11.858f,-7.395f,-1.632f,10.595f,11.232f,-7.459f,-3.913f,10.659f,13.513f,-10.814f,-4.049f,14.014f,13.649f,-0.403f,-1.113f,
                                 10.003f,10.713f,-1.718f,-0.461f,11.318f,10.061f,-2.043f,-2.258f,11.643f,11.858f,-4.195f,-1.632f,13.795f,11.232f,-4.259f,-3.913f,
                                 13.859f,13.513f,-7.614f,-4.049f,17.214f,13.649f,2.797f,-1.113f,13.203f,10.713f,1.482f,-0.461f,14.518f,10.061f,1.157f,-2.258f,14.843f,
                                 11.858f,-0.995f,-1.632f,16.995f,11.232f,-1.059f,-3.913f,17.059f,13.513f,-4.414f,-4.049f,20.414f,13.649f,5.997f,-1.113f,16.403f,10.713f,
                                 4.682f,-0.461f,17.718f,10.061f,4.357f,-2.258f,18.043f,11.858f,2.205f,-1.632f,20.195f,11.232f,2.141f,-3.913f,20.259f,13.513f,-1.214f,
                                 -4.049f,23.614f,13.649f,9.197f,-1.113f,19.603f,10.713f,7.882f,-0.461f,20.918f,10.061f,7.557f,-2.258f,21.243f,11.858f,5.405f,-1.632f,
                                 23.395f,11.232f,5.341f,-3.913f,23.459f,13.513f,1.986f,-4.049f,26.814f,13.649f,-3.603f,2.087f,6.803f,13.913f,-4.918f,2.739f,8.118f,
                                 13.261f,-5.243f,0.942f,8.443f,15.058f,-7.395f,1.568f,10.595f,14.432f,-7.459f,-0.713f,10.659f,16.713f,-10.814f,-0.849f,14.014f,16.849f,
                                 -0.403f,2.087f,10.003f,13.913f,-1.718f,2.739f,11.318f,13.261f,-2.043f,0.942f,11.643f,15.058f,-4.195f,1.568f,13.795f,14.432f,-4.259f,
                                 -0.713f,13.859f,16.713f,-7.614f,-0.849f,17.214f,16.849f,2.797f,2.087f,13.203f,13.913f,1.482f,2.739f,14.518f,13.261f,1.157f,0.942f,14.843f,
                                 15.058f,-0.995f,1.568f,16.995f,14.432f,-1.059f,-0.713f,17.059f,16.713f,-4.414f,-0.849f,20.414f,16.849f,5.997f,2.087f,16.403f,13.913f,
                                 4.682f,2.739f,17.718f,13.261f,4.357f,0.942f,18.043f,15.058f,2.205f,1.568f,20.195f,14.432f,2.141f,-0.713f,20.259f,16.713f,-1.214f,-0.849f,
                                 23.614f,16.849f,9.197f,2.087f,19.603f,13.913f,7.882f,2.739f,20.918f,13.261f,7.557f,0.942f,21.243f,15.058f,5.405f,1.568f,23.395f,14.432f,
                                 5.341f,-0.713f,23.459f,16.713f,1.986f,-0.849f,26.814f,16.849f,-3.603f,5.287f,6.803f,17.113f,-4.918f,5.939f,8.118f,16.461f,-5.243f,4.142f,
                                 8.443f,18.258f,-7.395f,4.768f,10.595f,17.632f,-7.459f,2.487f,10.659f,19.913f,-10.814f,2.351f,14.014f,20.049f,-0.403f,5.287f,10.003f,
                                 17.113f,-1.718f,5.939f,11.318f,16.461f,-2.043f,4.142f,11.643f,18.258f,-4.195f,4.768f,13.795f,17.632f,-4.259f,2.487f,13.859f,19.913f,
                                 -7.614f,2.351f,17.214f,20.049f,2.797f,5.287f,13.203f,17.113f,1.482f,5.939f,14.518f,16.461f,1.157f,4.142f,14.843f,18.258f,-0.995f,4.768f,
                                 16.995f,17.632f,-1.059f,2.487f,17.059f,19.913f,-4.414f,2.351f,20.414f,20.049f,5.997f,5.287f,16.403f,17.113f,4.682f,5.939f,17.718f,16.461f,
                                 4.357f,4.142f,18.043f,18.258f,2.205f,4.768f,20.195f,17.632f,2.141f,2.487f,20.259f,19.913f,-1.214f,2.351f,23.614f,20.049f,9.197f,5.287f,
                                 19.603f,17.113f,7.882f,5.939f,20.918f,16.461f,7.557f,4.142f,21.243f,18.258f,5.405f,4.768f,23.395f,17.632f,5.341f,2.487f,23.459f,19.913f,
                                 1.986f,2.351f,26.814f,20.049f,-3.603f,8.487f,6.803f,20.313f,-4.918f,9.139f,8.118f,19.661f,-5.243f,7.342f,8.443f,21.458f,-7.395f,7.968f,
                                 10.595f,20.832f,-7.459f,5.687f,10.659f,23.113f,-10.814f,5.551f,14.014f,23.249f,-0.403f,8.487f,10.003f,20.313f,-1.718f,9.139f,11.318f,
                                 19.661f,-2.043f,7.342f,11.643f,21.458f,-4.195f,7.968f,13.795f,20.832f,-4.259f,5.687f,13.859f,23.113f,-7.614f,5.551f,17.214f,23.249f,2.797f,
                                 8.487f,13.203f,20.313f,1.482f,9.139f,14.518f,19.661f,1.157f,7.342f,14.843f,21.458f,-0.995f,7.968f,16.995f,20.832f,-1.059f,5.687f,17.059f,
                                 23.113f,-4.414f,5.551f,20.414f,23.249f,5.997f,8.487f,16.403f,20.313f,4.682f,9.139f,17.718f,19.661f,4.357f,7.342f,18.043f,21.458f,2.205f,
                                 7.968f,20.195f,20.832f,2.141f,5.687f,20.259f,23.113f,-1.214f,5.551f,23.614f,23.249f,9.197f,8.487f,19.603f,20.313f,7.882f,9.139f,20.918f,
                                 19.661f,7.557f,7.342f,21.243f,21.458f,5.405f,7.968f,23.395f,20.832f,5.341f,5.687f,23.459f,23.113f,1.986f,5.551f,26.814f,23.249f,0.100f,
                                 0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,
                                 0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,
                                 0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,
                                 0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,
                                 0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,
                                 0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,
                                 0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,
                                 0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,
                                 0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,
                                 0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,
                                 0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,
                                 0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,
                                 0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,
                                 0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,
                                 0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,
                                 0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,
                                 0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,
                                 0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,
                                 0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,
                                 0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,
                                 0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,
                                 0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,
                                 0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,
                                 0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,
                                 0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,
                                 0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,
                                 0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,
                                 0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,
                                 0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,
                                 0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f,0.100f,0.100f,0.200f,0.200f};
    InferenceEngine::TBlob<float>::Ptr dstOut = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc(), refDst.data());

    compare(*output, *dstOut);
}

TEST_F(MKLDNNGraphStructureTests, TestGemmConvolutionWithConcat) {
    std::string model = R"V0G0N(
<net batch="1" name="squeezenet1.1" version="3">
	<layers>
		<layer id="0" name="data" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>227</dim>
					<dim>227</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="conv1" precision="FP32" type="Convolution">
			<data dilations="1,1" group="1" kernel="3,3" output="64" pads_begin="0,0" pads_end="0,0" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>227</dim>
					<dim>227</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>64</dim>
					<dim>113</dim>
					<dim>113</dim>
				</port>
			</output>
			<blobs>
				<weights offset="0" size="6912"/>
				<biases offset="6912" size="256"/>
			</blobs>
		</layer>
		<layer id="2" name="relu_conv1" precision="FP32" type="ReLU">
			<data negative_slope="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>113</dim>
					<dim>113</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>113</dim>
					<dim>113</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="pool1" precision="FP32" type="Pooling">
			<data exclude-pad="false" kernel="3,3" pads_begin="0,0" pads_end="0,0" pool-method="max" rounding_type="ceil" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>113</dim>
					<dim>113</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>56</dim>
					<dim>56</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="fire2/squeeze1x1" precision="FP32" type="Convolution">
			<data dilation="1,1" group="1" kernel="1,1" output="16" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>56</dim>
					<dim>56</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>16</dim>
					<dim>56</dim>
					<dim>56</dim>
				</port>
			</output>
			<blobs>
				<weights offset="7168" size="4096"/>
				<biases offset="11264" size="64"/>
			</blobs>
		</layer>
		<layer id="5" name="fire2/relu_squeeze1x1" precision="FP32" type="ReLU">
			<data negative_slope="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>56</dim>
					<dim>56</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>16</dim>
					<dim>56</dim>
					<dim>56</dim>
				</port>
			</output>
		</layer>
		<layer id="6" name="fire2/expand1x1" precision="FP32" type="Convolution">
			<data dilation="1,1" group="1" kernel="1,1" output="64" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>56</dim>
					<dim>56</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>64</dim>
					<dim>56</dim>
					<dim>56</dim>
				</port>
			</output>
			<blobs>
				<weights offset="11328" size="4096"/>
				<biases offset="15424" size="256"/>
			</blobs>
		</layer>
		<layer id="7" name="fire2/relu_expand1x1" precision="FP32" type="ReLU">
			<data negative_slope="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>56</dim>
					<dim>56</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>56</dim>
					<dim>56</dim>
				</port>
			</output>
		</layer>
		<layer id="8" name="fire2/expand3x3" precision="FP32" type="Convolution">
			<data dilation="1,1" group="1" kernel="3,3" output="64" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>56</dim>
					<dim>56</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>64</dim>
					<dim>56</dim>
					<dim>56</dim>
				</port>
			</output>
			<blobs>
				<weights offset="15680" size="36864"/>
				<biases offset="52544" size="256"/>
			</blobs>
		</layer>
		<layer id="9" name="fire2/relu_expand3x3" precision="FP32" type="ReLU">
			<data negative_slope="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>56</dim>
					<dim>56</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>56</dim>
					<dim>56</dim>
				</port>
			</output>
		</layer>
		<layer id="10" name="fire2/concat" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>56</dim>
					<dim>56</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>56</dim>
					<dim>56</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>128</dim>
					<dim>56</dim>
					<dim>56</dim>
				</port>
			</output>
		</layer>
    </layers>
    <edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
		<edge from-layer="1" from-port="3" to-layer="2" to-port="0"/>
		<edge from-layer="2" from-port="1" to-layer="3" to-port="0"/>
		<edge from-layer="3" from-port="1" to-layer="4" to-port="0"/>
		<edge from-layer="4" from-port="3" to-layer="5" to-port="0"/>
		<edge from-layer="5" from-port="1" to-layer="6" to-port="0"/>
		<edge from-layer="6" from-port="3" to-layer="7" to-port="0"/>
		<edge from-layer="5" from-port="1" to-layer="8" to-port="0"/>
		<edge from-layer="8" from-port="3" to-layer="9" to-port="0"/>
		<edge from-layer="7" from-port="1" to-layer="10" to-port="0"/>
		<edge from-layer="9" from-port="1" to-layer="10" to-port="1"/>
    </edges>
</net>
)V0G0N";

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(model.data(), model.length());

    InferenceEngine::TBlob<uint8_t> *weights = new InferenceEngine::TBlob<uint8_t>({ InferenceEngine::Precision::U8, {52800}, InferenceEngine::C });
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));
    InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr = InferenceEngine::TBlob<uint8_t>::Ptr(weights);
    net_reader.SetWeights(weights_ptr);

    auto graphInfer = [](InferenceEngine::CNNNetwork network, InferenceEngine::BlobMap& inBlobs,
            InferenceEngine::BlobMap& outBlobs, std::string primitivesPriority) {
        for (auto it = network.begin(); !primitivesPriority.empty() && it !=network.end(); it++) {
            (*it)->params["PrimitivesPriority"] = primitivesPriority;
        }

        MKLDNNGraphTestClass graph;
        graph.CreateGraph(network);
        graph.Infer(inBlobs, outBlobs);

#if 0
        std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> perfMap;
        graph.GetPerfData(perfMap);

        long long totalTime = 0;
        // Print performance counts

        std::cout << std::endl << "performance counts:" << std::endl << std::endl;
        for (const auto & it : perfMap) {
            std::string toPrint(it.first);
            const int maxLayerName = 30;

            if (it.first.length() >= maxLayerName) {
                toPrint  = it.first.substr(0, maxLayerName - 4);
                toPrint += "...";
            }


            std::cout << std::setw(maxLayerName) << std::left << toPrint;
            switch (it.second.status) {
                case InferenceEngine::InferenceEngineProfileInfo::EXECUTED:
                    std::cout << std::setw(15) << std::left << "EXECUTED";
                    break;
                case InferenceEngine::InferenceEngineProfileInfo::NOT_RUN:
                    std::cout << std::setw(15) << std::left << "NOT_RUN";
                    break;
                case InferenceEngine::InferenceEngineProfileInfo::OPTIMIZED_OUT:
                    std::cout << std::setw(15) << std::left << "OPTIMIZED_OUT";
                    break;
            }
            std::cout << std::setw(30) << std::left << "layerType: " + std::string(it.second.layer_type) + " ";
            std::cout << std::setw(20) << std::left << "realTime: " + std::to_string(it.second.realTime_uSec);
            std::cout << std::setw(20) << std::left << " cpu: "  + std::to_string(it.second.cpu_uSec);
            std::cout << " execType: " << it.second.exec_type << std::endl;
            if (it.second.realTime_uSec > 0) {
                totalTime += it.second.realTime_uSec;
            }
        }
        std::cout << std::setw(20) << std::left << "Total time: " + std::to_string(totalTime) << " microseconds" << std::endl;
#endif
    };

    InferenceEngine::InputsDataMap inputsMap = net_reader.getNetwork().getInputsInfo();
    InferenceEngine::BlobMap inputBlobs;

    for (const auto& input : inputsMap) {
        InferenceEngine::Blob::Ptr src = InferenceEngine::make_shared_blob<float>(input.second->getTensorDesc());
        src->allocate();
        fill_data((float *) src->buffer(), src->size());
        inputBlobs[input.first] = src;
    }

    InferenceEngine::OutputsDataMap outsMap = net_reader.getNetwork().getOutputsInfo();
    InferenceEngine::BlobMap outputBlobs1;
    InferenceEngine::BlobMap outputBlobs2;
    for (const auto& output : outsMap) {
        InferenceEngine::TBlob<float>::Ptr dst1, dst2;
        dst1 = InferenceEngine::make_shared_blob<float>(output.second->getTensorDesc());
        dst1->allocate();
        outputBlobs1[output.first] = dst1;
        dst2 = InferenceEngine::make_shared_blob<float>(output.second->getTensorDesc());
        dst2->allocate();
        outputBlobs2[output.first] = dst2;
    }

    graphInfer(net_reader.getNetwork(), inputBlobs, outputBlobs1, "");
    graphInfer(net_reader.getNetwork(), inputBlobs, outputBlobs2, "cpu:gemm_blas");
    compare(*outputBlobs1.begin()->second, *outputBlobs2.begin()->second);

    graphInfer(net_reader.getNetwork(), inputBlobs, outputBlobs2, "cpu:gemm_avx512");
    compare(*outputBlobs1.begin()->second, *outputBlobs2.begin()->second);

    graphInfer(net_reader.getNetwork(), inputBlobs, outputBlobs2, "cpu:gemm_avx2");
    compare(*outputBlobs1.begin()->second, *outputBlobs2.begin()->second);

    graphInfer(net_reader.getNetwork(), inputBlobs, outputBlobs2, "cpu:gemm_sse42");
    compare(*outputBlobs1.begin()->second, *outputBlobs2.begin()->second);

    graphInfer(net_reader.getNetwork(), inputBlobs, outputBlobs2, "cpu:gemm_any");
    compare(*outputBlobs1.begin()->second, *outputBlobs2.begin()->second);
}


TEST_F(MKLDNNGraphStructureTests, TestRefPoolingWithConcat) {
    std::string model = R"V0G0N(
<net batch="1" name="squeezenet1.1" version="3">
	<layers>
		<layer id="0" name="data" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>227</dim>
					<dim>227</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="conv1" precision="FP32" type="Convolution">
			<data dilation="1,1" group="1" kernel="3,3" output="64" pads_begin="0,0" pads_end="0,0" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>227</dim>
					<dim>227</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>64</dim>
					<dim>113</dim>
					<dim>113</dim>
				</port>
			</output>
			<blobs>
				<weights offset="0" size="6912"/>
				<biases offset="6912" size="256"/>
			</blobs>
		</layer>
		<layer id="2" name="relu_conv1" precision="FP32" type="ReLU">
			<data negative_slope="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>113</dim>
					<dim>113</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>113</dim>
					<dim>113</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="pool1" precision="FP32" type="Pooling">
			<data exclude-pad="false" kernel="3,3" pads_begin="0,0" pads_end="0,0" pool-method="max" rounding_type="ceil" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>113</dim>
					<dim>113</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>56</dim>
					<dim>56</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="fire2/squeeze1x1" precision="FP32" type="Convolution">
			<data dilation="1,1" group="1" kernel="1,1" output="16" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>56</dim>
					<dim>56</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>16</dim>
					<dim>56</dim>
					<dim>56</dim>
				</port>
			</output>
			<blobs>
				<weights offset="7168" size="4096"/>
				<biases offset="11264" size="64"/>
			</blobs>
		</layer>
		<layer id="5" name="fire2/relu_squeeze1x1" precision="FP32" type="ReLU">
			<data negative_slope="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>56</dim>
					<dim>56</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>16</dim>
					<dim>56</dim>
					<dim>56</dim>
				</port>
			</output>
		</layer>
		<layer id="6" name="fire2/expand1x1" precision="FP32" type="Convolution">
			<data dilation="1,1" group="1" kernel="1,1" output="64" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>56</dim>
					<dim>56</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>64</dim>
					<dim>56</dim>
					<dim>56</dim>
				</port>
			</output>
			<blobs>
				<weights offset="11328" size="4096"/>
				<biases offset="15424" size="256"/>
			</blobs>
		</layer>
		<layer id="7" name="fire2/relu_expand1x1" precision="FP32" type="ReLU">
			<data negative_slope="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>56</dim>
					<dim>56</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>64</dim>
					<dim>56</dim>
					<dim>56</dim>
				</port>
			</output>
		</layer>
		<layer id="8" name="fire2/expand3x3" precision="FP32" type="Pooling">
			<data exclude-pad="false" kernel="3,3" pads_begin="1,1" pool-method="avg" rounding_type="ceil" stride="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>16</dim>
					<dim>56</dim>
					<dim>56</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>16</dim>
					<dim>56</dim>
					<dim>56</dim>
				</port>
			</output>
			<blobs>
				<weights offset="15680" size="36864"/>
				<biases offset="52544" size="256"/>
			</blobs>
		</layer>
		<layer id="10" name="fire2/concat" precision="FP32" type="Concat">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>56</dim>
					<dim>56</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>16</dim>
					<dim>56</dim>
					<dim>56</dim>
				</port>
			</input>
			<output>
				<port id="2">
					<dim>1</dim>
					<dim>80</dim>
					<dim>56</dim>
					<dim>56</dim>
				</port>
			</output>
		</layer>
    </layers>
    <edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
		<edge from-layer="1" from-port="3" to-layer="2" to-port="0"/>
		<edge from-layer="2" from-port="1" to-layer="3" to-port="0"/>
		<edge from-layer="3" from-port="1" to-layer="4" to-port="0"/>
		<edge from-layer="4" from-port="3" to-layer="5" to-port="0"/>
		<edge from-layer="5" from-port="1" to-layer="6" to-port="0"/>
		<edge from-layer="6" from-port="3" to-layer="7" to-port="0"/>
		<edge from-layer="5" from-port="1" to-layer="8" to-port="0"/>
		<edge from-layer="7" from-port="1" to-layer="10" to-port="0"/>
		<edge from-layer="8" from-port="3" to-layer="10" to-port="1"/>
    </edges>
</net>
)V0G0N";

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(model.data(), model.length());

    InferenceEngine::TBlob<uint8_t> *weights = new InferenceEngine::TBlob<uint8_t>({ InferenceEngine::Precision::U8, {52800}, InferenceEngine::C });
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));
    InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr = InferenceEngine::TBlob<uint8_t>::Ptr(weights);
    net_reader.SetWeights(weights_ptr);

    auto graphInfer = [](InferenceEngine::CNNNetwork network, InferenceEngine::BlobMap& inBlobs,
                         InferenceEngine::BlobMap& outBlobs, std::string primitivesPriority) {
        for (auto it = network.begin(); !primitivesPriority.empty() && it !=network.end(); it++) {
            (*it)->params["PrimitivesPriority"] = primitivesPriority;
        }

        MKLDNNGraphTestClass graph;
        graph.CreateGraph(network);
        graph.Infer(inBlobs, outBlobs);

#if 1
        std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> perfMap;
        graph.GetPerfData(perfMap);

        long long totalTime = 0;
        // Print performance counts

        std::cout << std::endl << "performance counts:" << std::endl << std::endl;
        for (const auto & it : perfMap) {
            std::string toPrint(it.first);
            const int maxLayerName = 30;

            if (it.first.length() >= maxLayerName) {
                toPrint  = it.first.substr(0, maxLayerName - 4);
                toPrint += "...";
            }


            std::cout << std::setw(maxLayerName) << std::left << toPrint;
            switch (it.second.status) {
                case InferenceEngine::InferenceEngineProfileInfo::EXECUTED:
                    std::cout << std::setw(15) << std::left << "EXECUTED";
                    break;
                case InferenceEngine::InferenceEngineProfileInfo::NOT_RUN:
                    std::cout << std::setw(15) << std::left << "NOT_RUN";
                    break;
                case InferenceEngine::InferenceEngineProfileInfo::OPTIMIZED_OUT:
                    std::cout << std::setw(15) << std::left << "OPTIMIZED_OUT";
                    break;
            }
            std::cout << std::setw(30) << std::left << "layerType: " + std::string(it.second.layer_type) + " ";
            std::cout << std::setw(20) << std::left << "realTime: " + std::to_string(it.second.realTime_uSec);
            std::cout << std::setw(20) << std::left << " cpu: "  + std::to_string(it.second.cpu_uSec);
            std::cout << " execType: " << it.second.exec_type << std::endl;
            if (it.second.realTime_uSec > 0) {
                totalTime += it.second.realTime_uSec;
            }
        }
        std::cout << std::setw(20) << std::left << "Total time: " + std::to_string(totalTime) << " microseconds" << std::endl;
#endif
    };

    InferenceEngine::InputsDataMap inputsMap = net_reader.getNetwork().getInputsInfo();
    InferenceEngine::BlobMap inputBlobs;

    for (const auto& input : inputsMap) {
        InferenceEngine::Blob::Ptr src = InferenceEngine::make_shared_blob<float>(input.second->getTensorDesc());
        src->allocate();
        fill_data((float *) src->buffer(), src->size());
        inputBlobs[input.first] = src;
    }

    InferenceEngine::OutputsDataMap outsMap = net_reader.getNetwork().getOutputsInfo();
    InferenceEngine::BlobMap outputBlobs1;
    InferenceEngine::BlobMap outputBlobs2;
    for (const auto& output : outsMap) {
        InferenceEngine::TBlob<float>::Ptr dst1, dst2;
        dst1 = InferenceEngine::make_shared_blob<float>(output.second->getTensorDesc());
        dst1->allocate();
        outputBlobs1[output.first] = dst1;
        dst2 = InferenceEngine::make_shared_blob<float>(output.second->getTensorDesc());
        dst2->allocate();
        outputBlobs2[output.first] = dst2;
    }

    graphInfer(net_reader.getNetwork(), inputBlobs, outputBlobs1, "");
    graphInfer(net_reader.getNetwork(), inputBlobs, outputBlobs2, "cpu:gemm_blas,cpu:ref_any");
    compare(*outputBlobs1.begin()->second, *outputBlobs2.begin()->second);

    graphInfer(net_reader.getNetwork(), inputBlobs, outputBlobs2, "cpu:ref_any");
    compare(*outputBlobs1.begin()->second, *outputBlobs2.begin()->second);
}

TEST_F(MKLDNNGraphStructureTests, TestConvolutionWith2DepthwiseOpFusing) {
    std::string model = R"V0G0N(
<net name="net" version="2" batch="1">
    <layers>
        <layer name="data" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>32</dim>
                    <dim>300</dim>
                    <dim>600</dim>
                </port>
            </output>
        </layer>
        <layer name="conv" type="Convolution" precision="FP32" id="1">
            <convolution_data stride-x="1" stride-y="1" pad-x="0" pad-y="0" kernel-x="1" kernel-y="1" output="48" group="1"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>32</dim>
                    <dim>300</dim>
                    <dim>600</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>48</dim>
                    <dim>300</dim>
                    <dim>600</dim>
                </port>
            </output>
            <weights offset="0" size="6144"/>
            <biases offset="6144" size="192"/>
        </layer>
        <layer name="depthwise0" type="PReLU" precision="FP32" id="2">
            <data channel_shared="1"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>48</dim>
                    <dim>300</dim>
                    <dim>600</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>48</dim>
                    <dim>300</dim>
                    <dim>600</dim>
                </port>
            </output>
            <weights offset="6336" size="4"/>
        </layer>
        <layer name="depthwise1" type="ScaleShift" precision="FP32" id="3">
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>48</dim>
                    <dim>300</dim>
                    <dim>600</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>48</dim>
                    <dim>300</dim>
                    <dim>600</dim>
                </port>
            </output>
            <weights offset="6340" size="192"/>
            <biases offset="6532" size="192"/>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>
        <edge from-layer="2" from-port="1" to-layer="3" to-port="0"/>
    </edges>
</net>
)V0G0N";

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(model.data(), model.length());

    InferenceEngine::TBlob<uint8_t> *weights = new InferenceEngine::TBlob<uint8_t>({ InferenceEngine::Precision::U8, {6724}, InferenceEngine::C });
    weights->allocate();
    float* wdata = weights->buffer();

    for (int i = 0; i < weights->size() / sizeof(float); i++)
        wdata[i] = 1;
    wdata[1584] = 2; // 2 for prelu weights

    InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr = InferenceEngine::TBlob<uint8_t>::Ptr(weights);

    net_reader.SetWeights(weights_ptr);

    MKLDNNGraphTestClass graph;
    graph.CreateGraph(net_reader.getNetwork());

    const auto& nodes = graph.getNodes();
    ASSERT_EQ(nodes.size(), 5);
    ASSERT_EQ(nodes[0].get()->getType(), MKLDNNPlugin::Type::Input);
    ASSERT_EQ(nodes[1].get()->getType(), MKLDNNPlugin::Type::Reorder);
    ASSERT_EQ(nodes[2].get()->getType(), MKLDNNPlugin::Type::Convolution);
    ASSERT_TRUE(nodes[2].get()->isFusedWith(MKLDNNPlugin::Type::Depthwise));
    ASSERT_EQ(nodes[3].get()->getType(), MKLDNNPlugin::Type::Reorder);
    ASSERT_EQ(nodes[4].get()->getType(), MKLDNNPlugin::Type::Output);

    InferenceEngine::TensorDesc src_desc(InferenceEngine::Precision::FP32, {1, 32, 300, 600}, InferenceEngine::NCHW);
    InferenceEngine::Blob::Ptr src = InferenceEngine::make_shared_blob<float>(src_desc);
    src->allocate();
    float* sdata = src->buffer().as<float *>();
    for (size_t i = 0; i < src->size(); i++) {
        sdata[i] = -1;
    }

    std::vector<float> refDst(1 * 48 * 300 * 600);
    for (size_t i = 0; i < refDst.size(); i++) {
        refDst[i] = -61; // (-32 + 1) * 2 * 1 + 1
    }

    InferenceEngine::BlobMap srcs;
    srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("data", src));

    InferenceEngine::OutputsDataMap out = net_reader.getNetwork().getOutputsInfo();

    InferenceEngine::BlobMap outputBlobs;
    std::pair<std::string, InferenceEngine::DataPtr> item = *out.begin();

    InferenceEngine::TBlob<float>::Ptr output;
    output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
    output->allocate();
    outputBlobs[item.first] = output;

    graph.Infer(srcs, outputBlobs);

    InferenceEngine::TBlob<float>::Ptr dstOut = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc(), refDst.data());

    compare(*output, *dstOut);
}

TEST_F(MKLDNNGraphStructureTests, TestConvolutionWith2EltwiseOpFusing) {
    std::string model = R"V0G0N(
<net name="net" version="2" batch="1">
    <layers>
        <layer name="data" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>300</dim>
                    <dim>600</dim>
                </port>
            </output>
        </layer>
        <layer name="conv" type="Convolution" precision="FP32" id="1">
            <convolution_data stride-x="1" stride-y="1" pad-x="0" pad-y="0" kernel-x="1" kernel-y="1" output="48" group="1"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>300</dim>
                    <dim>600</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>48</dim>
                    <dim>300</dim>
                    <dim>600</dim>
                </port>
            </output>
            <weights offset="0" size="192"/>
            <biases offset="192" size="192"/>
        </layer>
        <layer name="eltwise0" type="Logistic" precision="FP32" id="2">
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>48</dim>
                    <dim>300</dim>
                    <dim>600</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>48</dim>
                    <dim>300</dim>
                    <dim>600</dim>
                </port>
            </output>
        </layer>
        <layer name="eltwise1" type="Clamp" precision="FP32" id="3">
            <data max="1" min="0.3"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>48</dim>
                    <dim>300</dim>
                    <dim>600</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>48</dim>
                    <dim>300</dim>
                    <dim>600</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>
        <edge from-layer="2" from-port="1" to-layer="3" to-port="0"/>
    </edges>
</net>
)V0G0N";

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(model.data(), model.length());

    InferenceEngine::TBlob<uint8_t> *weights = new InferenceEngine::TBlob<uint8_t>({ InferenceEngine::Precision::U8, {384}, InferenceEngine::C });
    weights->allocate();
    float* wdata = weights->buffer();

    for (int i = 0; i < weights->size() / sizeof(float); i++)
        wdata[i] = 1;

    InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr = InferenceEngine::TBlob<uint8_t>::Ptr(weights);

    net_reader.SetWeights(weights_ptr);

    MKLDNNGraphTestClass graph;
    graph.CreateGraph(net_reader.getNetwork());

    const auto& nodes = graph.getNodes();
    ASSERT_EQ(nodes.size(), 4);
    ASSERT_EQ(nodes[0].get()->getType(), MKLDNNPlugin::Type::Input);
    ASSERT_EQ(nodes[1].get()->getType(), MKLDNNPlugin::Type::Convolution);
    ASSERT_TRUE(nodes[1].get()->isFusedWith(MKLDNNPlugin::Type::Activation));
    ASSERT_EQ(nodes[2].get()->getType(), MKLDNNPlugin::Type::Reorder);
    ASSERT_EQ(nodes[3].get()->getType(), MKLDNNPlugin::Type::Output);

    InferenceEngine::TensorDesc src_desc(InferenceEngine::Precision::FP32, {1, 1, 300, 600}, InferenceEngine::NCHW);
    InferenceEngine::Blob::Ptr src = InferenceEngine::make_shared_blob<float>(src_desc);
    src->allocate();
    float* sdata = src->buffer().as<float *>();
    for (size_t i = 0; i < src->size(); i++) {
        sdata[i] = i % 2 == 0 ? 2 : -2;
    }

    std::vector<float> refDst(1 * 48 * 300 * 600);
    for (size_t i = 0; i < refDst.size(); i++) {
        refDst[i] = i % 2 == 0 ? 0.952574127f : 0.3f;
    }

    InferenceEngine::BlobMap srcs;
    srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("data", src));

    InferenceEngine::OutputsDataMap out = net_reader.getNetwork().getOutputsInfo();

    InferenceEngine::BlobMap outputBlobs;
    std::pair<std::string, InferenceEngine::DataPtr> item = *out.begin();

    InferenceEngine::TBlob<float>::Ptr output;
    output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
    output->allocate();
    outputBlobs[item.first] = output;

    graph.Infer(srcs, outputBlobs);

    InferenceEngine::TBlob<float>::Ptr dstOut = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc(), refDst.data());

    compare(*output, *dstOut);
}

TEST_F(MKLDNNGraphStructureTests, TestGemmConvolutionWith2DepthwiseOpFusing) {
    std::string model = R"V0G0N(
<net name="net" version="2" batch="1">
    <layers>
        <layer name="data" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>8</dim>
                    <dim>300</dim>
                    <dim>600</dim>
                </port>
            </output>
        </layer>
        <layer name="conv" type="Convolution" precision="FP32" id="1">
            <convolution_data stride-x="1" stride-y="1" pad-x="0" pad-y="0" kernel-x="1" kernel-y="1" output="8" group="2"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>8</dim>
                    <dim>300</dim>
                    <dim>600</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>8</dim>
                    <dim>300</dim>
                    <dim>600</dim>
                </port>
            </output>
            <weights offset="0" size="128"/>
            <biases offset="128" size="32"/>
        </layer>
        <layer name="depthwise0" type="PReLU" precision="FP32" id="2">
            <data channel_shared="1"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>8</dim>
                    <dim>300</dim>
                    <dim>600</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>8</dim>
                    <dim>300</dim>
                    <dim>600</dim>
                </port>
            </output>
            <weights offset="160" size="4"/>
        </layer>
        <layer name="depthwise1" type="ScaleShift" precision="FP32" id="3">
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>8</dim>
                    <dim>300</dim>
                    <dim>600</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>8</dim>
                    <dim>300</dim>
                    <dim>600</dim>
                </port>
            </output>
            <weights offset="164" size="32"/>
            <biases offset="196" size="32"/>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>
        <edge from-layer="2" from-port="1" to-layer="3" to-port="0"/>
    </edges>
</net>
)V0G0N";

    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(model.data(), model.length());

    InferenceEngine::TBlob<uint8_t> *weights = new InferenceEngine::TBlob<uint8_t>({ InferenceEngine::Precision::U8, {228}, InferenceEngine::C });
    weights->allocate();
    float* wdata = weights->buffer();

    for (int i = 0; i < weights->size() / sizeof(float); i++)
        wdata[i] = 1;
    wdata[40] = 2; // 2 for prelu weights

    InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr = InferenceEngine::TBlob<uint8_t>::Ptr(weights);

    net_reader.SetWeights(weights_ptr);

    MKLDNNGraphTestClass graph;
    graph.CreateGraph(net_reader.getNetwork());

    const auto& nodes = graph.getNodes();
    ASSERT_EQ(nodes.size(), 3);
    ASSERT_EQ(nodes[0].get()->getType(), MKLDNNPlugin::Type::Input);
    ASSERT_EQ(nodes[1].get()->getType(), MKLDNNPlugin::Type::Convolution);
    ASSERT_TRUE(nodes[1].get()->isFusedWith(MKLDNNPlugin::Type::Depthwise));
    ASSERT_EQ(nodes[2].get()->getType(), MKLDNNPlugin::Type::Output);

    InferenceEngine::TensorDesc src_desc(InferenceEngine::Precision::FP32, {1, 8, 300, 600}, InferenceEngine::NCHW);
    InferenceEngine::Blob::Ptr src = InferenceEngine::make_shared_blob<float>(src_desc);
    src->allocate();
    float* sdata = src->buffer().as<float *>();
    for (size_t i = 0; i < src->size(); i++) {
        sdata[i] = -1;
    }

    std::vector<float> refDst(1 * 8 * 300 * 600);
    for (size_t i = 0; i < refDst.size(); i++) {
        refDst[i] = -5; // (-4 + 1) * 2 * 1 + 1
    }

    InferenceEngine::BlobMap srcs;
    srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("data", src));

    InferenceEngine::OutputsDataMap out = net_reader.getNetwork().getOutputsInfo();

    InferenceEngine::BlobMap outputBlobs;
    std::pair<std::string, InferenceEngine::DataPtr> item = *out.begin();

    InferenceEngine::TBlob<float>::Ptr output;
    output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
    output->allocate();
    outputBlobs[item.first] = output;

    graph.Infer(srcs, outputBlobs);

    InferenceEngine::TBlob<float>::Ptr dstOut = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc(), refDst.data());

    compare(*output, *dstOut);
}

TEST_F(MKLDNNGraphStructureTests, TestCreateGraphWithSplit) {
    std::string model = R"V0G0N(
<net name="net" version="2" batch="1">
    <layers>
        <layer name="data" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>8</dim>
                    <dim>8</dim>
                </port>
            </output>
        </layer>
        <layer id="71" name="Split" precision="FP32" type="Split">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>2</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>1</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>1</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="71" to-port="0"/>
    </edges>
</net>
)V0G0N";

    const size_t batchHeight = 8;
    const size_t batchWidth = 8;
    const InferenceEngine::TensorDesc tensorDesc(InferenceEngine::Precision::FP32, { 1, 2, batchHeight, batchWidth }, InferenceEngine::NCHW);
    const size_t batchSize = batchHeight * batchWidth;
    const float channel1Value = 1.0;
    const float channel2Value = 2.0;

    InferenceEngine::Blob::Ptr inputBlob = InferenceEngine::make_shared_blob<float>(tensorDesc);
    inputBlob->allocate();
    float* inputData = inputBlob->buffer().as<float *>();
    for (size_t i = 0; i < inputBlob->size(); i++) {
        inputData[i] = (i < batchSize) ? channel1Value : channel2Value;
    }

    InferenceEngine::CNNNetReader reader;
    reader.ReadNetwork(model.data(), model.size());

    InferenceEngine::TBlob<uint8_t>* weights = new InferenceEngine::TBlob<uint8_t>({ InferenceEngine::Precision::U8, { 228 }, InferenceEngine::C });
    weights->allocate();
    float* weightsData = weights->buffer();
    for (size_t i = 0ULL; i < weights->size() / sizeof(float); i++) {
        weightsData[i] = 1.0;
    }

    const InferenceEngine::TBlob<uint8_t>::Ptr weightsPtr = InferenceEngine::TBlob<uint8_t>::Ptr(weights);
    reader.SetWeights(weightsPtr);

    MKLDNNGraphTestClass graph;
    graph.CreateGraph(reader.getNetwork());

    const auto& nodes = graph.getNodes();
    ASSERT_EQ(nodes.size(), 5);
    ASSERT_EQ(nodes[0].get()->getType(), MKLDNNPlugin::Type::Input);
    ASSERT_EQ(nodes[1].get()->getType(), MKLDNNPlugin::Type::Split);
    ASSERT_EQ(nodes[2].get()->getType(), MKLDNNPlugin::Type::Reorder);
    ASSERT_EQ(nodes[3].get()->getType(), MKLDNNPlugin::Type::Output);
    ASSERT_EQ(nodes[4].get()->getType(), MKLDNNPlugin::Type::Output);

    InferenceEngine::OutputsDataMap outputs = reader.getNetwork().getOutputsInfo();
    const std::pair<std::string, InferenceEngine::DataPtr> splitOutputItem1 {"Split.0", outputs["Split.0"]};
    const std::pair<std::string, InferenceEngine::DataPtr> splitOutputItem2 {"Split.1", outputs["Split.1"]};

    std::vector<float> splitExpectedOutputData1(batchSize);
    std::vector<float> splitExpectedOutputData2(batchSize);
    for (size_t i = 0; i < splitExpectedOutputData1.size(); i++) {
        splitExpectedOutputData1[i] = 1.0;
        splitExpectedOutputData2[i] = 2.0;
    }
    const InferenceEngine::TBlob<float>::Ptr splitExpectedOutputBlob1 = InferenceEngine::make_shared_blob<float>(
        splitOutputItem1.second->getTensorDesc(),
        splitExpectedOutputData1.data());
    const InferenceEngine::TBlob<float>::Ptr splitExpectedOutputBlob2 = InferenceEngine::make_shared_blob<float>(
        splitOutputItem2.second->getTensorDesc(),
        splitExpectedOutputData2.data());

    InferenceEngine::BlobMap outputBlobs;

    // Reshape
    InferenceEngine::TBlob<float>::Ptr splitOutputBlob1 = InferenceEngine::make_shared_blob<float>(splitOutputItem1.second->getTensorDesc());
    splitOutputBlob1->allocate();
    outputBlobs[splitOutputItem1.first] = splitOutputBlob1;

    // Split
    InferenceEngine::TBlob<float>::Ptr splitOutputBlob2 = InferenceEngine::make_shared_blob<float>(splitOutputItem2.second->getTensorDesc());
    splitOutputBlob2->allocate();
    outputBlobs[splitOutputItem2.first] = splitOutputBlob2;

    const InferenceEngine::BlobMap inputsBlobMap = { std::pair<std::string, InferenceEngine::Blob::Ptr>("data", inputBlob) };
    graph.Infer(inputsBlobMap, outputBlobs);

    compare(*splitOutputBlob1, *splitExpectedOutputBlob1);
    compare(*splitOutputBlob2, *splitExpectedOutputBlob2);
}

TEST_F(MKLDNNGraphStructureTests, TestCreateGraphWithFakeOutput) {
    std::string modelTemplate = R"V0G0N(
<net name="net" version="2" batch="1">
    <layers>
        <layer name="data" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>8</dim>
                    <dim>8</dim>
                </port>
            </output>
        </layer>
        <layer id="71" name="Split" precision="FP32" type="Split">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>2</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>1</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>1</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="72" name="Reshape" precision="FP32" type="Reshape">
			<data axis="0" dim="1,64,64" num_axes="-1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="71" to-port="0"/>
        <edge from-layer="71" from-port="%d" to-layer="72" to-port="0"/>
    </edges>
</net>
)V0G0N";

    const size_t bufferForValues = 1024;
    std::vector<char> model(modelTemplate.size() + bufferForValues);

    const size_t batchHeight = 8;
    const size_t batchWidth = 8;
    const InferenceEngine::TensorDesc tensorDesc(InferenceEngine::Precision::FP32, { 1, 2, batchHeight, batchWidth }, InferenceEngine::NCHW);
    const size_t batchSize = batchHeight * batchWidth;
    const float channel1Value = 1.0;
    const float channel2Value = 2.0;

    InferenceEngine::Blob::Ptr inputBlob = InferenceEngine::make_shared_blob<float>(tensorDesc);
    inputBlob->allocate();
    float* inputData = inputBlob->buffer().as<float *>();
    for (size_t i = 0; i < inputBlob->size(); i++) {
        inputData[i] = (i < batchSize) ? channel1Value : channel2Value;
    }

    for (int splitFromPortNumber = 1; splitFromPortNumber <= 2; ++splitFromPortNumber) {
        sprintf(model.data(), modelTemplate.c_str(), splitFromPortNumber);

        InferenceEngine::CNNNetReader reader;
        reader.ReadNetwork(model.data(), model.size());

        InferenceEngine::TBlob<uint8_t> *weights = new InferenceEngine::TBlob<uint8_t>({ InferenceEngine::Precision::U8, { 228 }, InferenceEngine::C });
        weights->allocate();
        float* weightsData = weights->buffer();
        for (size_t i = 0ULL; i < weights->size() / sizeof(float); i++) {
            weightsData[i] = 1.0;
        }

        const InferenceEngine::TBlob<uint8_t>::Ptr weightsPtr = InferenceEngine::TBlob<uint8_t>::Ptr(weights);
        reader.SetWeights(weightsPtr);

        MKLDNNGraphTestClass graph;
        graph.CreateGraph(reader.getNetwork());

        InferenceEngine::OutputsDataMap outputs = reader.getNetwork().getOutputsInfo();
        const std::pair<std::string, InferenceEngine::DataPtr> reshapeOutputItem = std::make_pair("Reshape", outputs["Reshape"]);
        const std::string splitOutputName = std::string("Split.") + (splitFromPortNumber == 1 ? "1" : "0");
        const std::pair<std::string, InferenceEngine::DataPtr> splitOutputItem = std::make_pair(splitOutputName, outputs[splitOutputName]);

        std::vector<float> reshapeExpectedOutputData(batchSize);
        std::vector<float> splitExpectedOutputData(batchSize);
        for (size_t i = 0; i < reshapeExpectedOutputData.size(); i++) {
            reshapeExpectedOutputData[i] = (splitFromPortNumber == 1) ? 1.0 : 2.0;
            splitExpectedOutputData[i] = (splitFromPortNumber == 1) ? 2.0 : 1.0;
        }
        const InferenceEngine::TBlob<float>::Ptr reshapeExpectedOutputBlob = InferenceEngine::make_shared_blob<float>(
            reshapeOutputItem.second->getTensorDesc(),
            reshapeExpectedOutputData.data());
        const InferenceEngine::TBlob<float>::Ptr splitExpectedOutputBlob = InferenceEngine::make_shared_blob<float>(
            splitOutputItem.second->getTensorDesc(),
            splitExpectedOutputData.data());

        InferenceEngine::BlobMap outputBlobs;

        // Reshape
        InferenceEngine::TBlob<float>::Ptr reshapeOutputBlob = InferenceEngine::make_shared_blob<float>(reshapeOutputItem.second->getTensorDesc());
        reshapeOutputBlob->allocate();
        outputBlobs[reshapeOutputItem.first] = reshapeOutputBlob;

        // Split
        InferenceEngine::TBlob<float>::Ptr splitOutputBlob = InferenceEngine::make_shared_blob<float>(splitOutputItem.second->getTensorDesc());
        splitOutputBlob->allocate();
        outputBlobs[splitOutputItem.first] = splitOutputBlob;

        const InferenceEngine::BlobMap inputsBlobMap = { std::pair<std::string, InferenceEngine::Blob::Ptr>("data", inputBlob) };
        graph.Infer(inputsBlobMap, outputBlobs);

        compare(*reshapeOutputBlob, *reshapeExpectedOutputBlob);
        compare(*splitOutputBlob, *splitExpectedOutputBlob);
    }
}

TEST_F(MKLDNNGraphStructureTests, TestCreateGraphWithMultipleData) {
    std::string model = R"V0G0N(
<net name="net" version="2" batch="1">
    <layers>
        <layer name="data" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>8</dim>
                    <dim>8</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="split" precision="FP32" type="Split">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>2</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>1</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>1</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="reshape1" precision="FP32" type="Reshape">
			<data axis="0" dim="1,64,64" num_axes="-1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
        <layer id="3" name="reshape2" precision="FP32" type="Reshape">
			<data axis="0" dim="1,64,64" num_axes="-1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
        <layer id="4" name="reshape3" precision="FP32" type="Reshape">
			<data axis="0" dim="1,64,64" num_axes="-1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="2" to-layer="4" to-port="0"/>
    </edges>
</net>
)V0G0N";

    const size_t batchHeight = 8;
    const size_t batchWidth = 8;
    const InferenceEngine::TensorDesc tensorDesc(InferenceEngine::Precision::FP32, { 1, 2, batchHeight, batchWidth }, InferenceEngine::NCHW);
    const size_t batchSize = batchHeight * batchWidth;
    const float channel1Value = 1.0;
    const float channel2Value = 2.0;

    InferenceEngine::Blob::Ptr inputBlob = InferenceEngine::make_shared_blob<float>(tensorDesc);
    inputBlob->allocate();
    float* inputData = inputBlob->buffer().as<float *>();
    for (size_t i = 0; i < inputBlob->size(); i++) {
        inputData[i] = (i < batchSize) ? channel1Value : channel2Value;
    }


    InferenceEngine::CNNNetReader reader;
    reader.ReadNetwork(model.data(), model.size());

    InferenceEngine::TBlob<uint8_t> *weights = new InferenceEngine::TBlob<uint8_t>({ InferenceEngine::Precision::U8, { 228 }, InferenceEngine::C });
    weights->allocate();
    float* weightsData = weights->buffer();
    for (size_t i = 0ULL; i < weights->size() / sizeof(float); i++) {
        weightsData[i] = 1.0;
    }

    const InferenceEngine::TBlob<uint8_t>::Ptr weightsPtr = InferenceEngine::TBlob<uint8_t>::Ptr(weights);
    reader.SetWeights(weightsPtr);

    reader.getNetwork().addOutput("split");

    MKLDNNGraphTestClass graph;
    graph.CreateGraph(reader.getNetwork());

    const auto& nodes = graph.getNodes();
    ASSERT_EQ(nodes.size(), 12);
    ASSERT_EQ(nodes[0]->getType(), MKLDNNPlugin::Type::Input);
    ASSERT_EQ(nodes[1]->getType(), MKLDNNPlugin::Type::Split);
    ASSERT_EQ(nodes[2]->getType(), MKLDNNPlugin::Type::Reorder);
    ASSERT_EQ(nodes[3]->getType(), MKLDNNPlugin::Type::Reshape);
    ASSERT_EQ(nodes[4]->getType(), MKLDNNPlugin::Type::Output);
    ASSERT_EQ(nodes[5]->getType(), MKLDNNPlugin::Type::Reorder);
    ASSERT_EQ(nodes[6]->getType(), MKLDNNPlugin::Type::Reshape);
    ASSERT_EQ(nodes[7]->getType(), MKLDNNPlugin::Type::Output);
    ASSERT_EQ(nodes[8]->getType(), MKLDNNPlugin::Type::Reorder);
    ASSERT_EQ(nodes[9]->getType(), MKLDNNPlugin::Type::Reshape);
    ASSERT_EQ(nodes[10]->getType(), MKLDNNPlugin::Type::Output);
    ASSERT_EQ(nodes[11]->getType(), MKLDNNPlugin::Type::Output);

    InferenceEngine::OutputsDataMap outputs = reader.getNetwork().getOutputsInfo();
    std::vector<std::pair<std::string, InferenceEngine::DataPtr>> outputItems = {
        std::make_pair("reshape1", outputs.find("reshape1")->second),
        std::make_pair("reshape2", outputs.find("reshape2")->second),
        std::make_pair("reshape3", outputs.find("reshape3")->second),
        std::make_pair("split.0", outputs.find("split.0")->second)
    };

    std::vector<std::vector<float>> expectedOutputData = {
        std::vector<float>(batchSize),
        std::vector<float>(batchSize),
        std::vector<float>(batchSize),
        std::vector<float>(batchSize)
    };
    for (size_t i = 0; i < batchSize; i++) {
        expectedOutputData[0][i] = channel1Value;
        expectedOutputData[1][i] = channel1Value;
        expectedOutputData[2][i] = channel2Value;

        expectedOutputData[3][i] = channel1Value;
    }

    std::vector<InferenceEngine::TBlob<float>::Ptr> expectedOutputBlobs(outputs.size());
    for (size_t i = 0; i < outputs.size(); i++) {
        expectedOutputBlobs[i] = InferenceEngine::make_shared_blob<float>(
            outputItems[i].second->getTensorDesc(),
            expectedOutputData[i].data());
    }

    std::vector<InferenceEngine::TBlob<float>::Ptr> outputBlobs;
    outputBlobs.reserve(outputItems.size());

    InferenceEngine::BlobMap outputBlobsMap;
    for(const std::pair<std::string, InferenceEngine::DataPtr>& item : outputItems) {
        InferenceEngine::TBlob<float>::Ptr blob = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
        outputBlobs.push_back(blob);
        blob->allocate();
        outputBlobsMap[item.first] = blob;
    }

    const InferenceEngine::BlobMap inputsBlobMap = { std::pair<std::string, InferenceEngine::Blob::Ptr>("data", inputBlob) };
    graph.Infer(inputsBlobMap, outputBlobsMap);

    for(size_t i = 0; i < 3; i++) {
        compare(*outputBlobs[i], *expectedOutputBlobs[i]);
    }
}

TEST_F(MKLDNNGraphStructureTests, TestCreateGraphWithMultipleData_2) {
    std::string model = R"V0G0N(
<net name="net" version="2" batch="1">
    <layers>
        <layer name="data" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>8</dim>
                    <dim>8</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="split" precision="FP32" type="Split">
			<data axis="1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>2</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>1</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>1</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="power" precision="FP32" type="Power">
			<data power="1" scale="-1.0" shift="0.0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</input>
			<output>
				<port id="1">
                    <dim>1</dim>
					<dim>1</dim>
					<dim>8</dim>
					<dim>8</dim>
				</port>
			</output>
		</layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    using namespace InferenceEngine;

    const size_t H = 8;
    const size_t W = 8;
    const size_t imgSz = H * W;
    const float channel1Value = 1.0;
    const float channel2Value = 2.0;

    const auto weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, SizeVector{0}, Layout::C));

    InferenceEngine::CNNNetReader reader;
    reader.ReadNetwork(model.data(), model.size());
    reader.SetWeights(weights);

    auto net = reader.getNetwork();
    net.addOutput("split", 0);

    MKLDNNGraphTestClass graph;
    graph.CreateGraph(net);

    auto inBlob   = make_shared_blob<float>({ Precision::FP32, SizeVector{1, 2, H, W}, Layout::NCHW });
    auto outBlob1 = make_shared_blob<float>({ Precision::FP32, SizeVector{1, 1, H, W}, Layout::NCHW });
    auto outBlob2 = make_shared_blob<float>({ Precision::FP32, SizeVector{1, 1, H, W}, Layout::NCHW });
    auto outBlob3 = make_shared_blob<float>({ Precision::FP32, SizeVector{1, 1, H, W}, Layout::NCHW });

    inBlob->allocate();
    outBlob1->allocate();
    outBlob2->allocate();
    outBlob3->allocate();

    auto in_ptr = inBlob->buffer().as<float*>();
    for (int i = 0; i < imgSz; i++) {
        in_ptr[i] = channel1Value;
        in_ptr[i + imgSz] = channel2Value;
    }

    BlobMap inputBlobMap  = { {"data"   , inBlob  } },
            outputBlobMap = { {"split.0", outBlob1},
                              {"split.1", outBlob2},
                              {"power"  , outBlob3} };

    graph.Infer(inputBlobMap, outputBlobMap);

    auto out_check = [] ( Blob::Ptr blob, float val) {
        auto size = blob->size();
        auto ptr = blob->buffer().as<float*>();
        bool res = true;
        for (int i = 0; i < size; i++)
            res &= ( std::abs( ptr[i] - val ) < 0.00001f );
        return res;
    };

    EXPECT_TRUE(out_check(outBlob1,  1));
    EXPECT_TRUE(out_check(outBlob2,  2));
    EXPECT_TRUE(out_check(outBlob3, -1));
}

TEST_F(MKLDNNGraphStructureTests, TestCreateGraphAllDataToConcat) {
    IE_SUPPRESS_DEPRECATED_START

    using namespace InferenceEngine;
    // Build the network.
    Builder::Network netBuilder("");

    // First input layer
    idx_t inpId = netBuilder.addLayer(InferenceEngine::Builder::InputLayer("input").setPort(InferenceEngine::Port({1, 1, 4, 5})));

    std::vector<size_t> weightsSize = {1, 1, 1, 1};  // OIHW
    std::vector<float> twos(1, 2);
    auto weights = make_shared_blob<float>({ Precision::FP32, weightsSize, InferenceEngine::Layout::OIHW }, &twos[0]);

    idx_t weightsId = netBuilder.addLayer({}, Builder::ConstLayer("weights").setData(weights));

    // Convolution layer
    idx_t firstConvId = netBuilder.addLayer({{inpId}, {weightsId}}, Builder::ConvolutionLayer("conv").setKernel({1, 1})
            .setStrides({1, 1}).setDilation({1, 1}).setPaddingsBegin({0, 0}).setPaddingsEnd({0, 0}).setGroup(1).setOutDepth(1));

    std::vector<float> threes(1, 3);
    weights = make_shared_blob<float>({ Precision::FP32, weightsSize, InferenceEngine::Layout::OIHW }, &threes[0]);

    weightsId = netBuilder.addLayer({}, Builder::ConstLayer("weights").setData(weights));
    // Convolution layer
    idx_t secondConvId = netBuilder.addLayer({{inpId}, {weightsId}}, Builder::ConvolutionLayer("conv").setKernel({1, 1})
            .setStrides({1, 1}).setDilation({1, 1}).setPaddingsBegin({0, 0}).setPaddingsEnd({0, 0}).setGroup(1).setOutDepth(1));

    // Concat layer
    idx_t concatId = netBuilder.addLayer({{inpId}, {firstConvId}, {secondConvId}},
                                         InferenceEngine::Builder::ConcatLayer("concat").setAxis(1).setInputPorts(std::vector<InferenceEngine::Port>(3)));

    // Output layer
    InferenceEngine::Builder::OutputLayer outLayer("output");
    netBuilder.addLayer({concatId}, outLayer);

    auto cnn = CNNNetwork(Builder::convertToICNNNetwork(netBuilder.build()));

    // Load the network
    std::vector<size_t> inpSize = {5, 4, 1, 1};
    std::vector<size_t> outSize = {5, 4, 3, 1};

    InferenceEngine::BlobMap inputBlobs;
    InferenceEngine::BlobMap outputBlobs;

    std::vector<float> inpData(4*5, 1);
    std::vector<float> outData(3*4*5, 1);
    for (int i = 0; i < 4*5; ++i) {
        inpData[i] = i;
    }

    inputBlobs["input"] = InferenceEngine::make_shared_blob<float>({ InferenceEngine::Precision::FP32, inpSize,
        InferenceEngine::TensorDesc::getLayoutByDims(inpSize) }, &inpData[0]);
    outputBlobs["concat"] = InferenceEngine::make_shared_blob<float>({ InferenceEngine::Precision::FP32, outSize,
        InferenceEngine::TensorDesc::getLayoutByDims(outSize) }, &outData[0]);

    MKLDNNGraphTestClass graph;
    graph.CreateGraph(cnn);
    graph.Infer(inputBlobs, outputBlobs);

    std::vector<float> refDst = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38,
                                 0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57};

    InferenceEngine::TBlob<float>::Ptr dstOut = InferenceEngine::make_shared_blob<float>(outputBlobs["concat"]->getTensorDesc(), refDst.data());

    compare(*outputBlobs["concat"], *dstOut);
}

TEST_F(MKLDNNGraphStructureTests, TestCreateGraphAllDataFromInputToConcat) {
    using namespace InferenceEngine;
    // Build the network.
    Builder::Network netBuilder("");

    // First input layer
    idx_t inpId = netBuilder.addLayer(InferenceEngine::Builder::InputLayer("input").setPort(InferenceEngine::Port({1, 1, 4, 5})));

    // Concat layer
    idx_t concatId = netBuilder.addLayer({{inpId}, {inpId}, {inpId}},
                                         InferenceEngine::Builder::ConcatLayer("concat").setAxis(1).setInputPorts(std::vector<InferenceEngine::Port>(3)));

    // Output layer
    InferenceEngine::Builder::OutputLayer outLayer("output");
    netBuilder.addLayer({concatId}, outLayer);

    auto cnn = CNNNetwork(Builder::convertToICNNNetwork(netBuilder.build()));

    // Load the network
    std::vector<size_t> inpSize = {5, 4, 1, 1};
    std::vector<size_t> outSize = {5, 4, 3, 1};

    InferenceEngine::BlobMap inputBlobs;
    InferenceEngine::BlobMap outputBlobs;

    std::vector<float> inpData(4*5, 1);
    std::vector<float> outData(3*4*5, 1);
    for (int i = 0; i < 4*5; ++i)
    {
        inpData[i] = i;
    }

    inputBlobs["input"] = InferenceEngine::make_shared_blob<float>({ InferenceEngine::Precision::FP32, inpSize,
        InferenceEngine::TensorDesc::getLayoutByDims(inpSize) }, &inpData[0]);
    outputBlobs["concat"] = InferenceEngine::make_shared_blob<float>({ InferenceEngine::Precision::FP32, outSize,
        InferenceEngine::TensorDesc::getLayoutByDims(outSize) }, &outData[0]);


    MKLDNNGraphTestClass graph;
    graph.CreateGraph(cnn);
    graph.Infer(inputBlobs, outputBlobs);

    std::vector<float> refDst = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,};

    InferenceEngine::TBlob<float>::Ptr dstOut = InferenceEngine::make_shared_blob<float>(outputBlobs["concat"]->getTensorDesc(), refDst.data());

    compare(*outputBlobs["concat"], *dstOut);

    IE_SUPPRESS_DEPRECATED_END
}


TEST_F(MKLDNNGraphStructureTests, TestCheckIncorrectScaleShift) {
    std::string model = R"V0G0N(
<net name="net" version="2" batch="1">
    <layers>
        <layer name="data" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>1000</dim>
                    <dim>16</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="test" precision="FP32" type="ScaleShift">
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>1000</dim>
                    <dim>16</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>100</dim>
                    <dim>16</dim>
                </port>
            </output>
            <blobs>
                <weights offset="0" size="64"/>
                <biases offset="0" size="64"/>
            </blobs>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
    </edges>
</net>
)V0G0N";
    using namespace InferenceEngine;
    const auto weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, SizeVector{64}, Layout::C));

    InferenceEngine::CNNNetReader reader;
    reader.ReadNetwork(model.data(), model.size());
    reader.SetWeights(weights);

    MKLDNNGraphTestClass graph;
    ASSERT_THROW(graph.CreateGraph(reader.getNetwork()), InferenceEngine::details::InferenceEngineException);
}

TEST_F(MKLDNNGraphStructureTests, TestConcatWithFourInputs) {
    IE_SUPPRESS_DEPRECATED_START

    using namespace InferenceEngine;
    // Build the network.
    Builder::Network netBuilder("");

    // First input layer
    idx_t inpId = netBuilder.addLayer(InferenceEngine::Builder::InputLayer("input").setPort(InferenceEngine::Port({1, 1, 4, 5})));

    std::vector<size_t> weightsSize = {1, 1, 1, 1};  // OIHW
    std::vector<float> twos(1, 2);
    auto weights = make_shared_blob<float>({ Precision::FP32, weightsSize, InferenceEngine::Layout::OIHW }, &twos[0]);
    idx_t weightsId = netBuilder.addLayer({}, Builder::ConstLayer("weights").setData(weights));

    // Convolution layer
    idx_t firstConvId = netBuilder.addLayer({{inpId}, {weightsId}}, Builder::ConvolutionLayer("conv").setKernel({1, 1})
            .setStrides({1, 1}).setDilation({1, 1}).setPaddingsBegin({0, 0}).setPaddingsEnd({0, 0}).setGroup(1).setOutDepth(1));

    std::vector<float> threes(1, 3);
    weights = make_shared_blob<float>({ Precision::FP32, weightsSize, InferenceEngine::Layout::OIHW }, &threes[0]);

    weightsId = netBuilder.addLayer({}, Builder::ConstLayer("weights").setData(weights));
    // Convolution layer
    idx_t secondConvId = netBuilder.addLayer({{inpId}, {weightsId}}, Builder::ConvolutionLayer("conv").setKernel({1, 1})
            .setStrides({1, 1}).setDilation({1, 1}).setPaddingsBegin({0, 0}).setPaddingsEnd({0, 0}).setGroup(1).setOutDepth(1));

    std::vector<float> four(1, -1);
    weights = make_shared_blob<float>({ Precision::FP32, weightsSize, InferenceEngine::Layout::OIHW }, &four[0]);

    weightsId = netBuilder.addLayer({}, Builder::ConstLayer("weights").setData(weights));
    // Convolution layer
    idx_t thirdConvId = netBuilder.addLayer({{inpId}, {weightsId}}, Builder::ConvolutionLayer("conv").setKernel({1, 1})
            .setStrides({1, 1}).setDilation({1, 1}).setPaddingsBegin({0, 0}).setPaddingsEnd({0, 0}).setGroup(1).setOutDepth(1));

    // Concat layer
    idx_t concatId = netBuilder.addLayer({{inpId}, {firstConvId}, {secondConvId}, {thirdConvId}},
                                         InferenceEngine::Builder::ConcatLayer("concat").setAxis(1).setInputPorts(std::vector<InferenceEngine::Port>(4)));

    // Output layer
    InferenceEngine::Builder::OutputLayer outLayer("output");
    netBuilder.addLayer({concatId}, outLayer);

    auto cnn = CNNNetwork(Builder::convertToICNNNetwork(netBuilder.build()));

    // Load the network
    std::vector<size_t> inpSize = {5, 4, 1, 1};
    std::vector<size_t> outSize = {5, 4, 4, 1};

    InferenceEngine::BlobMap inputBlobs;
    InferenceEngine::BlobMap outputBlobs;

    std::vector<float> inpData(4*5, 1);
    std::vector<float> outData(4*4*5, 1);
    for (int i = 0; i < 4*5; ++i) {
        inpData[i] = i;
    }

    inputBlobs["input"] = InferenceEngine::make_shared_blob<float>({ InferenceEngine::Precision::FP32, inpSize,
        InferenceEngine::TensorDesc::getLayoutByDims(inpSize) }, &inpData[0]);
    outputBlobs["concat"] = InferenceEngine::make_shared_blob<float>({ InferenceEngine::Precision::FP32, outSize,
        InferenceEngine::TensorDesc::getLayoutByDims(outSize) }, &outData[0]);


    MKLDNNGraphTestClass graph;
    graph.CreateGraph(cnn);
    graph.Infer(inputBlobs, outputBlobs);

    std::vector<float> refDst = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38,
                                 0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57,
                                 0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18, -19};

    InferenceEngine::TBlob<float>::Ptr dstOut = InferenceEngine::make_shared_blob<float>(outputBlobs["concat"]->getTensorDesc(), refDst.data());

    compare(*outputBlobs["concat"], *dstOut);

    IE_SUPPRESS_DEPRECATED_END
}
