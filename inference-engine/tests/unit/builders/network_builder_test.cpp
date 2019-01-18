// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <string.h>
#include <ie_builders.hpp>


#include "builder_test.hpp"

using namespace testing;
using namespace InferenceEngine;

class NetworkBuilderTest : public BuilderTestCommon {
protected:
    std::vector<std::string> alexNetNames = {
            "in1",
            "mean",
            "conv1",
            "relu1",
            "norm1",
            "pool1",
            "conv2",
            "relu2",
            "norm2",
            "pool2",
            "conv3",
            "relu3",
            "conv4",
            "relu4",
            "conv5",
            "relu5",
            "pool5",
            "fc6",
            "relu6",
            "fc7",
            "relu7",
            "fc8",
            "prob",
            "sf_out"
    };

public:

    Builder::Network prepateAlexnetBuilder() {
        Context ctx;
        Builder::Network builder(ctx, "AlexNet");
        idx_t layerId = builder.addLayer(Builder::InputLayer(alexNetNames[0]).setPort(Port({1,3, 227, 227})));
        layerId = builder.addLayer({{layerId}}, Builder::ScaleShiftLayer(alexNetNames[1]).setBiases(generateBlob(Precision::FP32, {3}, Layout::C)));
        layerId = builder.addLayer({{layerId}}, Builder::ConvolutionLayer(alexNetNames[2]).setKernel({11, 11}).setStrides({4, 4}).setOutDepth(96)
                .setWeights(generateBlob(Precision::FP32, {96, 3, 11, 11}, Layout::OIHW))
                .setBiases(generateBlob(Precision::FP32, {96}, Layout::C)));
        layerId = builder.addLayer({{layerId}}, Builder::ReLULayer(alexNetNames[3]));
        layerId = builder.addLayer({{layerId}}, Builder::NormLayer(alexNetNames[4]).setAlpha(9.999999747378752e-05f).setBeta(0.75f).setSize(5).setAcrossMaps(true));
        layerId = builder.addLayer({{layerId}}, Builder::PoolingLayer(alexNetNames[5]).setExcludePad(false).setKernel({3, 3}).setPaddingsBegin({0, 0})
                .setPaddingsEnd({0, 0}).setPoolingType(Builder::PoolingLayer::PoolingType::MAX).setStrides({2, 2}));
        layerId = builder.addLayer({{layerId}}, Builder::ConvolutionLayer(alexNetNames[6]).setKernel({5, 5}).setStrides({1, 1}).setOutDepth(256)
                .setPaddingsBegin({2, 2}).setPaddingsEnd({2, 2}).setGroup(2).setDilation({1, 1})
                .setWeights(generateBlob(Precision::FP32, {96, 256, 5, 5}, Layout::OIHW))
                .setBiases(generateBlob(Precision::FP32, {256}, Layout::C)));
        layerId = builder.addLayer({{layerId}}, Builder::ReLULayer(alexNetNames[7]));
        layerId = builder.addLayer({{layerId}}, Builder::NormLayer(alexNetNames[8]).setAlpha(9.999999747378752e-05f).setBeta(0.75f).setSize(5).setAcrossMaps(true));
        layerId = builder.addLayer({{layerId}}, Builder::PoolingLayer(alexNetNames[9]).setExcludePad(false).setKernel({3, 3}).setPaddingsBegin({0, 0})
                .setPaddingsEnd({0, 0}).setPoolingType(Builder::PoolingLayer::PoolingType::MAX).setStrides({2, 2}));
        layerId = builder.addLayer({{layerId}}, Builder::ConvolutionLayer(alexNetNames[10]).setKernel({3, 3}).setStrides({1, 1}).setOutDepth(384)
                .setPaddingsBegin({1, 1}).setPaddingsEnd({1, 1}).setGroup(1).setDilation({1, 1})
                .setWeights(generateBlob(Precision::FP32, {256, 384, 3, 3}, Layout::OIHW))
                .setBiases(generateBlob(Precision::FP32, {384}, Layout::C)));
        layerId = builder.addLayer({{layerId}}, Builder::ReLULayer(alexNetNames[11]));
        layerId = builder.addLayer({{layerId}}, Builder::ConvolutionLayer(alexNetNames[12]).setKernel({3, 3}).setStrides({1, 1}).setOutDepth(384)
                .setPaddingsBegin({1, 1}).setPaddingsEnd({1, 1}).setGroup(2).setDilation({1, 1})
                .setWeights(generateBlob(Precision::FP32, {384, 384, 3, 3}, Layout::OIHW))
                .setBiases(generateBlob(Precision::FP32, {384}, Layout::C)));
        layerId = builder.addLayer({{layerId}}, Builder::ReLULayer(alexNetNames[13]));
        layerId = builder.addLayer({{layerId}}, Builder::ConvolutionLayer(alexNetNames[14]).setKernel({3, 3}).setStrides({1, 1}).setOutDepth(256)
                .setPaddingsBegin({1, 1}).setPaddingsEnd({1, 1}).setGroup(2).setDilation({1, 1})
                .setWeights(generateBlob(Precision::FP32, {256, 384, 3, 3}, Layout::OIHW))
                .setBiases(generateBlob(Precision::FP32, {384}, Layout::C)));
        layerId = builder.addLayer({{layerId}}, Builder::ReLULayer(alexNetNames[15]));
        layerId = builder.addLayer({{layerId}}, Builder::PoolingLayer(alexNetNames[16]).setExcludePad(false).setKernel({3, 3}).setPaddingsBegin({0, 0})
                .setPaddingsEnd({0, 0}).setPoolingType(Builder::PoolingLayer::PoolingType::MAX).setStrides({2, 2}));
        layerId = builder.addLayer({{layerId}}, Builder::FullyConnectedLayer(alexNetNames[17]).setOutputNum(4096)
                .setWeights(generateBlob(Precision::FP32, {4096, 256, 6, 6}, Layout::OIHW))
                .setBiases(generateBlob(Precision::FP32, {4096}, Layout::C)));
        layerId = builder.addLayer({{layerId}}, Builder::ReLULayer(alexNetNames[18]));
        layerId = builder.addLayer({{layerId}}, Builder::FullyConnectedLayer(alexNetNames[19]).setOutputNum(4096)
                .setWeights(generateBlob(Precision::FP32, {4096, 4096}, Layout::NC))
                .setBiases(generateBlob(Precision::FP32, {4096}, Layout::C)));
        layerId = builder.addLayer({{layerId}}, Builder::ReLULayer(alexNetNames[20]));
        layerId = builder.addLayer({{layerId}}, Builder::FullyConnectedLayer(alexNetNames[21]).setOutputNum(1000)
                .setWeights(generateBlob(Precision::FP32, {1000, 4096}, Layout::NC))
                .setBiases(generateBlob(Precision::FP32, {1000}, Layout::C)));
        layerId = builder.addLayer({{layerId}}, Builder::SoftMaxLayer(alexNetNames[22]).setAxis(1));

        idx_t outputId = builder.addLayer({PortInfo(layerId)}, Builder::OutputLayer(alexNetNames[23]));
        return builder;
    }

    const INetwork::Ptr createAlexnet() {
        return prepateAlexnetBuilder().build();
    }

    void compareWithICNNNetwork(const INetwork& network, const ICNNNetwork& cnnNetwork) {
        for (const auto& layer : network) {
            auto connections = network.getLayerConnections(layer->getId());
            CNNLayerPtr cnnLayer;
            StatusCode sts = cnnNetwork.getLayerByName(layer->getName().c_str(), cnnLayer, nullptr);
            if (sts != OK && layer->getType() == "Output")
                continue;
            else if (sts != OK)
                THROW_IE_EXCEPTION << "Cannot find CNNLayer by name: " << layer->getName();


            // Output connections
            for (size_t i = 0; i < cnnLayer->outData.size(); i++) {
                for (const auto& it : cnnLayer->outData[i]->inputTo) {
                    size_t j = 0;
                    for (; j < it.second->insData.size(); j++) {
                        auto lockedData = it.second->insData[j].lock();
                        if (lockedData && lockedData.get() == cnnLayer->outData[i].get()) {
                            break;
                        }
                    }

                    for (auto conIt = connections.begin(); conIt != connections.end(); conIt++) {
                        if (conIt->from().layerId() == layer->getId() && conIt->from().portId() == i &&
                            network.getLayer(conIt->to().layerId())->getName() == it.second->name &&
                            conIt->to().portId() == j) {
                            connections.erase(conIt);
                            break;
                        }
                    }
                }
            }

            // Input connections
            for (size_t i = 0; i < cnnLayer->insData.size(); i++) {
                auto inData = cnnLayer->insData[i].lock();
                if (!inData)
                    continue;
                auto creatorLayer = inData->creatorLayer.lock();
                if (!creatorLayer)
                    continue;
                size_t j = 0;
                for (; j < creatorLayer->outData.size(); j++) {
                    if (creatorLayer->outData[j] && creatorLayer->outData[j].get() == inData.get()) {
                        break;
                    }
                }

                for (auto conIt = connections.begin(); conIt != connections.end(); conIt++) {
                    if (conIt->to().layerId() == layer->getId() && conIt->from().portId() == j &&
                        network.getLayer(conIt->from().layerId())->getName() == creatorLayer->name &&
                        conIt->to().portId() == i) {
                        connections.erase(conIt);
                        break;
                    }
                }
            }

            if (connections.size() == 1 && network.getLayer(connections[0].to().layerId())->getType() == "Output")
                connections.erase(connections.begin());

            if (!connections.empty())
                THROW_IE_EXCEPTION << "Not all connections were connected.";
        }
    }

    void compareICNNNetworks(const ICNNNetwork& newNetwork, const ICNNNetwork& oldNetwork) {
        CNNNetwork network((ICNNNetwork*)&newNetwork);

        if (newNetwork.layerCount() != oldNetwork.layerCount())
            THROW_IE_EXCEPTION << "ICNNNetworks have different numbers of layers!";
        for (const auto& layer : network) {
            CNNLayerPtr oldLayer;
            StatusCode sts = oldNetwork.getLayerByName(layer->name.c_str(), oldLayer, nullptr);
            bool success = sts == OK && layer->name == oldLayer->name &&
                    layer->type == oldLayer->type &&
                    layer->insData.size() == oldLayer->insData.size() &&
                    layer->outData.size() == oldLayer->outData.size() &&
                    layer->precision == oldLayer->precision;

            for (size_t i = 0; i < layer->insData.size() && success; i++) {
                auto lockedOldData = oldLayer->insData[i].lock();
                auto lockedData = layer->insData[i].lock();
                success = success && lockedOldData->name == lockedData->name &&
                          lockedOldData->getTensorDesc() == lockedData->getTensorDesc();
            }
            for (size_t i = 0; i < layer->outData.size() && success; i++) {
                success = success && oldLayer->outData[i]->name == layer->outData[i]->name &&
                        oldLayer->outData[i]->getTensorDesc() == layer->outData[i]->getTensorDesc();
            }

            if (!success)
                THROW_IE_EXCEPTION << "ICNNNetworks have different layers!";
        }

        InputsDataMap newInput;
        OutputsDataMap newOutput;
        newNetwork.getInputsInfo(newInput);
        newNetwork.getOutputsInfo(newOutput);
        InputsDataMap oldInput;
        OutputsDataMap oldOutput;
        oldNetwork.getInputsInfo(oldInput);
        oldNetwork.getOutputsInfo(oldOutput);

        bool success = newInput.size() == oldInput.size();
        for (const auto& it : newInput) {
            if (!success)
                break;
            success = success && oldInput.find(it.first) != oldInput.end();
        }
        if (!success)
            THROW_IE_EXCEPTION << "ICNNNetworks have different inputs!";

        success = newOutput.size() == oldOutput.size();
        for (const auto& it : newOutput) {
            if (!success)
                break;
            success = success && oldOutput.find(it.first) != oldOutput.end();
        }
        if (!success)
            THROW_IE_EXCEPTION << "ICNNNetworks have different outputs!";
    }
};

TEST_F(NetworkBuilderTest, checkReshapeAlexNet) {
    std::map<std::string, std::vector<SizeVector>> inPorts = {
            {alexNetNames[0], {}},
            {alexNetNames[1], {{1, 3, 227, 227}}},
            {alexNetNames[2], {{1, 3, 227, 227}}},
            {alexNetNames[3], {{1, 96, 55, 55}}},
            {alexNetNames[4], {{1, 96, 55, 55}}},
            {alexNetNames[5], {{1, 96, 55, 55}}},
            {alexNetNames[6], {{1, 96, 27, 27}}},
            {alexNetNames[7], {{1, 256, 27, 27}}},
            {alexNetNames[8], {{1, 256, 27, 27}}},
            {alexNetNames[9], {{1, 256, 27, 27}}},
            {alexNetNames[10], {{1, 256, 13, 13}}},
            {alexNetNames[11], {{1, 384, 13, 13}}},
            {alexNetNames[12], {{1, 384, 13, 13}}},
            {alexNetNames[13], {{1, 384, 13, 13}}},
            {alexNetNames[14], {{1, 384, 13, 13}}},
            {alexNetNames[15], {{1, 256, 13, 13}}},
            {alexNetNames[16], {{1, 256, 13, 13}}},
            {alexNetNames[17], {{1, 256, 6, 6}}},
            {alexNetNames[18], {{1, 4096}}},
            {alexNetNames[19], {{1, 4096}}},
            {alexNetNames[20], {{1, 4096}}},
            {alexNetNames[21], {{1, 4096}}},
            {alexNetNames[22], {{1, 1000}}},
            {alexNetNames[23], {{1, 1000}}}
    };

    std::map<std::string, std::vector<SizeVector>> outPorts = {
            {alexNetNames[0], {{1, 3, 227, 227}}},
            {alexNetNames[1], {{1, 3, 227, 227}}},
            {alexNetNames[2], {{1, 96, 55, 55}}},
            {alexNetNames[3], {{1, 96, 55, 55}}},
            {alexNetNames[4], {{1, 96, 55, 55}}},
            {alexNetNames[5], {{1, 96, 27, 27}}},
            {alexNetNames[6], {{1, 256, 27, 27}}},
            {alexNetNames[7], {{1, 256, 27, 27}}},
            {alexNetNames[8], {{1, 256, 27, 27}}},
            {alexNetNames[9], {{1, 256, 13, 13}}},
            {alexNetNames[10], {{1, 384, 13, 13}}},
            {alexNetNames[11], {{1, 384, 13, 13}}},
            {alexNetNames[12], {{1, 384, 13, 13}}},
            {alexNetNames[13], {{1, 384, 13, 13}}},
            {alexNetNames[14], {{1, 256, 13, 13}}},
            {alexNetNames[15], {{1, 256, 13, 13}}},
            {alexNetNames[16], {{1, 256, 6, 6}}},
            {alexNetNames[17], {{1, 4096}}},
            {alexNetNames[18], {{1, 4096}}},
            {alexNetNames[19], {{1, 4096}}},
            {alexNetNames[20], {{1, 4096}}},
            {alexNetNames[21], {{1, 1000}}},
            {alexNetNames[22], {{1, 1000}}},
            {alexNetNames[23], {}}
    };

    Builder::Network builder = prepateAlexnetBuilder();
    for (const auto &layer : builder.getLayers()) {
        if (layer.getType() == "Input") {
            ASSERT_EQ(outPorts[layer.getName()][0], layer.getOutputPorts()[0].shape());
        } else {
            for (size_t j = 0; j < layer.getOutputPorts().size(); j++) {
                ASSERT_TRUE(layer.getOutputPorts()[j].shape().empty());
            }
        }
    }
    INetwork::Ptr graph;
    ASSERT_NO_THROW(graph = builder.build());
    for (const auto &layer : *graph) {
        for (size_t i = 0; i < layer->getInputPorts().size(); i++) {
            ASSERT_EQ(inPorts[layer->getName()][i], layer->getInputPorts()[i].shape());
        }
        for (size_t i = 0; i < layer->getOutputPorts().size(); i++) {
            ASSERT_EQ(outPorts[layer->getName()][i], layer->getOutputPorts()[i].shape());
        }
    }
}

TEST_F(NetworkBuilderTest, checkNoImplWithCorrectPorts) {
    Context ctx;
    Builder::Network builder(ctx, "TestAlexNet");
    idx_t inId = builder.addLayer(Builder::InputLayer(alexNetNames[0]).setPort(Port({1,3, 227, 227})));
    idx_t convId = builder.addLayer({{inId}}, Builder::ConvolutionLayer(alexNetNames[2]).setKernel({11, 11}).setStrides({4, 4}).setOutDepth(96)
            .setInputPort(Port({1,3, 227, 227})).setOutputPort(Port({1, 96, 55, 55}))
            .setWeights(generateBlob(Precision::FP32, {96, 3, 11, 11}, Layout::OIHW))
            .setBiases(generateBlob(Precision::FP32, {96}, Layout::C)));
    idx_t testLayerId = builder.addLayer({PortInfo(convId)}, Builder::Layer("TestLayer", "testPort")
            .setInputPorts({Port({1, 96, 55, 55})}).setOutputPorts({Port({1, 96, 55, 55})}));
    idx_t outputId = builder.addLayer({PortInfo(testLayerId)}, Builder::OutputLayer("out").setPort({Port({1, 96, 55, 55})}));

    ASSERT_NO_THROW(builder.build());
}

TEST_F(NetworkBuilderTest, checkNoImplWithIncorrectPorts) {
    Context ctx;
    Builder::Network builder(ctx, "TestAlexNet");
    idx_t inId = builder.addLayer(Builder::InputLayer(alexNetNames[0]).setPort(Port({1,3, 227, 227})));
    idx_t convId = builder.addLayer({{inId}}, Builder::ConvolutionLayer(alexNetNames[2]).setKernel({11, 11}).setStrides({4, 4}).setOutDepth(96)
            .setInputPort(Port({1,3, 227, 227})).setOutputPort(Port({1, 96, 55, 55}))
            .setWeights(generateBlob(Precision::FP32, {96, 3, 11, 11}, Layout::OIHW))
            .setBiases(generateBlob(Precision::FP32, {96}, Layout::C)));
    idx_t testLayerId = builder.addLayer({PortInfo(convId)}, Builder::Layer("TestLayer", "testPort")
            .setInputPorts({Port({1, 3, 55, 55})}).setOutputPorts({Port({1, 96, 55, 55})}));

    ASSERT_THROW(builder.build(), InferenceEngine::details::InferenceEngineException);
}

TEST_F(NetworkBuilderTest, createNetworkIterator) {
    const INetwork::Ptr graph = createAlexnet();

    ASSERT_NO_THROW(graph->begin());
}

TEST_F(NetworkBuilderTest, checkNetworkSize) {
    const INetwork::Ptr graph = createAlexnet();

    ASSERT_EQ(24, graph->size());
}

TEST_F(NetworkBuilderTest, iterateNetworkForeach) {
    const INetwork::Ptr graph = createAlexnet();

    size_t idx = 0;
    for (const auto& layer : *graph) {
        ASSERT_NE(idx, alexNetNames.size());
        ASSERT_EQ(alexNetNames[idx], layer->getName());
        idx++;
    }
}

TEST_F(NetworkBuilderTest, iterateNetworkFor) {
    const INetwork::Ptr graph = createAlexnet();

    size_t idx = 0;
    for (auto it = graph->begin(); it != graph->end(); it++) {
        ASSERT_EQ(alexNetNames[idx], (*it)->getName());
        idx++;
    }
}

TEST_F(NetworkBuilderTest, convertFromICNNNetwork) {
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

    InferenceEngine::TBlob<uint8_t> *weights = new InferenceEngine::TBlob<uint8_t>(InferenceEngine::Precision::U8, InferenceEngine::C, {9728});
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));
    InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr = InferenceEngine::TBlob<uint8_t>::Ptr(weights);

    net_reader.SetWeights(weights_ptr);
    INetwork::Ptr network = Builder::Network(net_reader.getNetwork()).build();

    try {
        compareWithICNNNetwork(*network, net_reader.getNetwork());
    } catch (InferenceEngine::details::InferenceEngineException &ex) {
        FAIL() << ex.what();
    }
}

TEST_F(NetworkBuilderTest, convertFromICNNNetworkToICNNNetwork) {
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

    InferenceEngine::TBlob<uint8_t> *weights = new InferenceEngine::TBlob<uint8_t>(InferenceEngine::Precision::U8, InferenceEngine::C, {9728});
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));
    InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr = InferenceEngine::TBlob<uint8_t>::Ptr(weights);

    net_reader.SetWeights(weights_ptr);
    std::shared_ptr<ICNNNetwork> network = Builder::convertToICNNNetwork(Builder::Network(net_reader.getNetwork()).build());

    try {
        compareICNNNetworks(*network, net_reader.getNetwork());
    } catch (InferenceEngine::details::InferenceEngineException &ex) {
        FAIL() << ex.what();
    }
}

TEST_F(NetworkBuilderTest, connectTwoNetworks) {
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
            <convolution_data stride-x="2" stride-y="2" pad-x="3" pad-y="3" pad-r="3" pad-b="3" kernel-x="7" kernel-y="7" output="16" group="1"/>
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
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="2"/>
        <edge from-layer="2" from-port="3" to-layer="3" to-port="4"/>
        <edge from-layer="2" from-port="3" to-layer="4" to-port="6"/>
        <edge from-layer="3" from-port="5" to-layer="4" to-port="7"/>
    </edges>
</net>)V0G0N";

    InferenceEngine::CNNNetReader net_reader;
    ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

    InferenceEngine::TBlob<uint8_t> *weights = new InferenceEngine::TBlob<uint8_t>(InferenceEngine::Precision::U8, InferenceEngine::C, {9472});
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));
    InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr = InferenceEngine::TBlob<uint8_t>::Ptr(weights);

    net_reader.SetWeights(weights_ptr);
    Builder::Network originalNetwork(net_reader.getNetwork());
    Builder::Network addNetwork(net_reader.getNetwork());

    // Find output
    idx_t lastLayerId(0);
    for (const auto& layer : originalNetwork.getLayers()) {
        if (layer.getType() != "Output")
            continue;
        const auto connections = originalNetwork.getLayerConnections(layer.getId());
        ASSERT_EQ(1, connections.size());
        ASSERT_EQ(layer.getId(), connections[0].to().layerId());
        ASSERT_EQ(0, connections[0].from().portId());
        lastLayerId = connections[0].from().layerId();
        originalNetwork.disconnect(connections[0]);
        originalNetwork.removeLayer(layer.getId());
        break;
    }

    std::map<idx_t, idx_t> oldNewId;
    for (const auto& layer : addNetwork.getLayers()) {
        if (layer.getType() == "Input") {
            oldNewId[layer.getId()] = lastLayerId;
            continue;
        }
        oldNewId[layer.getId()] = originalNetwork.addLayer(layer);
        const auto connections = addNetwork.getLayerConnections(layer.getId());
        for (const auto& connection : connections) {
            if (oldNewId.find(connection.from().layerId()) == oldNewId.end() ||
                    oldNewId.find(connection.to().layerId()) == oldNewId.end())
                continue;
            originalNetwork.connect({oldNewId[connection.from().layerId()], connection.from().portId()},
                    {oldNewId[connection.to().layerId()], connection.to().portId()});
        }

        if (layer.getType() == "Convolution") {
            Builder::ConvolutionLayer(originalNetwork.getLayer(oldNewId[layer.getId()])).setWeights(generateBlob(Precision::FP32, {16, 32, 7, 7}, Layout::OIHW));
        }
    }
    ASSERT_NO_THROW(originalNetwork.build());
}

TEST_F(NetworkBuilderTest, createLayersWithTheSameNames) {
    InferenceEngine::Builder::Network netBuilder("");

    // Connect conolutional layer with it's inputs and outputs.
    InferenceEngine::Builder::InputLayer inpLayer("data");
    inpLayer.setPort(InferenceEngine::Port({1, 1, 10, 10}));
    auto inpLayerId = netBuilder.addLayer(inpLayer);

    // Create convolutional layer
    const size_t outCn = 1, inpCn = 1, kernelH = 3, kernelW = 3;
    InferenceEngine::Builder::ConvolutionLayer ieLayer("conv1");

    ieLayer.setKernel({outCn, inpCn, kernelH, kernelW});
    ieLayer.setStrides({1, 1, 1, 1});
    ieLayer.setDilation({1, 1, 1, 1});
    ieLayer.setPaddingsBegin({0, 0, 0, 0});
    ieLayer.setPaddingsEnd({0, 0, 0, 0});
    ieLayer.setGroup(1);
    ieLayer.setOutDepth(outCn);
    auto convLayerId = netBuilder.addLayer({inpLayerId}, ieLayer);

    // Connect convolution layer with it's output
    InferenceEngine::Builder::OutputLayer outLayer("conv1");
    auto convOutLayerId = netBuilder.addLayer({convLayerId}, outLayer);
    ASSERT_NE(netBuilder.getLayer(convLayerId).getName(), netBuilder.getLayer(convOutLayerId).getName());
    InferenceEngine::Builder::ReLULayer reLULayer("relu1");
    reLULayer.setNegativeSlope(0);
    auto reluLayerId = netBuilder.addLayer({convLayerId}, reLULayer);
    InferenceEngine::Builder::OutputLayer outReLULayer("relu1");
    auto reluOutLayerId = netBuilder.addLayer({reluLayerId}, outReLULayer);
    ASSERT_NE(netBuilder.getLayer(reluLayerId).getName(), netBuilder.getLayer(reluOutLayerId).getName());

    ASSERT_NO_THROW(netBuilder.build());
}

TEST_F(NetworkBuilderTest, RemoveLayerAndBuild) {
    auto builder = prepateAlexnetBuilder();
    builder.removeLayer(builder.getLayers()[2].getId());

    ASSERT_THROW(builder.build(), InferenceEngine::details::InferenceEngineException);
}

TEST_F(NetworkBuilderTest, DocumentationExample) {
    // Create graph with name
    InferenceEngine::Builder::Network graph("Example1");

    // Create network
    // In-place add input layer
    idx_t inputLayerId = graph.addLayer(Builder::InputLayer("in").setPort(Port({1, 3, 22, 22})));

    // In-place add ReLU layer builder with a negative slope 0.1 and connect it with 0 output port of the Input layer builder
    // In this example layerId is equal new Input layer builder ID, port index isn't set because 0 is a default value ({layerId} == {layerId, 0})
    idx_t relu1Id = graph.addLayer({{inputLayerId}}, Builder::ReLULayer("relu1").setNegativeSlope(0.1f));

    // In-place add ScaleShift layer builder
    InferenceEngine::Blob::Ptr blobWithScaleShiftBiases = make_shared_blob<float>(TensorDesc(Precision::FP32, {3}, Layout::C));
    blobWithScaleShiftBiases->allocate();
    auto *data = blobWithScaleShiftBiases->buffer().as<float *>();
    data[0] = 1;
    data[1] = 2;
    data[2] = 3;
    idx_t scaleShiftId = graph.addLayer(Builder::ScaleShiftLayer("scaleShift1").setBiases(blobWithScaleShiftBiases));

    // Connect ScaleShift layer with relu1
    graph.connect({relu1Id}, {scaleShiftId}); // Also port indexes could be defined (0 is default value) builder.connect({layerId, outPortIdx}, {scaleShiftId, inPortIdx});

    // Create ReLU layer with a negative slope 0.2 using generic layer builder and connect it with scaleShift
    idx_t relu2Id = graph.addLayer({{scaleShiftId}}, Builder::Layer("ReLU", "relu2").setParameters({{"negative_slope", 0.2f}}).setOutputPorts({Port()}).setInputPorts({Port()}));

    // All branches in the graph should be ended by Output layer. Let's create Output layer
    idx_t outId = graph.addLayer({{relu2Id, 0}}, Builder::OutputLayer("out"));

    // Build original network
    InferenceEngine::INetwork::Ptr finalNetwork = graph.build();
    std::shared_ptr<InferenceEngine::ICNNNetwork> cnnNetwork = InferenceEngine::Builder::convertToICNNNetwork(finalNetwork);

    // Modify network
    // Remove relu2 layer from the topology
    std::vector<InferenceEngine::Connection> connections = graph.getLayerConnections(relu2Id);
    for (const auto& connection : connections) {
        graph.disconnect(connection);
    }
    graph.removeLayer(relu2Id);

    // Connect scaleShift1 and out
    graph.connect({scaleShiftId}, {outId});
    // Build network without relu2
    InferenceEngine::INetwork::Ptr changedNetwork = graph.build();
}
