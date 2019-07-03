// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <gtest/gtest.h>
#include <inference_engine/layer_transform.hpp>
#include <gna_plugin/quantization/model_quantizer.hpp>
#include <cpp/ie_cnn_net_reader.h>
#include "gna_plugin/quantization/layer_quantizer.hpp"
#include "gna_matcher.hpp"

using namespace InferenceEngine;
using namespace GNAPluginNS;
using namespace GNATestIRs;

class I8QuantisationTest : public GNATest {
 protected:
    LayersQuantizer<QuantI8> lc = LayersQuantizer<QuantI8> (1.0f);

    InferenceEngine::CNNLayerPtr  quantize (InferenceEngine::CNNLayerPtr lp) {
        auto newLayer = InferenceEngine::injectData<QuantizedLayerParams>(lp);
        transformLayer(newLayer, lc);
        return newLayer;
    };

    void SetUp() override  {
    }

};

// TODO: add test for FC weights after quantization
TEST_F(I8QuantisationTest, canQuantizeFCLayer){

    auto fc = std::make_shared<FullyConnectedLayer>(LayerParams{"name", "type", Precision::FP32});
    fc->_out_num = 9;
    auto weights = make_shared_blob<float>(Precision::FP32, {1, 1});
    fc->_weights = weights;
    fc->_biases = make_shared_blob<float>(Precision::FP32, {1, 1});
    fc->_weights->allocate();
    fc->_biases->allocate();
    std::shared_ptr<Data> outData = std::make_shared<Data>("data", SizeVector({1, 1}), Precision::FP32, Layout::NC);
    fc->outData.push_back(outData);
    fc->insData.push_back(outData);

    // actual quantisation algorithm is involved
    for (auto && w : *weights) {
        w =  MAX_OUT_MULTIPLIER * MAX_VAL_1B_WEIGHT;
    }

    fillWeights(fc->_biases);

    ASSERT_NO_THROW(quantize(fc));
}

TEST_F(I8QuantisationTest, canQuantizeActivation){

    auto sigmoid = std::make_shared<GenericLayer >(LayerParams{"name", "type", Precision::FP32});
    sigmoid->params["value"] = 2;
    sigmoid->type = "Activation";

    ASSERT_NO_THROW(quantize(sigmoid));
}

TEST_F(I8QuantisationTest, inputPrecisionIs16Bits){

    ModelQuantizer<QuantI8> q;

    CNNNetReader net_reader;
    ASSERT_NO_THROW(net_reader.ReadNetwork(Fc2DOutputModel().data(), Fc2DOutputModel().length()));

    auto weights = make_shared_blob<uint8_t >(Precision::U8, C, {440});
    weights->allocate();
    fillWeights(weights);
    net_reader.SetWeights(weights);
    auto newNet = q.quantize(net_reader.getNetwork(), 1000);
    InputsDataMap inputs;
    newNet->getInputsInfo(inputs);
    auto inputLayer = inputs.begin()->second->getInputData()->inputTo.begin()->second->insData.front().lock()->creatorLayer.lock();

    ASSERT_EQ(inputLayer->precision, Precision::I16);
}

TEST_F(I8QuantisationTest, failIfFCDimensionIs1){

    ModelQuantizer<QuantI8> q;

    CNNNetReader net_reader;
    ASSERT_NO_THROW(net_reader.ReadNetwork(FCOnlyModel().data(), FCOnlyModel().length()));

    auto weights = make_shared_blob<uint8_t >(Precision::U8, C, {440});
    weights->allocate();
    fillWeights(weights);
    net_reader.SetWeights(weights);

    ASSERT_ANY_THROW(q.quantize(net_reader.getNetwork(), 1000));
}

TEST_F(I8QuantisationTest, outputAffinePrecisionIs32Bits){

    ModelQuantizer<QuantI8> q;

    CNNNetReader net_reader;
    ASSERT_NO_THROW(net_reader.ReadNetwork(Fc2DOutputModel().data(), Fc2DOutputModel().length()));

    auto weights = make_shared_blob<uint8_t >(Precision::U8, C, {440});
    weights->allocate();
    fillWeights(weights);
    net_reader.SetWeights(weights);

    auto newNet = q.quantize(net_reader.getNetwork(), 1000);
    InputsDataMap inputs;
    newNet->getInputsInfo(inputs);
    auto affineDataPtr = inputs.begin()->second->getInputData()->inputTo.begin()->second->outData.front();

    ASSERT_EQ(affineDataPtr->precision, Precision::I32);
}
