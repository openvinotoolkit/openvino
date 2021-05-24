// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <gtest/gtest.h>
#include <legacy/layer_transform.hpp>
#include <frontend/model_quantizer.hpp>
#include "frontend/layer_quantizer.hpp"
#include "gna_matcher.hpp"
#include <ie_core.hpp>

using namespace InferenceEngine;
using namespace GNAPluginNS;
using namespace GNATestIRs;

class I8QuantisationTest : public GNATest<> {
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
    auto weights = make_shared_blob<float>({ Precision::FP32, {1, 1}, Layout::NC });
    fc->_weights = weights;
    fc->_biases = make_shared_blob<float>({ Precision::FP32, {1, 1}, Layout::NC });
    fc->_weights->allocate();
    fc->_biases->allocate();
    std::shared_ptr<Data> outData = std::make_shared<Data>("data", TensorDesc(Precision::FP32, SizeVector({ 1, 1 }), Layout::NC));
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

    auto weights = make_shared_blob<uint8_t >({ Precision::U8, {440}, C });
    weights->allocate();
    fillWeights(weights);

    Core ie;
    auto network = ie.ReadNetwork(Fc2DOutputModel(), weights);

    auto newNet = q.quantize(network, 1000);
    InputsDataMap inputs = newNet.getInputsInfo();
    auto inputLayer = getCreatorLayer(getInputTo(inputs.begin()->second->getInputData()).begin()->second->insData.front().lock()).lock();

    ASSERT_EQ(inputLayer->precision, Precision::I16);
}

TEST_F(I8QuantisationTest, FCDimensionIs1){
    ModelQuantizer<QuantI8> q;

    auto weights = make_shared_blob<uint8_t >({ Precision::U8, {440}, C });
    weights->allocate();
    fillWeights(weights);
    
    Core ie;
    auto network = ie.ReadNetwork(FCOnlyModel(), weights);

    ASSERT_NO_THROW(q.quantize(network, 1000));
}

TEST_F(I8QuantisationTest, outputAffinePrecisionIs32Bits){
    ModelQuantizer<QuantI8> q;

    auto weights = make_shared_blob<uint8_t >({ Precision::U8, {440}, C });
    weights->allocate();
    fillWeights(weights);
    
    Core ie;
    auto network = ie.ReadNetwork(Fc2DOutputModel(), weights);

    auto newNet = q.quantize(network, 1000);
    InputsDataMap inputs = newNet.getInputsInfo();
    auto affineDataPtr = getInputTo(inputs.begin()->second->getInputData()).begin()->second->outData.front();

    ASSERT_EQ(affineDataPtr->getTensorDesc().getPrecision(), Precision::I32);
}

TEST_F(I8QuantisationTest, fp16tofp32_on_fullyConnected_model) {
    ModelQuantizer<QuantI8> q;

    auto weights = make_shared_blob<uint8_t>({ Precision::U8, {220}, Layout::C });
    weights->allocate();
    fillWeights(weights);
    
    Core ie;
    auto network = ie.ReadNetwork(FCOnlyModelFP16(), weights);

    q.quantize(network, 1000);
}

TEST_F(I8QuantisationTest, LSTMCell_quantize) {
    ModelQuantizer<QuantI8> q;

    auto weights = make_shared_blob<uint8_t>({ Precision::U8, {33664}, C });
    weights->allocate();
    fillWeights(weights);
    
    Core ie;
    auto network = ie.ReadNetwork(LSTMCellOnlyModel(), weights);

    ASSERT_NO_THROW(q.quantize(network, 1000));
}

TEST_F(I8QuantisationTest, LSTMCell_unaligned_quantize) {
    ModelQuantizer<QuantI8> q;

    auto weights = make_shared_blob<uint8_t>({ Precision::U8, {3480}, C });
    weights->allocate();
    fillWeights(weights);
    
    Core ie;
    auto network = ie.ReadNetwork(LSTMCellOnlyModelUnaligned(), weights);

    ASSERT_NO_THROW(q.quantize(network, 1000));
}

TEST_F(I8QuantisationTest, TI_quantize) {
    ModelQuantizer<QuantI8> q;

    auto weights = make_shared_blob<uint8_t>({ Precision::U8, {249748}, C });
    weights->allocate();
    fillWeights(weights);
        
    Core ie;
    auto network = ie.ReadNetwork(TIModelWithLSTMCell2(), weights);

    ASSERT_NO_THROW(q.quantize(network, 1000));
}
