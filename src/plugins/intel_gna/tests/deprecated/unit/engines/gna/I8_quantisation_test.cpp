// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ie_core.hpp>
#include <legacy/layer_transform.hpp>
#include <vector>

#include "backend/gna_limitations.hpp"
#include "frontend/layer_quantizer.hpp"
#include "frontend/model_quantizer.hpp"
#include "gna_matcher.hpp"

using namespace InferenceEngine;
using namespace ov::intel_gna::limitations;
using namespace ov::intel_gna::frontend;
using namespace GNATestIRs;

class I8QuantisationTest : public GNATest<> {
protected:
    InferenceEngine::CNNLayerPtr quantize(InferenceEngine::CNNLayerPtr lp) {
        auto newLayer = InferenceEngine::injectData<QuantizedLayerParams>(lp);
        Config gna_config;
        gna_config.gnaPrecision = InferenceEngine::Precision::I16;
        gna_config.gnaFlags.input_low_precision = false;
        LayerQuantizer lq(gna_config);
        lq.quantize(*newLayer);
        return newLayer;
    };

    InferenceEngine::CNNNetwork quantize_single_input_model(const InferenceEngine::CNNNetwork& model,
                                                            float scale_factor) const {
        auto scale_factors = std::vector<float>({scale_factor});

        GnaInputs inputs;
        InferenceEngine::InputsDataMap inputs_map = model.getInputsInfo();

        auto input_layer = getCreatorLayer(inputs_map.begin()->second->getInputData()).lock();
        inputs[input_layer->name].scale_factor = scale_factor;

        Config gna_config;
        gna_config.gnaPrecision = InferenceEngine::Precision::I16;
        gna_config.gnaFlags.input_low_precision = false;

        auto transformer = ov::intel_gna::TransformationsPipeline(gna_config);

        return ModelQuantizer(transformer).quantize(model, inputs);
    }

    void SetUp() override {
        Limitations::init(target::DeviceVersion::Default);
    }
};

// TODO: add test for FC weights after quantization
TEST_F(I8QuantisationTest, canQuantizeFCLayer) {
    auto fc = std::make_shared<FullyConnectedLayer>(LayerParams{"name", "type", Precision::FP32});
    fc->_out_num = 9;
    auto weights = make_shared_blob<float>({Precision::FP32, {1, 1}, Layout::NC});
    fc->_weights = weights;
    fc->_biases = make_shared_blob<float>({Precision::FP32, {1, 1}, Layout::NC});
    fc->_weights->allocate();
    fc->_biases->allocate();
    std::shared_ptr<Data> outData =
        std::make_shared<Data>("data", TensorDesc(Precision::FP32, SizeVector({1, 1}), Layout::NC));
    fc->outData.push_back(outData);
    fc->insData.push_back(outData);

    // actual quantisation algorithm is involved
    for (auto&& w : *weights) {
        w = MAX_OUT_MULTIPLIER * MAX_VAL_1B_WEIGHT;
    }

    fillWeights(fc->_biases);

    ASSERT_NO_THROW(quantize(fc));
}

TEST_F(I8QuantisationTest, canQuantizeActivation) {
    auto sigmoid = std::make_shared<GenericLayer>(LayerParams{"name", "type", Precision::FP32});
    sigmoid->params["value"] = 2;
    sigmoid->type = "Activation";

    ASSERT_NO_THROW(quantize(sigmoid));
}

TEST_F(I8QuantisationTest, inputPrecisionIs16Bits) {
    auto weights = make_shared_blob<uint8_t>({Precision::U8, {440}, C});
    weights->allocate();
    fillWeights(weights);

    Core ie;
    auto network = ie.ReadNetwork(Fc2DOutputModel(), weights);

    auto newNet = quantize_single_input_model(network, 1000);
    InputsDataMap inputs = newNet.getInputsInfo();
    auto inputLayer =
        getCreatorLayer(getInputTo(inputs.begin()->second->getInputData()).begin()->second->insData.front().lock())
            .lock();

    ASSERT_EQ(inputLayer->precision, Precision::I16);
}

TEST_F(I8QuantisationTest, FCDimensionIs1) {
    auto weights = make_shared_blob<uint8_t>({Precision::U8, {440}, C});
    weights->allocate();
    fillWeights(weights);

    Core ie;
    auto network = ie.ReadNetwork(FCOnlyModel(), weights);

    ASSERT_NO_THROW(quantize_single_input_model(network, 1000));
}

TEST_F(I8QuantisationTest, outputAffinePrecisionIs32Bits) {
    auto weights = make_shared_blob<uint8_t>({Precision::U8, {440}, C});
    weights->allocate();
    fillWeights(weights);

    Core ie;
    auto network = ie.ReadNetwork(Fc2DOutputModel(), weights);

    auto newNet = quantize_single_input_model(network, 1000);
    InputsDataMap inputs = newNet.getInputsInfo();
    auto affineDataPtr = getInputTo(inputs.begin()->second->getInputData()).begin()->second->outData.front();

    ASSERT_EQ(affineDataPtr->getTensorDesc().getPrecision(), Precision::I32);
}

TEST_F(I8QuantisationTest, fp16tofp32_on_fullyConnected_model) {
    auto weights = make_shared_blob<uint8_t>({Precision::U8, {220}, Layout::C});
    weights->allocate();
    fillWeights(weights);

    Core ie;
    auto network = ie.ReadNetwork(FCOnlyModelFP16(), weights);

    quantize_single_input_model(network, 1000);
}

TEST_F(I8QuantisationTest, LSTMCell_quantize) {
    auto weights = make_shared_blob<uint8_t>({Precision::U8, {33664}, C});
    weights->allocate();
    fillWeights(weights);

    Core ie;
    auto network = ie.ReadNetwork(LSTMCellOnlyModel(), weights);

    ASSERT_NO_THROW(quantize_single_input_model(network, 1000));
}

TEST_F(I8QuantisationTest, LSTMCell_unaligned_quantize) {
    auto weights = make_shared_blob<uint8_t>({Precision::U8, {3480}, C});
    weights->allocate();
    fillWeights(weights);

    Core ie;
    auto network = ie.ReadNetwork(LSTMCellOnlyModelUnaligned(), weights);

    ASSERT_NO_THROW(quantize_single_input_model(network, 1000));
}

TEST_F(I8QuantisationTest, TI_quantize) {
    auto weights = make_shared_blob<uint8_t>({Precision::U8, {249748}, C});
    weights->allocate();
    fillWeights(weights);

    Core ie;
    auto network = ie.ReadNetwork(TIModelWithLSTMCell2(), weights);

    ASSERT_NO_THROW(quantize_single_input_model(network, 1000));
}
