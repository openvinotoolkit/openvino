// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformer_single_layer_tests.hpp"

std::string ConvolutionAndQuantizeOnWeightsWithMultiOutputIntervalsTestModel::getModel(SingleLayerTransformationsTestParams& p) const {
    size_t type_size = sizeof(InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type);
    if (p._network_precision == "FP16")
        type_size = sizeof(InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP16>::value_type);

    CommonTestUtils::conv_common_params conv =
            { {1, 1}, {3, 3}, {0, 0}, {0, 0}, {1, 1}, "valid", 1, 64, false, false };
    std::vector<size_t> convOutShape(p.inputDimensions[0].size());
    getConvOutShape(p.inputDimensions[0], conv, convOutShape);

    std::vector<size_t> weightsConstInputDims = { 64lu, 32lu, 3lu, 3lu };
    std::vector<size_t> weightsConstOutputDims = { 64lu, 1lu, 1lu, 1lu };
    std::vector<size_t> biasesConvolutionConstDims = { conv.out_c };
    std::map<std::string, std::string> const_params = {};
    std::map<std::string, std::string> fake_quantize_params = {
        {"levels", "256"}
    };
    std::map<std::string, std::string> power_params = {
        {"power", "1"}, {"scale", "1"}, {"shift", "0"}
    };

    std::vector<std::pair<std::string, std::string>> edges = {
        {"0,0", "1,1"}, {"1,2", "6,7"}, // Power
        {"2,3", "6,8"}, {"3,4", "6,9"}, {"4,5", "6,10"}, {"5,6", "6,11"}, // Const layers
        {"7,13", "12,18"}, {"8,14", "12,19"}, {"9,15", "12,20"}, {"10,16", "12,21"}, {"11,17", "12,22"}, // Const layers
        {"6,12", "14,25"},  {"12,23", "14,26"}, // Fake quantize to Conv
        {"13,24", "14,27"} // biases to Conv
    };

    return CommonTestUtils::DefaultNetBuilder::buildNetworkWithOneInput(
            "QuantizationOnWeights", p.inputDimensions[0], p._network_precision)
        .addLayer("Power", p._network_precision, &power_params, {{p.inputDimensions[0]}, {p.inputDimensions[0]}})
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        .addLayer("FakeQuantize", p._network_precision, &fake_quantize_params, {{p.inputDimensions[0], {1}, {1}, {1}, {1}}, {{p.inputDimensions[0]}}})
        .addLayer("Const", p._network_precision, &const_params, {{}, {weightsConstInputDims}},
                std::accumulate(weightsConstInputDims.begin(), weightsConstInputDims.end(), 1lu, std::multiplies<size_t>()) * type_size, 0)
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        .addLayer("Const", p._network_precision, &const_params, {{}, {weightsConstOutputDims}},
                std::accumulate(weightsConstOutputDims.begin(), weightsConstOutputDims.end(), 1lu, std::multiplies<size_t>()) * type_size, 0)
        .addLayer("Const", p._network_precision, &const_params, {{}, {weightsConstOutputDims}},
                std::accumulate(weightsConstOutputDims.begin(), weightsConstOutputDims.end(), 1lu, std::multiplies<size_t>()) * type_size, 0)
        .addLayer("FakeQuantize", p._network_precision, &fake_quantize_params, {{weightsConstInputDims, {1}, {1}, weightsConstOutputDims, weightsConstOutputDims}, {{weightsConstInputDims}}})
        .addLayer("Const", p._network_precision, &const_params, {{}, {biasesConvolutionConstDims}}, type_size * conv.out_c, 0)
        .convolutionLayer(p._network_precision, {{p.inputDimensions[0], weightsConstInputDims, biasesConvolutionConstDims }, {convOutShape}}, conv)
        .finish(&edges);
}

std::string ConvolutionAndQuantizeOnWeightsWithMultiOutputIntervalsTestModel::getName() const {
    return "ConvolutionAndQuantizeOnWeightsWithMultiOutputIntervalsTestModel";
}

bool ConvolutionAndQuantizeOnWeightsWithMultiOutputIntervalsTestModel::transform(CNNNetwork& network, LayerTransformation::Params& params) const {
    LowPrecisionTransformer transformer = getLowPrecisionTransformer(params);
    transformer.transform(network);
    return true;
}

void ConvolutionAndQuantizeOnWeightsWithMultiOutputIntervalsTestModel::resetTransformation(CNNNetwork& network) const {
    // int values for range test
    fillData(getLayer(network, "Const2"), 0.0, "custom");
    fillData(getLayer(network, "Const3"), 255.0, "custom");
    fillData(getLayer(network, "Const4"), 0.0, "custom");
    fillData(getLayer(network, "Const5"), 255.0, "custom");

    fillData(getLayer(network, "Const7"), 4.0, "custom");

    fillData(getLayer(network, "Const8"), -128.0, "custom");
    fillData(getLayer(network, "Const9"), 127.0, "custom");
    fillData(getLayer(network, "Const10"), -128.0, "custom");
    fillData(getLayer(network, "Const11"), 127.0, "custom");

    fillData(getLayer(network, "Const13"), 5.0, "custom");
}
