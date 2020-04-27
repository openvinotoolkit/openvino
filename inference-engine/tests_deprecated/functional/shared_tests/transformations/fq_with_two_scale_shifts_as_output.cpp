// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformer_single_layer_tests.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

std::string FakeQuantizeWithTwoScaleShiftsAsOutput::getModel(SingleLayerTransformationsTestParams& p) const {
    size_t type_size = sizeof(PrecisionTrait<Precision::FP32>::value_type);
    if (p._network_precision == "FP16")
        type_size = sizeof(PrecisionTrait<Precision::FP16>::value_type);

    std::map<std::string, std::string> scale_shift_params = {};

    std::map<std::string, std::string> const_params = {};
    std::map<std::string, std::string> fake_quantize_params = {
        {"levels", "256"}
    };
    std::map<std::string, std::string> power_params = {
        {"power", "1"}, {"scale", "1"}, {"shift", "0"}
    };

    std::vector<std::pair<std::string, std::string>> edges = {
        {"0,0", "5,5"}, // input -> fq
        {"1,1", "5,6"}, {"2,2", "5,7"}, {"3,3", "5,8"}, {"4,4", "5,9"}, // Const layers
        {"5,10", "6,11"}, {"5,10", "7,13"}, // FQ -> SS
        {"6,12", "8,15"}, {"7,14", "9,17"} // SS -> Power
    };

    return CommonTestUtils::DefaultNetBuilder::buildNetworkWithOneInput(
            "FakeQuantizeWithTwoScaleShiftsAsOutput", p.inputDimensions[0], p._network_precision)
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, "inputLow")
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, "inputHigh")
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, "outputLow")
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, "outputHigh")
        .addLayer("FakeQuantize", p._network_precision, &fake_quantize_params, {{p.inputDimensions[0], {1}, {1}, {1}, {1}}, {{p.inputDimensions[0]}}})
        .addLayer("ScaleShift", p._network_precision, &scale_shift_params, {{p.inputDimensions[0]}, {p.inputDimensions[0]}}, p.inputDimensions[0][1] * type_size, p.inputDimensions[0][1] * type_size)
        .addLayer("ScaleShift", p._network_precision, &scale_shift_params, {{p.inputDimensions[0]}, {p.inputDimensions[0]}}, p.inputDimensions[0][1] * type_size, p.inputDimensions[0][1] * type_size)
        .addLayer("Power", p._network_precision, &power_params, {{p.inputDimensions[0]}, {p.inputDimensions[0]}})
        .addLayer("Power", p._network_precision, &power_params, {{p.inputDimensions[0]}, {p.inputDimensions[0]}})
        .finish(&edges);
}

std::string FakeQuantizeWithTwoScaleShiftsAsOutput::getName() const {
    return "FakeQuantizeWithTwoScaleShiftsAsOutput";
}

bool FakeQuantizeWithTwoScaleShiftsAsOutput::transform(CNNNetwork& network, LayerTransformation::Params& params) const {
    LowPrecisionTransformer transformer(LowPrecisionTransformer::getAllTransformations(params));
    transformer.transform(network);

    return true;
}

void FakeQuantizeWithTwoScaleShiftsAsOutput::resetTransformation(CNNNetwork& network) const {
    fillData(getLayer(network, "inputLow"), 0.f, "custom");
    fillData(getLayer(network, "inputHigh"), 5.f, "custom");
    fillData(getLayer(network, "outputLow"), 0.f, "custom");
    fillData(getLayer(network, "outputHigh"), 5.f, "custom");

    fillData(getLayer(network, "ScaleShift6"), 3.f, "weights");
    fillData(getLayer(network, "ScaleShift6"), 3.f, "biases");
    fillData(getLayer(network, "ScaleShift7"), 1.5f, "weights");
    fillData(getLayer(network, "ScaleShift7"), 1.5f, "biases");
}
