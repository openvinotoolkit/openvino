// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformer_single_layer_tests.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

std::string EltwiseBroadcastTestModel::getModel(SingleLayerTransformationsTestParams& p) const {
    size_t type_size = sizeof(PrecisionTrait<Precision::FP32>::value_type);
    if (p._network_precision == "FP16")
        type_size = sizeof(PrecisionTrait<Precision::FP16>::value_type);

    std::map<std::string, std::string> const_params = {};
    std::map<std::string, std::string> fake_quantize_params = {
        {"levels", "256"}
    };
    std::map<std::string, std::string> eltwise_params = {
        {"operation", "sum"}
    };
    std::map<std::string, std::string> power_params = {
        {"power", "1"}, {"scale", "1"}, {"shift", "0"}
    };

    std::vector<std::pair<std::string, std::string>> edges = {
        {"0,0", "6,6"}, {"1,1", "11,16"}, // Inputs
        {"2,2", "6,7"}, {"3,3", "6,8"}, {"4,4", "6,9"}, {"5,5", "6,10"}, // Const layers
        {"7,12", "11,17"}, {"8,13", "11,18"}, {"9,14", "11,19"}, {"10,15", "11,20"}, // Const layers
        {"6,11", "12,22"}, {"11,21", "12,23"} // Fake quantize to Convolution
    };

    return CommonTestUtils::DefaultNetBuilder::buildNetworkWithOneInput(
            "Eltwise", p.inputDimensions[0], p._network_precision)
        .addLayer("Const", p._network_precision, &const_params, {{}, {p.inputDimensions[1]}}, type_size, 0)
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        .addLayer("FakeQuantize", p._network_precision, &fake_quantize_params, {{p.inputDimensions[0], {1}, {1}, {1}, {1}}, {{p.inputDimensions[0]}}})
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        .addLayer("FakeQuantize", p._network_precision, &fake_quantize_params, {{p.inputDimensions[1], {1}, {1}, {1}, {1}}, {{p.inputDimensions[1]}}})
        .addLayer("Eltwise", p._network_precision, &eltwise_params, {{p.inputDimensions[0], p.inputDimensions[1]}, {{p.outputDimensions[0]}}}, 0, 0)
        .finish(&edges);
}

std::string EltwiseBroadcastTestModel::getName() const {
    return "EltwiseBroadcastTestModel";
}

bool EltwiseBroadcastTestModel::transform(CNNNetwork& network, LayerTransformation::Params& params) const {
    LowPrecisionTransformer transformer(LowPrecisionTransformer::getAllTransformations(
        LayerTransformation::Params(params)));
    transformer.transform(network);
    return true;
}

void EltwiseBroadcastTestModel::resetTransformation(CNNNetwork& network) const {
    fillData(getLayer(network, "Const2"), 255.f / 10.0, "custom");
    fillData(getLayer(network, "Const3"), 255.f / 4.0, "custom");
    fillData(getLayer(network, "Const4"), 255.f / 10.0, "custom");
    fillData(getLayer(network, "Const5"), 255.f / 4.0, "custom");

    fillData(getLayer(network, "Const7"), 255.f / 10.0, "custom");
    fillData(getLayer(network, "Const8"), 255.f / 2.0, "custom");
    fillData(getLayer(network, "Const9"), 255.f / 10.0, "custom");
    fillData(getLayer(network, "Const10"), 255.f / 2.0, "custom");
}
