// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformer_single_layer_tests.hpp"

std::string FakeQuantizeAndScaleShiftTestModel::getModel(SingleLayerTransformationsTestParams& p) const {
    size_t type_size = sizeof(InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type);
    if (p._network_precision == "FP16")
        type_size = sizeof(InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP16>::value_type);

    std::map<std::string, std::string> scale_shift_params = {};
    std::map<std::string, std::string> const_params = {};
    std::map<std::string, std::string> fake_quantize_params = {
        {"levels", "256"}
    };

    std::map<std::string, std::string> power_params = {
        {"power", "2"}, {"scale", "1"}, {"shift", "0"}
    };

    std::vector<std::pair<std::string, std::string>> edges = {
        {"0,0", "1,1"}, {"1,2", "6,7"}, // ScaleShift
        {"2,3", "6,8"}, {"3,4", "6,9"}, {"4,5", "6,10"}, {"5,6", "6,11"}, // Const layers
        {"6,12", "7,13"} // Fake quantize to Power
    };

    return CommonTestUtils::DefaultNetBuilder::buildNetworkWithOneInput(
            "DWConvFQ", p.inputDimensions[0], p._network_precision)
        .addLayer("ScaleShift", p._network_precision, &scale_shift_params, {{p.inputDimensions[0]}, {p.inputDimensions[0]}}, p.inputDimensions[0][1] * type_size, p.inputDimensions[0][1] * type_size)
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        .addLayer("FakeQuantize", p._network_precision, &fake_quantize_params, {{p.inputDimensions[0], {1}, {1}, {1}, {1}}, {{p.inputDimensions[0]}}})
        .addLayer("Power", p._network_precision, &power_params, {{p.inputDimensions[0]}, {p.inputDimensions[0]}})
        .finish(&edges);
}

std::string FakeQuantizeAndScaleShiftTestModel::getName() const {
    return "FakeQuantizeAndScaleShiftTestModel";
}

bool FakeQuantizeAndScaleShiftTestModel::transform(CNNNetwork& network, LayerTransformation::Params& params) const {
    LowPrecisionTransformer transformer = getLowPrecisionTransformer(params);
    transformer.transform(network);
    return true;
}

void FakeQuantizeAndScaleShiftTestModel::resetTransformation(CNNNetwork& network) const {
    fillData(getLayer(network, "Const2"), -128.f / 4.f, "custom");
    fillData(getLayer(network, "Const3"), 127.f / 4.f, "custom");
    fillData(getLayer(network, "Const4"), -128.f / 4.f, "custom");
    fillData(getLayer(network, "Const5"), 127.f / 4.f, "custom");

    fillDataWithInitValue(getLayer(network, "ScaleShift1"), "weights", 1.234f);
    fillDataWithInitValue(getLayer(network, "ScaleShift1"), "biases", 5.678f);
}
