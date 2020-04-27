// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformer_single_layer_tests.hpp"

void ScaleShiftAndFakeQuantizeTestModel::initInput(Blob::Ptr input) const {
    const Precision& precision = input->getTensorDesc().getPrecision();
    const size_t dataSize = input->size();

    std::vector<float> data(input->size(), 4.0);
    float value = -64.0;
    for (size_t i = 0ul; i < std::min(static_cast<size_t>(256), dataSize); ++i) {
        if (precision == Precision::FP32) {
            float* buffer = input->buffer().as<float*>();
            buffer[i] = InferenceEngine::PrecisionUtils::f32tof16(value);
        } else if (precision == Precision::FP16) {
            short* buffer = input->buffer().as<short*>();
            buffer[i] = InferenceEngine::PrecisionUtils::f32tof16(value);
        }
        value += 1.0;
    }
}

std::string ScaleShiftAndFakeQuantizeTestModel::getModel(SingleLayerTransformationsTestParams& p) const {
    size_t type_size = sizeof(InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type);
    if (p._network_precision == "FP16")
        type_size = sizeof(InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP16>::value_type);

    std::map<std::string, std::string> const_params = {};
    std::map<std::string, std::string> fake_quantize_params = {{"levels", "256"}};
    std::map<std::string, std::string> power_params = {{"power", "1"}, {"scale", "1"}, {"shift", "0"}};

    std::vector<std::pair<std::string, std::string>> edges = {
        {"0,0", "1,1"}, {"1,2", "6,7"}, // Power
        {"2,3", "6,8"}, {"3,4", "6,9"}, {"4,5", "6,10"}, {"5,6", "6,11"}, // Const layers
        {"6,12", "7,13"}, // Fake quantize to ScaleShift
        {"7,14", "8,15"}
    };

    return CommonTestUtils::DefaultNetBuilder::buildNetworkWithOneInput("FakeQuantizeAndActivationTestModel", p.inputDimensions[0], p._network_precision)
        // 1
        .addLayer("Power", p._network_precision, &power_params, {{p.inputDimensions[0]}, {p.inputDimensions[0]}})
        // 2
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        // 3
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        // 4
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        // 5
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        // 6
        .addLayer("FakeQuantize", p._network_precision, &fake_quantize_params, {{p.inputDimensions[0], {1}, {1}, {1}, {1}}, {{p.inputDimensions[0]}}})
        // 7
        .addLayer("ScaleShift", p._network_precision, {}, { {p.inputDimensions[0]}, {p.inputDimensions[0]} }, 3 * type_size, 3 * type_size)
        // 8
        .addLayer("Power", p._network_precision, &power_params, {{p.inputDimensions[0]}, {p.inputDimensions[0]}})
        .finish(&edges);
}

std::string ScaleShiftAndFakeQuantizeTestModel::getName() const {
    return "ScaleShiftAndFakeQuantizeTestModel";
}

bool ScaleShiftAndFakeQuantizeTestModel::transform(CNNNetwork& network, LayerTransformation::Params& params) const {
    LowPrecisionTransformer transformer(LowPrecisionTransformer::getAllTransformations(params));
    transformer.transform(network);
    return true;
}

void ScaleShiftAndFakeQuantizeTestModel::resetTransformation(CNNNetwork& network) const {
    fillData(getLayer(network, "Const2"), -128.f / 4.f, "custom");
    fillData(getLayer(network, "Const3"), 127.f / 4.f, "custom");
    fillData(getLayer(network, "Const4"), -128.f / 4.f, "custom");
    fillData(getLayer(network, "Const5"), 127.f / 4.f, "custom");

    fillData(getLayer(network, "ScaleShift7"), 1.0, "weights");
    fillData(getLayer(network, "ScaleShift7"), 0.0, "biases");
}
