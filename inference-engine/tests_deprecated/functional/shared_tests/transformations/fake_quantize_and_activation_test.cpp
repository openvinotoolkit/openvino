// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformer_single_layer_tests.hpp"
#include <vector>

FakeQuantizeAndActivationTestModel::FakeQuantizeAndActivationTestModel(const std::vector<std::pair<float, float>>& intervals) :
    intervals(intervals) {}

void FakeQuantizeAndActivationTestModel::initInput(Blob::Ptr input) const {
    const Precision& precision = input->getTensorDesc().getPrecision();
    const size_t dataSize = input->size();

    std::vector<float> data(input->size(), 4.0);
    const float step = (intervals[0].second - intervals[0].first) / dataSize;
    float value = intervals[0].first;
    for (size_t i = 0ul; i < dataSize; ++i) {
        if (precision == Precision::FP32) {
            float* buffer = input->buffer().as<float*>();
            buffer[i] = InferenceEngine::PrecisionUtils::f32tof16(value);
        } else if (precision == Precision::FP16) {
            short* buffer = input->buffer().as<short*>();
            buffer[i] = InferenceEngine::PrecisionUtils::f32tof16(value);
        }

        value += step;
        if (value > intervals[0].second) {
            value = intervals[0].first;
        }
    }
}

float FakeQuantizeAndActivationTestModel::getZeroThreshold() const {
    const float interval = intervals[0].second - intervals[0].first;
    return interval / (256.f * 1.e3f);
}

std::string FakeQuantizeAndActivationTestModel::getModel(SingleLayerTransformationsTestParams& p) const {
    size_t type_size = sizeof(InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type);
    if (p._network_precision == "FP16")
        type_size = sizeof(InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP16>::value_type);

    std::map<std::string, std::string> const_params = {};
    std::map<std::string, std::string> fake_quantize_params = {{"levels", "256"}};
    std::map<std::string, std::string> power_params = {{"power", "1"}, {"scale", "1"}, {"shift", "0"}};

    std::vector<std::pair<std::string, std::string>> edges = {
        {"0,0", "1,1"}, {"1,2", "6,7"}, // Power
        {"2,3", "6,8"}, {"3,4", "6,9"}, {"4,5", "6,10"}, {"5,6", "6,11"}, // Const layers
        {"6,12", "7,13"}, // Fake quantize to ReLU
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
        .addLayer("ReLU", p._network_precision, {}, { {p.inputDimensions[0]}, {p.inputDimensions[0]} })
        // 8
        .addLayer("Power", p._network_precision, &power_params, {{p.inputDimensions[0]}, {p.inputDimensions[0]}})
        .finish(&edges);
}

std::string FakeQuantizeAndActivationTestModel::getName() const {
    return
        "FakeQuantizeAndActivationTestModel_" +
        std::to_string(intervals.size()) + "_" +
        std::to_string(intervals[0].first) + "_" + std::to_string(intervals[0].second);
}

bool FakeQuantizeAndActivationTestModel::transform(CNNNetwork& network, LayerTransformation::Params& params) const {
    LowPrecisionTransformer transformer = getLowPrecisionTransformer(params);
    transformer.transform(network);
    return true;
}

void FakeQuantizeAndActivationTestModel::resetTransformation(CNNNetwork& network) const {
    std::vector<float> low(intervals.size());
    std::vector<float> high(intervals.size());
    for (size_t i = 0ul; i < intervals.size(); ++i) {
        const std::pair<float, float> interval = intervals[i];
        low[i] = interval.first;
        high[i] = interval.second;
    }

    fillData(getLayer(network, "Const2"), low, "custom");
    fillData(getLayer(network, "Const3"), high, "custom");
    fillData(getLayer(network, "Const4"), low, "custom");
    fillData(getLayer(network, "Const5"), high, "custom");
}
