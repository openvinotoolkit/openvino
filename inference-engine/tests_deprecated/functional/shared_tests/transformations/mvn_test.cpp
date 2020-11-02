// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformer_single_layer_tests.hpp"

void MvnTestModel::initInput(Blob::Ptr input) const {
    const size_t dataSize = input->size();
    std::shared_ptr<float> floatPtr(new float[dataSize], std::default_delete<float[]>());

    float value = 0.f;
    for (size_t i = 0ul; i < dataSize; ++i) {
        floatPtr.get()[i] = value;
        if (value > 255.0) {
            value = 0.f;
        }
        value += 1.f;
    }

    CNNNetworkHelper::fillBlobByFP32(input, floatPtr.get());
}

MvnTestModel::MvnTestModel(const size_t acrossChannels, const size_t normalizeVariance) :
    acrossChannels(acrossChannels),
    normalizeVariance(normalizeVariance) {}

std::string MvnTestModel::getModel(SingleLayerTransformationsTestParams& p) const {
    size_t type_size = sizeof(InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type);
    if (p._network_precision == "FP16") {
        type_size = sizeof(InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP16>::value_type);
    }

    std::map<std::string, std::string> power_params = {{"power", "1"}, {"scale", "1"}, {"shift", "0"}};
    std::map<std::string, std::string> const_params = {};
    std::map<std::string, std::string> fake_quantize_params = {{"levels", "256"}};
    std::map<std::string, std::string> mvn_params = {
        {"eps", "0.001"},
        {"across_channels", std::to_string(acrossChannels)},
        {"normalize_variance", std::to_string(acrossChannels)}
    };

    std::vector<std::pair<std::string, std::string>> edges = {
        {"0,0", "1,1"}, {"1,2", "6,7"}, // power
        {"2,3", "6,8"}, {"3,4", "6,9"}, {"4,5", "6,10"}, {"5,6", "6,11"}, // const
        {"6,12", "7,13"}, {"7,14", "8,15"} // pool, power
    };

    const std::vector<size_t> dimensions = p.outputDimensions[0];

    return CommonTestUtils::DefaultNetBuilder::buildNetworkWithOneInput("MvnTestModel", dimensions, p._network_precision)
        .addLayer("Power", p._network_precision, &power_params, {{dimensions}, {dimensions}})
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        .addLayer("FakeQuantize", p._network_precision, &fake_quantize_params, {{dimensions, {1}, {1}, {1}, {1}}, {{dimensions}}})
        .addLayer("MVN", p._network_precision, &mvn_params, { {dimensions}, {dimensions} })
        .addLayer("Power", p._network_precision, &power_params, {{dimensions}, {dimensions}})
        .finish(&edges);
}

bool MvnTestModel::transform(CNNNetwork& network, LayerTransformation::Params& params) const {
    LowPrecisionTransformer transformer(LowPrecisionTransformer::getAllTransformations(params));
    transformer.transform(network);
    return true;
}

std::string MvnTestModel::getName() const {
    return
        "MvnTestModel" +
        (acrossChannels == 1ul ? std::string("_AcrossChannels") : "") +
        (normalizeVariance == 1ul ? std::string("_NormalizeVariance") : "");
}

void MvnTestModel::resetTransformation(CNNNetwork& network) const {
    fillData(getLayer(network, "Const2"), 0.f, "custom");
    fillData(getLayer(network, "Const3"), 255.f / 2.f, "custom");
    fillData(getLayer(network, "Const4"), 0.f, "custom");
    fillData(getLayer(network, "Const5"), 255.f / 2.f, "custom");
}
