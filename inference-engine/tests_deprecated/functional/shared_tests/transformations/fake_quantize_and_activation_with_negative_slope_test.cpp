// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformer_single_layer_tests.hpp"

void FakeQuantizeAndActivationWithNegativeSlopeTestModel::initInput(Blob::Ptr input) const {
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

std::string FakeQuantizeAndActivationWithNegativeSlopeTestModel::getModel(SingleLayerTransformationsTestParams& p) const {
    size_t type_size = sizeof(InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type);
    if (p._network_precision == "FP16")
        type_size = sizeof(InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP16>::value_type);

    std::map<std::string, std::string> const_params = {};
    std::map<std::string, std::string> fake_quantize_params = { {"levels", "256"} };
    std::map<std::string, std::string> power_params = {{"power", "1"}, {"scale", "1"}, {"shift", "0"}};
    std::map<std::string, std::string> reluParams = { {"negative_slope", "-1.0"} };

    std::vector<std::pair<std::string, std::string>> edges = {
        {"0,0", "1,1"}, // Input -> Power
        {"1,2", "6,7"}, // Power -> FakeQuantize
        {"2,3", "6,8"}, {"3,4", "6,9"}, {"4,5", "6,10"}, {"5,6", "6,11"}, // Const layers
        {"6,12", "7,13"}, // FakeQuantize -> ScaleShift
        {"7,14", "8,15"}, // ScaleShift -> ReLU
        {"8,16", "9,17"}  // ReLU -> Power
    };

    return CommonTestUtils::DefaultNetBuilder::buildNetworkWithOneInput("FakeQuantizeAndActivationWithNegativeSlopeTestModel", p.inputDimensions[0], p._network_precision)
        // 1
        .addLayer("Power", p._network_precision, &power_params, { {p.inputDimensions[0]}, {p.inputDimensions[0]} })
        // 2
        .addLayer("Const", p._network_precision, &const_params, { {}, {{1}} }, type_size, 0)
        // 3
        .addLayer("Const", p._network_precision, &const_params, { {}, {{1}} }, type_size, 0)
        // 4
        .addLayer("Const", p._network_precision, &const_params, { {}, {{1}} }, type_size, 0)
        // 5
        .addLayer("Const", p._network_precision, &const_params, { {}, {{1}} }, type_size, 0)
        // 6
        .addLayer("FakeQuantize", p._network_precision, &fake_quantize_params, { {p.inputDimensions[0], {1}, {1}, {1}, {1}}, {{p.inputDimensions[0]}} })
        // 7
        .addLayer("ScaleShift", p._network_precision, {}, {{p.inputDimensions[0]}, {p.inputDimensions[0]}}, p.inputDimensions[0][1] * type_size, p.inputDimensions[0][1] * type_size)
        // 8
        .addLayer("ReLU", p._network_precision, &reluParams, { {p.inputDimensions[0]}, {p.inputDimensions[0]} })
        // 9
        .addLayer("Power", p._network_precision, &power_params, {{p.inputDimensions[0]}, {p.inputDimensions[0]}})
        .finish(&edges);
}

std::string FakeQuantizeAndActivationWithNegativeSlopeTestModel::getName() const {
    return "FakeQuantizeAndActivationWithNegativeSlopeTestModel";
}

bool FakeQuantizeAndActivationWithNegativeSlopeTestModel::transform(CNNNetwork& network, LayerTransformation::Params& params) const {
    LowPrecisionTransformer transformer = getLowPrecisionTransformer(params);
    transformer.transform(network);

    CNNLayerPtr relu = getLayer(network, "ReLU8");
    if (relu == nullptr) {
        THROW_IE_EXCEPTION << "layer was not found " << relu->name;
    }

    const std::vector<CNNLayerPtr> parents = CNNNetworkHelper::getParents(*relu);
    if (parents.size() != 1) {
        THROW_IE_EXCEPTION << "unexpected parent layers size " << parents.size();
    }

    if (parents[0]->name != "FakeQuantize6") {
        // FQ -> dequantization -> ReLU
        if (parents[0]->name != "ScaleShift7") {
            THROW_IE_EXCEPTION << "unexpected parent layer " << parents[0]->name;
        }

        if (parents[0]->type == "ScaleShift") {
            CNNLayerPtr dequantizationScaleShift = parents[0];
            const Blob::Ptr weightsBlob = CNNNetworkHelper::getBlob(dequantizationScaleShift, "weights");
            auto weights = CNNNetworkHelper::getFloatData(weightsBlob);
            const std::vector<float> scales = std::vector<float>(weights.get(), weights.get() + weightsBlob->size());

            const Blob::Ptr biasesBlob = CNNNetworkHelper::getBlob(dequantizationScaleShift, "biases");
            auto biases = CNNNetworkHelper::getFloatData(biasesBlob);
            const std::vector<float> shifts = std::vector<float>(biases.get(), biases.get() + biasesBlob->size());

            if ((std::all_of(shifts.begin(), shifts.end(), [](float value) { return value == 0.0; })) &&
                (std::all_of(scales.begin(), scales.end(), [](float value) { return value >= 0.0; }))) {
                THROW_IE_EXCEPTION << "dequantization " << parents[0]->type << " " << parents[0]->name << " was not moved via " << " " << relu->type << " " << relu->name;
            }
        } else if (parents[0]->type == "Convolution") {
            const CNNLayerPtr convolution = parents[0];
            const std::vector<CNNLayerPtr> parents =  CNNNetworkHelper::getParents(*convolution);

            const Blob::Ptr weightsBlob = CNNNetworkHelper::getBlob(parents[1], "custom");
            if (weightsBlob == nullptr) {
                THROW_IE_EXCEPTION << "weights are absent";
            }
            const std::shared_ptr<float> weights = CNNNetworkHelper::getFloatData(weightsBlob);
            if (weights == nullptr) {
                THROW_IE_EXCEPTION << "weights are not received";
            }
            const std::vector<float> scales = std::vector<float>(weights.get(), weights.get() + weightsBlob->size());


            if (std::any_of(scales.begin(), scales.end(), [](float value) { return value < 0.0; })) {
                THROW_IE_EXCEPTION << "dequantization scales are not correct";
            }

            const Blob::Ptr biasesBlob = CNNNetworkHelper::getBlob(parents[2], "custom");
            if (biasesBlob == nullptr) {
                THROW_IE_EXCEPTION << "biases are absent";
            }
            const std::shared_ptr<float> biases = CNNNetworkHelper::getFloatData(biasesBlob);
            if (biases == nullptr) {
                THROW_IE_EXCEPTION << "biases are not received";
            }
        } else {
            THROW_IE_EXCEPTION << "unexpected parent layer type " << parents[0]->type;
        }
    } else {
        // FQ -> ReLU -> dequantization or FQ -> ReLU -> Power
        const std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildren(*relu);
        if (children.size() != 1lu) {
            THROW_IE_EXCEPTION << "unexpected children layers size " << children.size();
        }
        if (children[0]->name != "Power9" && children[0]->name != "ReLU8_ScaleShift_Power9") {
            THROW_IE_EXCEPTION << "Unexpected child layer '" << children[0]->name << "'";
        }
    }

    return true;
}

void FakeQuantizeAndActivationWithNegativeSlopeTestModel::resetTransformation(CNNNetwork& network) const {
    fillData(getLayer(network, "Const2"), 0.f, "custom");
    fillData(getLayer(network, "Const3"), 255.f / 8.f, "custom");
    fillData(getLayer(network, "Const4"), 0.f, "custom");
    fillData(getLayer(network, "Const5"), 255.f / 8.f, "custom");

    fillData(getLayer(network, "ScaleShift7"), 3.f, "weights");
    fillData(getLayer(network, "ScaleShift7"), 0.f, "biases");
}
