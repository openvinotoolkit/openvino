// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformer_single_layer_tests.hpp"
#include "low_precision_transformations/scaleshift_to_convolution.hpp"

ScaleShiftToConvolutionAfterConcatTestModel::ScaleShiftToConvolutionAfterConcatTestModel(const bool scaleShiftIsOutput) :
    scaleShiftIsOutput(scaleShiftIsOutput) {}

std::string ScaleShiftToConvolutionAfterConcatTestModel::getModel(SingleLayerTransformationsTestParams& p) const {
//    ASSERT_EQ(2, p.inputDimensions.size());
    size_t type_size = sizeof(InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type);
    if (p._network_precision == "FP16")
        type_size = sizeof(InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP16>::value_type);

    const size_t axis = 1; // should be passed in 'p' argument

    std::vector<size_t> concat_out_dims = p.inputDimensions[0];
    concat_out_dims[axis] += p.inputDimensions[1][axis];

    std::map<std::string, std::string> const_params = {};
    std::map<std::string, std::string> fake_quantize_params = {
        {"levels", "256"}
    };
    std::map<std::string, std::string> concat_params = {
        {"axis", "1"}
    };
    std::map<std::string, std::string> power_params = {
        {"power", "1"}, {"scale", "1"}, {"shift", "0"}
    };

    std::vector<std::pair<std::string, std::string>> edges = {
        {"0,0", "10,10"}, {"1,1", "11,16"}, // Inputs to FakeQuantize
        {"2,2", "10,11"}, {"3,3", "10,12"}, {"4,4", "10,13"}, {"5,5", "10,14"}, // Const layers
        {"6,6", "11,17"}, {"7,7", "11,18"}, {"8,8", "11,19"}, {"9,9", "11,20"}, // Const layers
        {"10,15", "12,22"}, {"11,21", "12,23"}, // FakeQuantize to Concat
        {"12,24", "13,25"} // Concat to ScaleShift
    };

    if (!scaleShiftIsOutput) {
        edges.push_back({ "13,26", "14,27" });
    }

    auto layers = CommonTestUtils::DefaultNetBuilder::buildNetworkWithOneInput("ScaleShiftToConvolutionAfterConcatTestModel", p.inputDimensions[0], p._network_precision)
        .addInputLayer(p._network_precision, p.inputDimensions[1])
        .addLayer("Const", p._network_precision, &const_params, { {}, {{1}} }, type_size, 0)
        .addLayer("Const", p._network_precision, &const_params, { {}, {{1}} }, type_size, 0)
        .addLayer("Const", p._network_precision, &const_params, { {}, {{1}} }, type_size, 0)
        .addLayer("Const", p._network_precision, &const_params, { {}, {{1}} }, type_size, 0)
        .addLayer("Const", p._network_precision, &const_params, { {}, {{1}} }, type_size, 0)
        .addLayer("Const", p._network_precision, &const_params, { {}, {{1}} }, type_size, 0)
        .addLayer("Const", p._network_precision, &const_params, { {}, {{1}} }, type_size, 0)
        .addLayer("Const", p._network_precision, &const_params, { {}, {{1}} }, type_size, 0)
        .addLayer("FakeQuantize", p._network_precision, &fake_quantize_params, { {p.inputDimensions[0], {1}, {1}, {1}, {1}}, {{p.inputDimensions[0]}} })
        .addLayer("FakeQuantize", p._network_precision, &fake_quantize_params, { {p.inputDimensions[1], {1}, {1}, {1}, {1}}, {{p.inputDimensions[1]}} })
        .addLayer("Concat", p._network_precision, &concat_params, { {p.inputDimensions[0], p.inputDimensions[1]}, { concat_out_dims } })
        .addLayer("ScaleShift", p._network_precision, {}, { {p.outputDimensions[0]}, {p.outputDimensions[0]} }, p.outputDimensions[0][1] * type_size, p.outputDimensions[0][1] * type_size);

    if (!scaleShiftIsOutput) {
        layers.addLayer("Power", p._network_precision, &power_params, { {p.outputDimensions[0]}, {p.outputDimensions[0]} });
    }

    return layers.finish(&edges);
}

std::string ScaleShiftToConvolutionAfterConcatTestModel::getName() const {
    return std::string("ScaleShiftToConvolutionAfterConcatTestModel") +
        (scaleShiftIsOutput ? "_scaleShiftIsOutput" : "_scaleShiftIsNotOutput");
}

bool ScaleShiftToConvolutionAfterConcatTestModel::transform(CNNNetwork& network, LayerTransformation::Params& params) const {
    if (std::any_of(
        params.precisionsOnActivations.begin(),
        params.precisionsOnActivations.end(),
        [](const Precision precision) { return precision == Precision::U8; })) {
        params.updatePrecisions = true;
    }

    LowPrecisionTransformer transformer(LowPrecisionTransformer::getAllTransformations(params).
        addCleanup<ScaleShiftToConvolutionTransformation>(
            LayerTransformation::Params(params).setPrecisionsOnActivations({ Precision::U8 }),
            "ScaleShift"));

    transformer.transform(network);

    if (scaleShiftIsOutput || (!params.updatePrecisions)) {
        CNNLayerPtr scaleShift = CNNNetworkHelper::getLayer(network, "ScaleShift13");
        if (scaleShift->type != "ScaleShift") {
            THROW_IE_EXCEPTION << "unexpected layer type " << scaleShift->type << " '" << scaleShift->name << "'";
        }
    } else {
        CNNLayerPtr convolution = CNNNetworkHelper::getLayer(network, "ScaleShift13");
        if (convolution->type != "Convolution") {
            THROW_IE_EXCEPTION << "unexpected layer type " << convolution->type << " '" << convolution->name << "'";
        }

        if (CNNNetworkHelper::getInputChannelsCount(*convolution) != CNNNetworkHelper::getOutputChannelsCount(*convolution)) {
            THROW_IE_EXCEPTION <<
                "input channels count " << CNNNetworkHelper::getInputChannelsCount(*convolution) <<
                " is not not equal output channels count " << CNNNetworkHelper::getOutputChannelsCount(*convolution);
        }

        const std::vector<CNNLayerPtr> parents = CNNNetworkHelper::getParents(*convolution);

        const Blob::Ptr weightsBlob = CNNNetworkHelper::getBlob(parents[1], "custom");
        if (weightsBlob == nullptr) {
            THROW_IE_EXCEPTION << "weights are absent";
        }
        if (weightsBlob->getTensorDesc().getPrecision() != Precision::FP16) {
            const std::shared_ptr<float> weightsData = CNNNetworkHelper::getFloatData(weightsBlob);
            if (weightsData == nullptr) {
                THROW_IE_EXCEPTION << "weights are not received";
            }
            const float* weights = weightsData.get();
            size_t notZeroWeightsValues = 0ul;
            for (size_t i = 0ul; i < weightsBlob->size(); ++i) {
                if (weights[i] != 0.f) {
                    notZeroWeightsValues++;
                }
            }
            if (notZeroWeightsValues != CNNNetworkHelper::getOutputChannelsCount(*convolution)) {
                THROW_IE_EXCEPTION << "unexpected weights not zero values " << notZeroWeightsValues;
            }
        }

        const Blob::Ptr biasesBlob = CNNNetworkHelper::getBlob(parents[2], "custom");
        if (biasesBlob == nullptr) {
            THROW_IE_EXCEPTION << "biases are absent";
        }
        const std::shared_ptr<float> biases = CNNNetworkHelper::getFloatData(biasesBlob);
        if (biases == nullptr) {
            THROW_IE_EXCEPTION << "biases are not received";
        }
    }

    return true;
}

void ScaleShiftToConvolutionAfterConcatTestModel::resetTransformation(CNNNetwork& network) const {
    fillData(getLayer(network, "Const2"), 0.0, "custom");
    fillData(getLayer(network, "Const3"), 255.0 / 10.0, "custom");
    fillData(getLayer(network, "Const4"), 0.0, "custom");
    fillData(getLayer(network, "Const5"), 255.0 / 10.0, "custom");

    fillData(getLayer(network, "Const6"), -255.0 / 400.0, "custom");
    fillData(getLayer(network, "Const7"), 255.0 / 200.0, "custom");
    fillData(getLayer(network, "Const8"), -255.0 / 400.0, "custom");
    fillData(getLayer(network, "Const9"), 255.0 / 200.0, "custom");
}
