// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformer_single_layer_tests.hpp"
#include "low_precision_transformations/fully_connected.hpp"

void PrecisionSelectionMultibranchPreservedTestModel::initInput(Blob::Ptr input) const {
    fillData(input, 2.f);
    return;

    const size_t dataSize = input->size();
    std::shared_ptr<float> floatPtr(new float[dataSize], std::default_delete<float[]>());

    const float lowValue = signedIntervalOnActivation ? -128.f : 0.f;
    const float highValue = signedIntervalOnActivation ? 127.f : 255.f;

    float value = lowValue;
    for (size_t i = 0ul; i < dataSize; ++i) {
        floatPtr.get()[i] = value;
        value += 1.f;
        if (value > highValue) {
            value = lowValue;
        }
    }

    CNNNetworkHelper::fillBlobByFP32(input, floatPtr.get());
}

PrecisionSelectionMultibranchPreservedTestModel::PrecisionSelectionMultibranchPreservedTestModel(const bool signedIntervalOnActivation) :
    signedIntervalOnActivation(signedIntervalOnActivation),
    acrossChannels(0),
    normalizeVariance(0) {}

std::string PrecisionSelectionMultibranchPreservedTestModel::getModel(SingleLayerTransformationsTestParams& p) const {
    size_t type_size = sizeof(InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type);
    if (p._network_precision == "FP16")
        type_size = sizeof(InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP16>::value_type);

    CommonTestUtils::conv_common_params conv =
            { {1, 1}, {3, 3}, {0, 0}, {0, 0}, {1, 1}, "valid", 1, 32, false, false };
    std::vector<size_t> convOutShape(p.inputDimensions[0].size());
    CommonTestUtils::getConvOutShape(p.inputDimensions[0], conv, convOutShape);

    std::vector<size_t> weightsConstInputDims = { 32lu, 32lu, 3lu, 3lu };
    std::vector<size_t> biasesConvolutionConstDims = { conv.out_c };
    std::map<std::string, std::string> const_params = {};
    std::map<std::string, std::string> fake_quantize_params = {
        {"levels", "256"}
    };
    std::map<std::string, std::string> power_params = { {"power", "1"}, {"scale", "1"}, {"shift", "0"}};
    std::map<std::string, std::string> poolingParams = {
        {"kernel", "1,1"},
        {"pool-method", "max"},
        {"exclude-pad", "false"}
    };
    const std::vector<size_t> dimensions = p.outputDimensions[0];

    std::vector<std::pair<std::string, std::string>> edges = {
        {"0,0", "1,1"}, {"1,2", "6,7"}, // Power
        {"2,3", "6,8"}, {"3,4", "6,9"}, {"4,5", "6,10"}, {"5,6", "6,11"}, // Const layers
        {"6,12", "7,13"},  // Fake quantize to Pooling7
        {"6,12", "8,15"}   // Fake quantize to Pooling8
    };

    return CommonTestUtils::DefaultNetBuilder::buildNetworkWithOneInput("QuantizationOnWeights", p.inputDimensions[0], p._network_precision)
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
        .addLayer(
            "FakeQuantize",
            p._network_precision,
            &fake_quantize_params,
            {{p.inputDimensions[0], {1}, {1}, {1}, {1}}, {{p.inputDimensions[0]}}},
            "fakeQuantize")
        // 7
        .addLayer("Pooling", p._network_precision, &poolingParams, { {dimensions}, {dimensions} })
        // 8
        .addLayer("Pooling", p._network_precision, &poolingParams, { {dimensions}, {dimensions} })
        // 9
        .finish(&edges);
}

void PrecisionSelectionMultibranchPreservedTestModel::resetTransformation(CNNNetwork& network) const {
    if (signedIntervalOnActivation) {
        fillData(getLayer(network, "Const2"), -128.f / 4.f, "custom");
        fillData(getLayer(network, "Const3"), 127.f / 4.f, "custom");
        fillData(getLayer(network, "Const4"), -128.f / 4.f, "custom");
        fillData(getLayer(network, "Const5"), 127.f / 4.f, "custom");
    } else {
        fillData(getLayer(network, "Const2"), 0.f, "custom");
        fillData(getLayer(network, "Const3"), 255.f / 4.f, "custom");
        fillData(getLayer(network, "Const4"), 0.f, "custom");
        fillData(getLayer(network, "Const5"), 255.f / 4.f, "custom");
    }
}

std::string PrecisionSelectionMultibranchPreservedTestModel::getName() const {
    return std::string("PrecisionSelectionMultibranchPreservedTestModel") + (signedIntervalOnActivation ? "_Signed" : "_Unsigned");
}

bool PrecisionSelectionMultibranchPreservedTestModel::transform(CNNNetwork& network, LayerTransformation::Params& params) const {
    params.updatePrecisions = true;

    LowPrecisionTransformer transformer = getLowPrecisionTransformer(params);
    transformer.transform(network);

    if (params.quantizeOutputs && params.updatePrecisions) {
        Precision expectedPrecision;
        if (params.precisionsOnActivations.size() == 1ul) {
            expectedPrecision = params.precisionsOnActivations[0];
        } else {
            expectedPrecision = signedIntervalOnActivation ? Precision::I8 : Precision::U8;
        }
        const CNNLayerPtr fakeQuantize = CNNNetworkHelper::getLayer(network, "fakeQuantize");
        const Precision actualPrecision = fakeQuantize->outData[0]->getTensorDesc().getPrecision();
        if (actualPrecision != expectedPrecision) {
            THROW_IE_EXCEPTION << "expected precision " << expectedPrecision << ", actual " << actualPrecision << "";
        }
    }

    return true;
}
