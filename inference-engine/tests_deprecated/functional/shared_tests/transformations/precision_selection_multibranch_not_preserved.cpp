// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformer_single_layer_tests.hpp"
#include "low_precision_transformations/fully_connected.hpp"

void PrecisionSelectionMultibranchNotPreservedTestModel::initInput(Blob::Ptr input) const {
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

PrecisionSelectionMultibranchNotPreservedTestModel::PrecisionSelectionMultibranchNotPreservedTestModel(const bool signedIntervalOnActivation) :
    signedIntervalOnActivation(signedIntervalOnActivation),
    acrossChannels(0),
    normalizeVariance(0) {}

std::string PrecisionSelectionMultibranchNotPreservedTestModel::getModel(SingleLayerTransformationsTestParams& p) const {
    size_t type_size = sizeof(InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type);
    if (p._network_precision == "FP16")
        type_size = sizeof(InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP16>::value_type);

    CommonTestUtils::conv_common_params conv =
            { {1, 1}, {3, 3}, {0, 0}, {0, 0}, {1, 1}, "valid", 1, 32, false, false };
    std::vector<size_t> convOutShape(p.inputDimensions[0].size());
    getConvOutShape(p.inputDimensions[0], conv, convOutShape);

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
        {"7,13", "12,18"}, {"8,14", "12,19"}, {"9,15", "12,20"}, {"10,16", "12,21"}, {"11,17", "12,22"}, // Const layers
        {"6,12", "14,25"},  {"12,23", "14,26"}, // Fake quantize to Conv
        {"13,24", "14,27"}, // biases to Conv
        {"6,12", "15,29"} // Fake quantize to Pooling
        //{"14,28", "15,29"} // Fake quantize to Power
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
        .addLayer("Const", p._network_precision, &const_params, {{}, {weightsConstInputDims}},
            std::accumulate(weightsConstInputDims.begin(), weightsConstInputDims.end(), 1lu, std::multiplies<size_t>()) * type_size)
        // 8
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        // 9
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        // 10
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        // 11
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        // 12
        .addLayer("FakeQuantize", p._network_precision, &fake_quantize_params, {{weightsConstInputDims, {1}, {1}, {1}, {1}}, {{weightsConstInputDims}}})
        // 13
        .addLayer("Const", p._network_precision, &const_params, {{}, {biasesConvolutionConstDims}}, type_size * conv.out_c, 0)
        // 14
        .convolutionLayer(
            p._network_precision,
            { {p.inputDimensions[0], weightsConstInputDims, biasesConvolutionConstDims },
            {convOutShape} }, conv, {}, "convolution")
        // 15
        .addLayer("Pooling", p._network_precision, &poolingParams, { {dimensions}, {dimensions} })
        .finish(&edges);
}

void PrecisionSelectionMultibranchNotPreservedTestModel::resetTransformation(CNNNetwork& network) const {
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

    fillDataWithInitValue(getLayer(network, "Const7"), "custom", 2.f);

    fillData(getLayer(network, "Const8"), -128.f / 4.f, "custom");
    fillData(getLayer(network, "Const9"), 127.f / 4.f, "custom");
    fillData(getLayer(network, "Const10"), -128.f / 4.f, "custom");
    fillData(getLayer(network, "Const11"), 127.f / 4.f, "custom");

    fillDataWithInitValue(getLayer(network, "Const13"), "custom", 1.f);
}

std::string PrecisionSelectionMultibranchNotPreservedTestModel::getName() const {
    return std::string("PrecisionSelectionMultibranchNotPreservedTestModel") + (signedIntervalOnActivation ? "_Signed" : "_Unsigned");
}

bool PrecisionSelectionMultibranchNotPreservedTestModel::transform(CNNNetwork& network, LayerTransformation::Params& params) const {
    params.weightsToConst = true;
    params.updatePrecisions = true;

    LowPrecisionTransformer transformer = getLowPrecisionTransformer(params);
    transformer.transform(network);

    const CNNLayerPtr fakeQuantize = CNNNetworkHelper::getLayer(network, "fakeQuantize");
    const Precision actualPrecision = fakeQuantize->outData[0]->getTensorDesc().getPrecision();

    if (std::any_of(
        params.precisionsOnActivations.begin(),
        params.precisionsOnActivations.end(),
        [&](const Precision precision) { return precision == Precision::U8; })) {
        if (params.quantizeOutputs) {
            if (actualPrecision != Precision::U8) {
                THROW_IE_EXCEPTION << "expected precision " << Precision::U8 << ", actual " << actualPrecision << "";
            }

            // Convolution has to be quantized
            CNNLayerPtr scaleShfit = CNNNetworkHelper::getLayer(network, "convolution");
            if (scaleShfit->type != "ScaleShift") {
                THROW_IE_EXCEPTION << "unexpected last output dequantization layer type " << scaleShfit->type << " " << scaleShfit->name;
            }

            if (params.updateBiases) {
                const Blob::Ptr shiftsBlob = CNNNetworkHelper::getBlob(scaleShfit, "biases");
                std::shared_ptr<float> shiftsBuffer = CNNNetworkHelper::getFloatData(shiftsBlob);
                for (size_t i = 0ul; i < shiftsBlob->size(); ++i) {
                    if (shiftsBuffer.get()[i] != 0.0) {
                        THROW_IE_EXCEPTION << "unexpected dequantization shift value";
                    }
                }
            }

            //if (signedIntervalOnActivation)
            //scaleShfit = CNNNetworkHelper::getLayer(network, "MVN15");
            //if (scaleShfit->type != "ScaleShift") {
            //    THROW_IE_EXCEPTION << "unexpected last output dequantization layer type " << scaleShfit->type << " " << scaleShfit->name;
            //}
        }

        return true;
    } else {
        if ((actualPrecision != Precision::FP16) && (actualPrecision != Precision::FP32)) {
            THROW_IE_EXCEPTION << "unexpected precision " << actualPrecision << "";
        }

        // convolution can not be quantized
        CNNLayerPtr convolution = CNNNetworkHelper::getLayer(network, "convolution");
        if (convolution->type != "Convolution") {
            THROW_IE_EXCEPTION << "unexpected last output dequantization layer type " << convolution->type << " " << convolution->name;
        }

        const std::vector<CNNLayerPtr> parents = CNNNetworkHelper::getParents(*convolution);
        if (parents.size() != 3ul) {
            THROW_IE_EXCEPTION << "unexpected parents count " << parents.size();
        }

        if (parents[0]->type != "FakeQuantize") {
            THROW_IE_EXCEPTION << "unexpected parents type " << parents[0]->type;
        }

        return false;
    }
}
