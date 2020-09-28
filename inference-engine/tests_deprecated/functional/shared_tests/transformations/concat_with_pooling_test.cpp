// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformer_single_layer_tests.hpp"
#include "low_precision_transformations/concat.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

std::string ConcatWithPoolingTestModel::getModel(SingleLayerTransformationsTestParams& p) const {
    size_t type_size = sizeof(PrecisionTrait<Precision::FP32>::value_type);
    if (p._network_precision == "FP16")
        type_size = sizeof(PrecisionTrait<Precision::FP16>::value_type);

    std::map<std::string, std::string> constParams = {};
    std::map<std::string, std::string> fakeQuantizeParams = { {"levels", "256"} };
    std::map<std::string, std::string> concatParams = { {"axis", "1"} };
    std::map<std::string, std::string> powerParams = { {"power", "1"}, {"scale", "1"}, {"shift", "0"} };
    std::map<std::string, std::string> poolingParams = {
        {"kernel", "1,1"},
        {"pool-method", "max"},
        {"exclude-pad", "false"}
    };

    CommonTestUtils::conv_common_params convolutionParams = { {1, 1}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, "valid", 1, 3, false, false };
    std::vector<size_t> weightsConstInputDims = { 3lu, 3lu, 1lu, 1lu };
    std::vector<size_t> biasesConvolutionConstDims = { convolutionParams.out_c };

    std::vector<std::pair<std::string, std::string>> edges = {
        {"0,0", "11,17"}, {"1,2", "6,7"}, // Inputs
        {"2,3", "6,8"}, {"3,4", "6,9"}, {"4,5", "6,10"}, {"5,6", "6,11"}, // Const layers
        {"7,13", "11,18"}, {"8,14", "11,19"}, {"9,15", "11,20"}, {"10,16", "11,21"}, // Const layers
        {"6,12", "17,33"}, {"11,22", "12,23"}, // Pooling12
        {"12,24", "15,27"}, // Pooling12 -> Convolution15
        {"13,25", "15,28"}, // Const13 -> Convolution15
        {"14,26", "15,29"}, // Const14 -> Convolution15
        {"15,30", "1,1"}, // Convolution15 -> Power
        {"12,24", "16,31"}, // Pooling12 -> Pooling16
        {"16,32", "17,34"}  // Pooling16 -> FakeQuantize20
    };

    auto modelBuilder = CommonTestUtils::DefaultNetBuilder::buildNetworkWithOneInput("ConcatWithPoolingTestModel", p.inputDimensions[0], p._network_precision)
        // 1
        //.addInputLayer(p._network_precision, p.inputDimensions[1])
        .addLayer("Power", p._network_precision, &powerParams, { {p.inputDimensions[1]}, {p.inputDimensions[1]} })
        // 2
        .addLayer("Const", p._network_precision, &constParams, { {}, {{1}} }, type_size, 0)
        // 3
        .addLayer("Const", p._network_precision, &constParams, { {}, {{1}} }, type_size, 0)
        // 4
        .addLayer("Const", p._network_precision, &constParams, { {}, {{1}} }, type_size, 0)
        // 5
        .addLayer("Const", p._network_precision, &constParams, { {}, {{1}} }, type_size, 0)
        // 6
        .addLayer("FakeQuantize", p._network_precision, &fakeQuantizeParams, { {p.inputDimensions[0], {1}, {1}, {1}, {1}}, {{p.inputDimensions[0]}} })
        // 7
        .addLayer("Const", p._network_precision, &constParams, { {}, {{1}} }, type_size, 0)
        // 8
        .addLayer("Const", p._network_precision, &constParams, { {}, {{1}} }, type_size, 0)
        // 9
        .addLayer("Const", p._network_precision, &constParams, { {}, {{1}} }, type_size, 0)
        // 10
        .addLayer("Const", p._network_precision, &constParams, { {}, {{1}} }, type_size, 0)
        // 11
        .addLayer("FakeQuantize", p._network_precision, &fakeQuantizeParams, { {p.inputDimensions[1], {1}, {1}, {1}, {1}}, {{p.inputDimensions[1]}} })
        // 12
        .addLayer("Pooling", p._network_precision, &poolingParams, { {p.inputDimensions[1]}, {p.inputDimensions[1]} })
        // 13
        .addLayer("Const", p._network_precision, &constParams, { {}, {weightsConstInputDims} },
            std::accumulate(weightsConstInputDims.begin(), weightsConstInputDims.end(), 1lu, std::multiplies<size_t>()) * type_size)
        // 14
        .addLayer("Const", p._network_precision, &constParams, { {}, {biasesConvolutionConstDims} }, type_size * convolutionParams.out_c, 0)
        // 15
        .convolutionLayer(p._network_precision, { {p.inputDimensions[0], weightsConstInputDims, biasesConvolutionConstDims }, {p.inputDimensions[0]} }, convolutionParams)
        // 16
        .addLayer("Pooling", p._network_precision, &poolingParams, { {p.inputDimensions[1]}, {p.inputDimensions[1]} })
        // 17
        .addLayer("Concat", p._network_precision, &concatParams, { {p.inputDimensions[0], p.inputDimensions[1]}, {{p.outputDimensions[0]}} }, 0, 0);

    auto modelString = modelBuilder.finish(&edges);
    return modelString;
}

std::string ConcatWithPoolingTestModel::getName() const {
    return std::string("ConcatWithPoolingTestModel") +
        (multiChannel ? "_multiChannel" : "_oneChannel") +
        (signedIntervals ? "_signedInterval" : "_notSignedInterval") +
        (shift ? "_withShift" : "") +
        "_" + std::to_string(dequantizationIntervalsDifference);
}

bool ConcatWithPoolingTestModel::transform(CNNNetwork& network, LayerTransformation::Params& params) const {
    // TODO: remove when updatePrecisions is configurable
    params.updatePrecisions = true;

    LowPrecisionTransformations transformations = getLowPrecisionTransformations(params);
    if (!multiChannel) {
        // avoid ConcatMultiChannelsTransformation
        transformations = transformations.
            removeBranchSpecificTransformations("Concat").
            addBranchSpecific<ConcatTransformation>(params, "Concat");
    }

    LowPrecisionTransformer transformer(transformations);
    transformer.transform(network);

    const std::string intermediateDequantizationLayerName = "Pooling12_ScaleShift_Convolution15";
    const CNNLayerPtr intermediateDequantizationLayer = CNNNetworkHelper::getLayer(network, intermediateDequantizationLayerName);
    if (intermediateDequantizationLayer == nullptr) {
        THROW_IE_EXCEPTION << "DequantizationLayer '" << intermediateDequantizationLayerName << "' was not found";
    }

    return true;
}

void ConcatWithPoolingTestModel::resetTransformation(CNNNetwork& network) const {
    const float low = signedIntervals ? -128 : 0.f;
    const float high = signedIntervals ? 127 : 255.f;

    const float coefficient1 = 10.f;
    const float coefficient2 = coefficient1 * dequantizationIntervalsDifference;
    const float shift1 = shift ? (low / coefficient1) / 3 : 0.f;
    const float shift2 = shift ? (low / coefficient1) / 3 : 0.f;

    fillData(getLayer(network, "Const2"), low / coefficient1 + shift1, "custom");
    fillData(getLayer(network, "Const3"), high / coefficient1, "custom");
    fillData(getLayer(network, "Const4"), low / coefficient1 + shift1, "custom");
    fillData(getLayer(network, "Const5"), high / coefficient1, "custom");

    fillData(getLayer(network, "Const7"), low / coefficient2 + shift2, "custom");
    fillData(getLayer(network, "Const8"), high / coefficient2, "custom");
    fillData(getLayer(network, "Const9"), low / coefficient2 + shift2, "custom");
    fillData(getLayer(network, "Const10"), high / coefficient2, "custom");

    fillData(getLayer(network, "Const13"), 3.f, "custom");
    fillData(getLayer(network, "Const14"), 2.f, "custom");
}

float ConcatWithPoolingTestModel::getThreshold(
    const std::string& deviceName,
    const Precision precision,
    LayerTransformation::Params& params) const {
    if (params.quantizeOutputs && signedIntervals && shift && (dequantizationIntervalsDifference != 0.f)) {
        return 0.0153;
    }

    return SingleLayerTestModel::getThreshold(deviceName, precision, params);
}
