// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformer_single_layer_tests.hpp"

//const size_t channelsCount = 32ul;
//const size_t group = channelsCount;
//std::vector<size_t> weightsConstInputDims = { channelsCount, 1lu, 3lu, 3lu };

FullyConnectedBaseTestModel::FullyConnectedBaseTestModel(const bool addBiasesLayer) : addBiasesLayer(addBiasesLayer) {}

std::string FullyConnectedBaseTestModel::getModel(SingleLayerTransformationsTestParams& p) const {
    size_t type_size = sizeof(InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type);
    if (p._network_precision == "FP16")
        type_size = sizeof(InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP16>::value_type);

    const size_t inputChannelsCount = p.inputDimensions[0][1];
    const size_t outputChannelsCount = p.outputDimensions[0][1];
    //conv_common_params conv = { {1, 1}, {3, 3}, {1, 1}, {1, 1}, {1, 1}, "valid", group, outputChannelsCount, false, false };
    std::vector<size_t> weightsConstInputDims = { outputChannelsCount, inputChannelsCount };

    //std::vector<size_t> convOutShape(p.inputDimensions[0].size());
    //getConvOutShape(p.inputDimensions[0], conv, convOutShape);

    std::map<std::string, std::string> const_params = {};
    std::map<std::string, std::string> fake_quantize_params = { {"levels", "256"} };
    std::map<std::string, std::string> fake_quantize_params2 = { {"levels", "255"} };
    std::map<std::string, std::string> power_params = { {"power", "1"}, {"scale", "1"}, {"shift", "0"} };
    std::map<std::string, std::string> poolingParams = { {"kernel", "112,112"}, {"pool-method", "max"} };
    std::map<std::string, std::string> reshapeParams = { };
    std::map<std::string, std::string> fullyConnectedParams = { {"out-size", std::to_string(p.outputDimensions[0][1])} };

    std::vector<size_t> biasesConstDims = { p.outputDimensions[0][1] };

    const std::vector<std::vector<size_t>> convolutionDims = addBiasesLayer ?
        std::vector<std::vector<size_t>>({ p.inputDimensions[0], weightsConstInputDims, biasesConstDims }) :
        std::vector<std::vector<size_t>>({p.inputDimensions[0], weightsConstInputDims });

    std::vector<std::pair<std::string, std::string>> edges = {
        {"0,0", "1,1"}, {"1,2", "6,7"}, // Power
        {"2,3", "6,8"}, {"3,4", "6,9"}, {"4,5", "6,10"}, {"5,6", "6,11"}, // Const layers
        {"6,12", "7,13"},  // FakeQuantize to Pooling
        {"7,14", "8,15"},  // Pooling to Reshape
        {"8,16", "15,28"},  // Reshape to FullyConnected
        {"9,17", "14,22"}, {"10,18", "14,23"}, {"11,19", "14,24"}, {"12,20", "14,25"}, {"13,21", "14,26"}, // Const layers
        {"14,27", "15,29"}
    };

    if (addBiasesLayer) {
        edges.push_back({ "16,32", "15,30" }); // biases to Conv
    }

    const std::vector<std::vector<size_t>> fullyConnectedDims = addBiasesLayer ?
        std::vector<std::vector<size_t>>({ {p.inputDimensions[0][0], p.inputDimensions[0][1]}, weightsConstInputDims, biasesConstDims }) :
        std::vector<std::vector<size_t>>({ {p.inputDimensions[0][0], p.inputDimensions[0][1]}, weightsConstInputDims });

    std::vector<size_t> quantizationParamsDims(p.inputDimensions[0].size(), 1);
    quantizationParamsDims[1] = inputChannelsCount;

    CommonTestUtils::DefaultNetBuilder builder = CommonTestUtils::DefaultNetBuilder::buildNetworkWithOneInput(
        "FullyConnectedBaseTestModel", p.inputDimensions[0], p._network_precision)
        // 1
        .addLayer("Power", p._network_precision, &power_params, { {p.inputDimensions[0]}, {p.inputDimensions[0]} })
        // 2
        .addLayer("Const", p._network_precision, &const_params, { {}, {quantizationParamsDims} }, inputChannelsCount * type_size, "dataInputLowConst")
        // 3
        .addLayer("Const", p._network_precision, &const_params, { {}, {quantizationParamsDims} }, inputChannelsCount * type_size, "dataInputHighConst")
        // 4
        .addLayer("Const", p._network_precision, &const_params, { {}, {quantizationParamsDims} }, inputChannelsCount * type_size, "dataOutputLowConst")
        // 5
        .addLayer("Const", p._network_precision, &const_params, { {}, {quantizationParamsDims} }, inputChannelsCount * type_size, "dataOutputHighConst")
        // 6
        .addLayer("FakeQuantize",
            p._network_precision,
            &fake_quantize_params,
            { {p.inputDimensions[0], quantizationParamsDims, quantizationParamsDims, quantizationParamsDims, quantizationParamsDims}, {{p.inputDimensions[0]}} },
            "fakeQuantize")
        // 7
        .addLayer("Pooling", p._network_precision, &poolingParams, { {p.inputDimensions[0]}, {{1, 32, 1, 1}} }, "pooling")
        // 8
        .addLayer("Reshape", p._network_precision, &reshapeParams, { {{1, 32, 1, 1}}, {{1, 32}} }, "reshape")
        // 9
        .addLayer("Const", p._network_precision, &const_params, { {}, {weightsConstInputDims} },
            std::accumulate(weightsConstInputDims.begin(), weightsConstInputDims.end(), 1lu, std::multiplies<size_t>()) * type_size, "weigthsConst")
        // 10
        .addLayer("Const", p._network_precision, &const_params, { {}, {{1}} }, type_size, "weigthsInputLowConst")
        // 11
        .addLayer("Const", p._network_precision, &const_params, { {}, {{1}} }, type_size, "weigthsInputHighConst")
        // 12
        .addLayer("Const", p._network_precision, &const_params, { {}, {{1}} }, type_size, "weigthsOutputLowConst")
        // 13
        .addLayer("Const", p._network_precision, &const_params, { {}, {{1}} }, type_size, "weigthsOutputHighConst")
        // 14
        .addLayer(
            "FakeQuantize",
            p._network_precision,
            &fake_quantize_params,
            { {weightsConstInputDims, {1}, {1}, {1}, {1}}, {{weightsConstInputDims}} },
            "fakeQuantizeOnWeights")
        // 15
        .addLayer("FullyConnected", p._network_precision, &fullyConnectedParams, { fullyConnectedDims, {p.outputDimensions[0]} }, "fullyConnected");

    if (addBiasesLayer) {
        // 16
        builder.addLayer("Const", p._network_precision, &const_params, { {}, {biasesConstDims} }, type_size * biasesConstDims[0], "biasesConst");
    }

    return builder.finish(&edges);
}

bool FullyConnectedBaseTestModel::transform(CNNNetwork& network, LayerTransformation::Params& params) const {
    LowPrecisionTransformer transformer(LowPrecisionTransformer::getAllTransformations(params));
    transformer.transform(network);
    return true;
}

void FullyConnectedBaseTestModel::resetTransformation(CNNNetwork& network) const {
    CNNLayerPtr fakeQuantize = CNNNetworkHelper::getLayer(network, "fakeQuantize");
    const size_t inputChannels = fakeQuantize->outData[0]->getTensorDesc().getDims()[1];

    CNNLayerPtr fullyConnected = CNNNetworkHelper::getLayer(network, "fullyConnected");
    const size_t outputChannels = fullyConnected->outData[0]->getTensorDesc().getDims()[1];

    // Const on activations
    std::vector<float> lowValues(inputChannels, 1.0);  // to have shifts
    std::vector<float> highValues(inputChannels);
    if (areScalesOnActivationsDifferent()) {
        for (size_t inputChannel = 0; inputChannel < highValues.size(); ++inputChannel) {
            highValues[inputChannel] = static_cast<float>(inputChannel);
        }
    } else {
        highValues = std::vector<float>(inputChannels, 255.f);
    }

    fillData(getLayer(network, "dataInputLowConst"), lowValues, "custom");
    fillData(getLayer(network, "dataInputHighConst"), highValues, "custom");
    fillData(getLayer(network, "dataOutputLowConst"), lowValues, "custom");
    fillData(getLayer(network, "dataOutputHighConst"), highValues, "custom");

    // Const on weights
    std::vector<float> weights(outputChannels * inputChannels);
    for (size_t outputChannel = 0ul; outputChannel < outputChannels; ++outputChannel) {
        for (size_t inputChannel = 0ul; inputChannel < inputChannels; ++inputChannel) {
            weights[outputChannel * inputChannels + inputChannel] = inputChannel;
        }
    }
    fillData(getLayer(network, "weigthsConst"), weights, "custom");

    fillData(getLayer(network, "weigthsInputLowConst"), -128.f, "custom");
    fillData(getLayer(network, "weigthsInputHighConst"), 127.f, "custom");
    fillData(getLayer(network, "weigthsOutputLowConst"), -128.f, "custom");
    fillData(getLayer(network, "weigthsOutputHighConst"), 127.f, "custom");

    if (addBiasesLayer) {
        std::vector<float> biases(outputChannels);
        for (size_t i = 0ul; i < outputChannels; ++i) {
            biases[i] = static_cast<float>(i);
        }
        fillData(getLayer(network, "biasesConst"), biases, "custom");
    }
}

bool FullyConnectedBaseTestModel::areScalesOnActivationsDifferent() const {
    return false;
}
