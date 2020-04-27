// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformer_single_layer_tests.hpp"

//const size_t channelsCount = 32ul;
//const size_t group = channelsCount;
//std::vector<size_t> weightsConstInputDims = { channelsCount, 1lu, 3lu, 3lu };

ConvolutionBaseTestModel::ConvolutionBaseTestModel(const bool addBiasesLayer) : addBiasesLayer(addBiasesLayer) {}

std::string ConvolutionBaseTestModel::getModel(SingleLayerTransformationsTestParams& p) const {
    size_t type_size = sizeof(InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type);
    if (p._network_precision == "FP16")
        type_size = sizeof(InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP16>::value_type);

    const size_t group = getGroupsCount(p);
    const size_t inputChannelsCount = p.inputDimensions[0][1];
    const size_t outputChannelsCount = p.outputDimensions[0][1];
    CommonTestUtils::conv_common_params conv = { {1, 1}, {3, 3}, {1, 1}, {1, 1}, {1, 1}, "valid", group, outputChannelsCount, false, false };
    std::vector<size_t> weightsConstInputDims = { outputChannelsCount, inputChannelsCount / group, 3lu, 3lu };

    std::vector<size_t> convOutShape(p.inputDimensions[0].size());
    getConvOutShape(p.inputDimensions[0], conv, convOutShape);

    std::map<std::string, std::string> const_params = {};
    std::map<std::string, std::string> fake_quantize_params = { {"levels", "256"} };
    std::map<std::string, std::string> fake_quantize_params2 = { {"levels", "255"} };
    std::map<std::string, std::string> power_params = {
        {"power", "1"}, {"scale", "1"}, {"shift", "0"}
    };

    std::vector<size_t> biasesConvolutionConstDims = { conv.out_c };

    const std::vector<std::vector<size_t>> convolutionDims = addBiasesLayer ?
        std::vector<std::vector<size_t>>({p.inputDimensions[0], weightsConstInputDims, biasesConvolutionConstDims }) :
        std::vector<std::vector<size_t>>({p.inputDimensions[0], weightsConstInputDims });

    std::vector<std::pair<std::string, std::string>> edges = {
        {"0,0", "1,1"}, {"1,2", "6,7"}, // Power
        {"2,3", "6,8"}, {"3,4", "6,9"}, {"4,5", "6,10"}, {"5,6", "6,11"}, // Const layers
        {"7,13", "12,18"}, {"8,14", "12,19"}, {"9,15", "12,20"}, {"10,16", "12,21"}, {"11,17", "12,22"}, // Const layers
        {"6,12", "13,24"},  {"12,23", "13,25"} // Fake quantize to Conv
    };

    if (addBiasesLayer) {
        edges.push_back({ "14,28", "13,26" }); // biases to Conv
    }

    std::vector<size_t> quantizationParamsDims(p.inputDimensions[0].size(), 1);
    quantizationParamsDims[1] = inputChannelsCount;

    CommonTestUtils::DefaultNetBuilder builder = CommonTestUtils::DefaultNetBuilder::buildNetworkWithOneInput(
        "QuantizationOnWeights", p.inputDimensions[0], p._network_precision)
        .addLayer("Power", p._network_precision, &power_params, { {p.inputDimensions[0]}, {p.inputDimensions[0]} })
        .addLayer("Const", p._network_precision, &const_params, { {}, {quantizationParamsDims} }, inputChannelsCount * type_size, "dataInputLowConst")
        .addLayer("Const", p._network_precision, &const_params, { {}, {quantizationParamsDims} }, inputChannelsCount * type_size, "dataInputHighConst")
        .addLayer("Const", p._network_precision, &const_params, { {}, {quantizationParamsDims} }, inputChannelsCount * type_size, "dataOutputLowConst")
        .addLayer("Const", p._network_precision, &const_params, { {}, {quantizationParamsDims} }, inputChannelsCount * type_size, "dataOutputHighConst")
        .addLayer("FakeQuantize",
            p._network_precision,
            &fake_quantize_params,
            { {p.inputDimensions[0], quantizationParamsDims, quantizationParamsDims, quantizationParamsDims, quantizationParamsDims},
              {{p.inputDimensions[0]}} },
            "fakeQuantizeOnActivations")
        .addLayer("Const", p._network_precision, &const_params, { {}, {weightsConstInputDims} },
            std::accumulate(weightsConstInputDims.begin(), weightsConstInputDims.end(), 1lu, std::multiplies<size_t>()) * type_size, "weigthsConst")
        .addLayer("Const", p._network_precision, &const_params, { {}, {{1}} }, type_size, "weigthsInputLowConst")
        .addLayer("Const", p._network_precision, &const_params, { {}, {{1}} }, type_size, "weigthsInputHighConst")
        .addLayer("Const", p._network_precision, &const_params, { {}, {{1}} }, type_size, "weigthsOutputLowConst")
        .addLayer("Const", p._network_precision, &const_params, { {}, {{1}} }, type_size, "weigthsOutputHighConst")
        .addLayer(
            "FakeQuantize",
            p._network_precision,
            &fake_quantize_params,
            { {weightsConstInputDims, {1}, {1}, {1}, {1}}, {{weightsConstInputDims}} },
            "fakeQuantizeOnWeights")
        .convolutionLayer(p._network_precision, { convolutionDims, {convOutShape} }, conv, {}, "Convolution");

    if (addBiasesLayer) {
        builder.addLayer("Const", p._network_precision, &const_params, { {}, {biasesConvolutionConstDims} }, type_size * conv.out_c, "biasesConst");
    }

    return builder.finish(&edges);
}

bool ConvolutionBaseTestModel::transform(CNNNetwork& network, LayerTransformation::Params& params) const {
    LowPrecisionTransformer transformer = getLowPrecisionTransformer(params);
    transformer.transform(network);
    return true;
}

void ConvolutionBaseTestModel::resetTransformation(CNNNetwork& network) const {
    CNNLayerPtr convolution = CNNNetworkHelper::getLayer(network, "Convolution");

    const size_t channelsCount = convolution->GetParamAsUInt("output");
    const size_t groupsCount = convolution->GetParamAsUInt("group");
    const size_t filtersCountPerOutputChannel = channelsCount / groupsCount;
    const size_t kernelH = convolution->GetParamAsUInts("kernel")[0];
    const size_t kernelW = convolution->GetParamAsUInts("kernel")[1];

    // Const on activations
    std::vector<float> lowValues(channelsCount);  // to have shifts
    std::vector<float> highValues(channelsCount);
    if (areScalesOnActivationsDifferent()) {
        for (size_t inputChannel = 0; inputChannel < highValues.size(); ++inputChannel) {
            highValues[inputChannel] = 255.f / (1.f + inputChannel);
        }
    } else {
        highValues = std::vector<float>(channelsCount, 255.f);
    }

    fillData(getLayer(network, "dataInputLowConst"), lowValues, "custom");
    fillData(getLayer(network, "dataInputHighConst"), highValues, "custom");
    fillData(getLayer(network, "dataOutputLowConst"), lowValues, "custom");
    fillData(getLayer(network, "dataOutputHighConst"), highValues, "custom");

    // Const on weights
    std::vector<float> weights(channelsCount * filtersCountPerOutputChannel * kernelH * kernelW);
    for (size_t outputChannel = 0ul; outputChannel < channelsCount; ++outputChannel) {
        for (size_t filter = 0ul; filter < filtersCountPerOutputChannel; ++filter) {
            for (size_t kernel = 0ul; kernel < kernelH * kernelW; ++kernel) {
                weights[outputChannel * filtersCountPerOutputChannel * kernelH * kernelW + filter * kernelH * kernelW + kernel] =
                    static_cast<float>(outputChannel * filtersCountPerOutputChannel + filter) + 1.f;
            }
        }
    }
    fillData(getLayer(network, "weigthsConst"), weights, "custom");

    fillData(getLayer(network, "weigthsInputLowConst"), -128.f / 4.0, "custom");
    fillData(getLayer(network, "weigthsInputHighConst"), 127.f / 4.0, "custom");
    fillData(getLayer(network, "weigthsOutputLowConst"), -128.f / 4.0, "custom");
    fillData(getLayer(network, "weigthsOutputHighConst"), 127.f / 4.0, "custom");

    if (addBiasesLayer) {
        fillData(getLayer(network, "biasesConst"), 2.f, "custom");
    }
}

size_t ConvolutionBaseTestModel::getGroupsCount(SingleLayerTransformationsTestParams& p) const {
    return 1ul;
}

bool ConvolutionBaseTestModel::areScalesOnActivationsDifferent() const {
    return false;
}
