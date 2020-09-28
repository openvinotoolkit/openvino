// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformer_single_layer_tests.hpp"
#include "low_precision_transformations/fake_quantize.hpp"
#include "low_precision_transformations/convolution.hpp"
#include "low_precision_transformations/fully_connected.hpp"
#include "low_precision_transformations/scaleshift_to_convolution.hpp"

FullyConnectedTestModel::FullyConnectedTestModel(
    const std::vector<size_t>& inputDimentions,
    const std::vector<size_t>& outputDimentions) :
    addBiasesLayer(false),
    inputDimentions(inputDimentions),
    outputDimentions(outputDimentions) {}

std::string FullyConnectedTestModel::getName() const {
    return std::string("FullyConnectedTestModel") +
        (addBiasesLayer ? "WithBiases" : "") +
        "_D" + std::to_string(inputDimentions.size()) +
        "_D" + std::to_string(outputDimentions.size());
}

void FullyConnectedTestModel::initInput(Blob::Ptr input) const {
    fillDataWithInitValue(input, -1.f);
}

bool FullyConnectedTestModel::transform(CNNNetwork& network, LayerTransformation::Params& params) const {
    params.updatePrecisions = true;

    // TODO: use getLowPrecisionTransformer(params) instead
    LowPrecisionTransformer transformer(LowPrecisionTransformer::getAllTransformations(params).
        add<FullyConnectedTransformation>(LayerTransformation::Params(params).setSupportAsymmetricQuantization(false), "FullyConnected").
        add<ConvolutionTransformation>(LayerTransformation::Params(params).setPrecisionsOnActivations({ Precision::U8 }), "Convolution").
        addCleanup<ScaleShiftToConvolutionTransformation>(
            LayerTransformation::Params(params).setPrecisionsOnActivations({ Precision::U8 }),
            "ScaleShift"));

    // network.serialize("c:\\Projects\\temp\\fully_connected.original.xml", "c:\\Projects\\temp\\fully_connected.original.bin");
    transformer.transform(network);
    // network.serialize("c:\\Projects\\temp\\fully_connected.transformed.xml", "c:\\Projects\\temp\\fully_connected.transformed.bin");

    if (params.quantizeOutputs) {
        const CNNLayerPtr dequantizationLayer = getLayer(network, "fullyConnected");
        if (dequantizationLayer->type != "ScaleShift") {
            THROW_IE_EXCEPTION << "was not quantized";
        }

        const Blob::Ptr biases = CNNNetworkHelper::getBiases(*dequantizationLayer);
        const std::shared_ptr<float> biasesData = CNNNetworkHelper::getFloatData(biases);
        if (params.updateBiases) {
            for (size_t i = 0ul; i < biases->size(); ++i) {
                if (biasesData.get()[i] != 0.f) {
                    THROW_IE_EXCEPTION << "biases value is not zero";
                }
            }
        } else {
            // FakeQuantize layer has to have shift
            for (size_t i = 0ul; i < biases->size(); ++i) {
                if (biasesData.get()[i] == 0.f) {
                    THROW_IE_EXCEPTION << "biases value is zero";
                }
            }
        }
    }

    return true;
}

std::string FullyConnectedTestModel::getModel(SingleLayerTransformationsTestParams& p) const {
    size_t type_size = sizeof(InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type);
    if (p._network_precision == "FP16")
        type_size = sizeof(InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP16>::value_type);

    const size_t inputChannelsCount = p.inputDimensions[0][1];
    const size_t outputChannelsCount = p.outputDimensions[0][1];
    std::vector<size_t> weightsConstInputDims = {
        p.inputDimensions[0][2] * p.inputDimensions[0][3],
        p.outputDimensions[0][p.outputDimensions[0].size() == 2ul ? 1ul : 2ul] };

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
        std::vector<std::vector<size_t>>({ p.inputDimensions[0], weightsConstInputDims });

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
        std::vector<std::vector<size_t>>({ p.outputDimensions[0], weightsConstInputDims, biasesConstDims }) :
        std::vector<std::vector<size_t>>({ p.outputDimensions[0], weightsConstInputDims });

    std::vector<size_t> quantizationParamsDims(p.inputDimensions[0].size(), 1);
    quantizationParamsDims[1] = inputChannelsCount;

    const std::vector<size_t> reshape1OuputDims = { p.inputDimensions[0][0], p.inputDimensions[0][1], p.inputDimensions[0][2] * p.inputDimensions[0][3] };
    const std::vector<size_t> reshape2OuputDims = p.outputDimensions[0].size() == 2ul ?
        std::vector<size_t>({ p.inputDimensions[0][0] * p.inputDimensions[0][1], p.inputDimensions[0][2] * p.inputDimensions[0][3] }) :
        std::vector<size_t>({ p.inputDimensions[0][0], p.inputDimensions[0][1], p.inputDimensions[0][2] * p.inputDimensions[0][3] });

    CommonTestUtils::DefaultNetBuilder builder = CommonTestUtils::DefaultNetBuilder::buildNetworkWithOneInput(
        "FullyConnectedTestModel", p.inputDimensions[0], p._network_precision)
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
        .addLayer("Reshape", p._network_precision, &reshapeParams, { { p.inputDimensions[0] }, { reshape1OuputDims } }, "reshape1")
        // 8
        .addLayer("Reshape", p._network_precision, &reshapeParams, { {{ reshape1OuputDims }}, { reshape2OuputDims } }, "reshape2")
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

void FullyConnectedTestModel::resetTransformation(CNNNetwork& network) const {
    CNNLayerPtr fakeQuantize = CNNNetworkHelper::getLayer(network, "fakeQuantize");
    const size_t inputChannels = fakeQuantize->outData[0]->getTensorDesc().getDims()[1];

    CNNLayerPtr fullyConnected = CNNNetworkHelper::getLayer(network, "fullyConnected");
    const size_t outputChannels = fullyConnected->outData[0]->getTensorDesc().getDims()[1];

    // Const on activations
    //std::vector<float> lowValues(inputChannels, 1.0);  // to have shifts
    //std::vector<float> highValues(inputChannels);
    //if (areScalesOnActivationsDifferent()) {
    //    for (size_t inputChannel = 0; inputChannel < highValues.size(); ++inputChannel) {
    //        highValues[inputChannel] = static_cast<float>(inputChannel);
    //    }
    //}
    //else {
    //    highValues = std::vector<float>(inputChannels, 255.f);
    //}

    //std::vector<float> lowValues(inputChannels, 1.275f);
    //std::vector<float> highValues(inputChannels, 2.55f);

    std::vector<float> lowValues(inputChannels, 127.5f);
    std::vector<float> highValues(inputChannels, 255.f);

    fillData(getLayer(network, "dataInputLowConst"), lowValues, "custom");
    fillData(getLayer(network, "dataInputHighConst"), highValues, "custom");
    fillData(getLayer(network, "dataOutputLowConst"), lowValues, "custom");
    fillData(getLayer(network, "dataOutputHighConst"), highValues, "custom");


    const size_t fakeQuantizeInputChannel = outputChannels;

    // Const on weights
    //std::vector<float> weights(
    //    fakeQuantize->outData[0]->getTensorDesc().getDims()[2] *
    //    fakeQuantize->outData[0]->getTensorDesc().getDims()[3] *
    //    fullyConnected->outData[0]->getTensorDesc().getDims()[fullyConnected->outData[0]->getTensorDesc().getDims().size() == 2ul ? 1 : 2]);
    //for (size_t outputChannel = 0ul; outputChannel < outputChannels; ++outputChannel) {
    //    for (size_t inputChannel = 0ul; inputChannel < fakeQuantizeInputChannel; ++inputChannel) {
    //        weights[outputChannel * fakeQuantizeInputChannel + inputChannel] = inputChannel;
    //    }
    //}

    const std::vector<size_t> dims = fakeQuantize->outData[0]->getTensorDesc().getDims();
    // const size_t weightsSize = dims[2] * dims[3] * dims[dims.size() == 2ul ? 1 : 2];
    const size_t weightsSize = (dims[2] * dims[3]) * (dims[2] * dims[3]);
    std::vector<float> weights(weightsSize, 2.f);

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

bool FullyConnectedTestModel::areScalesOnActivationsDifferent() const {
    return false;
}
