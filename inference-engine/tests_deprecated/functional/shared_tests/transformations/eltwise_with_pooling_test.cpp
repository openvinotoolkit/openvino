// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformer_single_layer_tests.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

std::string EltwiseWithPoolingTestModel::getModel(SingleLayerTransformationsTestParams& p) const {
    size_t type_size = sizeof(PrecisionTrait<Precision::FP32>::value_type);
    if (p._network_precision == "FP16")
        type_size = sizeof(PrecisionTrait<Precision::FP16>::value_type);

    std::map<std::string, std::string> constParams = {};
    std::map<std::string, std::string> fakeQuantizeParams = { {"levels", "256"} };
    std::map<std::string, std::string> eltwiseParams = { {"operation", operation} };
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

    auto modelBuilder = CommonTestUtils::DefaultNetBuilder::buildNetworkWithOneInput("EltwiseWithPoolingTestModel", p.inputDimensions[0], p._network_precision)
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
        .addLayer("Eltwise", p._network_precision, &eltwiseParams, { {p.inputDimensions[0], p.inputDimensions[1]}, {{p.inputDimensions[0]}} }, 0, 0);

    auto modelString = modelBuilder.finish(&edges);
    return modelString;
}

std::string EltwiseWithPoolingTestModel::getName() const {
    return std::string("EltwiseWithPoolingTestModel") +
        (cpuSpecific ? "_cpuSpecific" : "") +
        "_" + operation +
        (signedIntervals ? "_signedInterval" : "_notSignedInterval") +
        (minLevels != 2ul ? ("_minLevels" + std::to_string(minLevels)) : "");
}

bool EltwiseWithPoolingTestModel::transform(CNNNetwork& network, LayerTransformation::Params& params) const {
    if (std::any_of(
        params.precisionsOnActivations.begin(),
        params.precisionsOnActivations.end(),
        [](const Precision precision) { return precision == Precision::U8; })) {
        params.updatePrecisions = true;
    }

    LowPrecisionTransformations transformations = getLowPrecisionTransformations(params);
    if (cpuSpecific) {
        transformations = transformations.
            remove("Eltwise").
            add<EltwiseTransformation>(LayerTransformation::Params(params), "Eltwise");
    } else {
        THROW_IE_EXCEPTION << "not CPU/GPU specific Eltwise is not supported";
    }

    LayerTransformationPtr eltwiseTransformation = transformations.find("Eltwise");
    eltwiseTransformation->setMinQuantizationLevels(minLevels);

    LowPrecisionTransformer transformer(transformations);
    transformer.transform(network);

    if (params.quantizeOutputs && params.updatePrecisions) {
        // INT8 way
        const CNNLayerPtr fakeQuantize11 = getLayer(network, "FakeQuantize11");
        if ((fakeQuantize11->outData[0]->getPrecision() != Precision::U8) && (fakeQuantize11->outData[0]->getPrecision() != Precision::I8)) {
            THROW_IE_EXCEPTION <<
                "layer " << fakeQuantize11->type << " " << fakeQuantize11->name <<
                " was not quantized " << fakeQuantize11->outData[0]->getPrecision();
        }

        const CNNLayerPtr pooling12 = getLayer(network, "Pooling16");
        if ((pooling12->outData[0]->getPrecision() != Precision::U8) && (pooling12->outData[0]->getPrecision() != Precision::I8)) {
            THROW_IE_EXCEPTION <<
                "layer " << pooling12->type << " " << pooling12->name <<
                " was not quantized " << pooling12->outData[0]->getPrecision();
        }

        const CNNLayerPtr pooling16 = getLayer(network, "Pooling16");
        if ((pooling16->outData[0]->getPrecision() != Precision::U8) && (pooling16->outData[0]->getPrecision() != Precision::I8)) {
            THROW_IE_EXCEPTION <<
                "layer " << pooling16->type << " " << pooling16->name <<
                " was not quantized " << pooling16->outData[0]->getPrecision();
        }

        if (operation == "sum") {
            const CNNLayerPtr eltwise = getLayer(network, "Eltwise17_original");
            if (eltwise->type != "Eltwise") {
                THROW_IE_EXCEPTION << "layer type " << eltwise->type << " " << eltwise->name << " is not correct";
            }

            if ((eltwise->outData[0]->getPrecision() != Precision::FP32) && (eltwise->outData[0]->getPrecision() != Precision::FP16)) {
                THROW_IE_EXCEPTION << "layer " << eltwise->type << " " << eltwise->name << " output port precision is not correct";
            }

            const CNNLayerPtr dequantizationScaleShift = getLayer(network, "Eltwise17");
            if (dequantizationScaleShift == nullptr) {
                THROW_IE_EXCEPTION << "dequantization layer was not found";
            }

            Blob::Ptr shiftsBlob = CNNNetworkHelper::getBlob(dequantizationScaleShift, "biases");
            const auto shiftsBuffer = CNNNetworkHelper::getFloatData(shiftsBlob);
            const size_t shiftsBlobSize = shiftsBlob->size();
            for (size_t i = 0; i < shiftsBlobSize; ++i) {
                if (shiftsBuffer.get()[i] != 0.f) {
                    THROW_IE_EXCEPTION << "unexpected shift value " << shiftsBuffer.get()[i] << " for dequantization layer";
                }
            }
        } else if ((operation == "mul") || (operation == "prod")) {
            const CNNLayerPtr eltwise = getLayer(network, "Eltwise17");
            if (eltwise->type != "Eltwise") {
                THROW_IE_EXCEPTION << "layer type " << eltwise->type << " " << eltwise->name << " is not correct";
            }

            const CNNLayerPtr dequantizationScaleShift = getLayer(network, "Eltwise17_original");
            if (dequantizationScaleShift != nullptr) {
                THROW_IE_EXCEPTION
                    << "dequantization layer " << dequantizationScaleShift->type << " " << dequantizationScaleShift->name
                    << " has to be absent (moved to full path branch)";
            }
        }
    } else {
        const CNNLayerPtr eltwise = getLayer(network, "Eltwise17");
        if (eltwise->type != "Eltwise") {
            THROW_IE_EXCEPTION << "layer type " << eltwise->type << " " << eltwise->name << " is not correct";
        }

        if ((eltwise->outData[0]->getPrecision() != Precision::FP32) && (eltwise->outData[0]->getPrecision() != Precision::FP16)) {
            THROW_IE_EXCEPTION << "layer " << eltwise->type << " " << eltwise->name << " output port precision is not correct";
        }
    }

    // FP32 way
    const CNNLayerPtr fakeQuantize6 = getLayer(network, "FakeQuantize6");
    if ((fakeQuantize6->outData[0]->getPrecision() != Precision::FP32) && (fakeQuantize6->outData[0]->getPrecision() != Precision::FP16)) {
        THROW_IE_EXCEPTION << "layer " << fakeQuantize6->type << " " << fakeQuantize6->name << " was quantized";
    }


    return true;
}

void EltwiseWithPoolingTestModel::resetTransformation(CNNNetwork& network) const {
    const float low = signedIntervals ? -128 : 0.f;
    const float high = signedIntervals ? 127 : 255.f;

    fillData(getLayer(network, "Const2"), low / 4.f, "custom");
    fillData(getLayer(network, "Const3"), high / 4.f, "custom");
    fillData(getLayer(network, "Const4"), low / 4.f, "custom");
    fillData(getLayer(network, "Const5"), high / 4.f, "custom");

    fillData(getLayer(network, "Const7"), low / 2.f, "custom");
    fillData(getLayer(network, "Const8"), high / 2.f, "custom");
    fillData(getLayer(network, "Const9"), low / 2.f, "custom");
    fillData(getLayer(network, "Const10"), high / 2.f, "custom");
}
