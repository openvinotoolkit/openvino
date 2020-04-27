// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformer_single_layer_tests.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

std::string EltwiseTestModel::getModel(SingleLayerTransformationsTestParams& p) const {
    size_t type_size = sizeof(PrecisionTrait<Precision::FP32>::value_type);
    if (p._network_precision == "FP16")
        type_size = sizeof(PrecisionTrait<Precision::FP16>::value_type);

    std::map<std::string, std::string> const_params = {};
    std::map<std::string, std::string> fake_quantize_params = { {"levels", "256"} };
    std::map<std::string, std::string> eltwise_params = { {"operation", operation} };
    std::map<std::string, std::string> power_params = { {"power", "1"}, {"scale", "1"}, {"shift", "0"} };

    std::vector<std::pair<std::string, std::string>> edges = {
        {"0,0", "6,6"}, {"1,1", "11,16"}, // Inputs
        {"2,2", "6,7"}, {"3,3", "6,8"}, {"4,4", "6,9"}, {"5,5", "6,10"}, // Const layers
        {"7,12", "11,17"}, {"8,13", "11,18"}, {"9,14", "11,19"}, {"10,15", "11,20"}, // Const layers
        {"6,11", "12,22"}, {"11,21", "12,23"} // Fake quantize to Convolution
    };

    return CommonTestUtils::DefaultNetBuilder::buildNetworkWithOneInput("EltwiseTestModel", p.inputDimensions[0], p._network_precision)
        .addInputLayer(p._network_precision, p.inputDimensions[1])
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        .addLayer("FakeQuantize", p._network_precision, &fake_quantize_params, {{p.inputDimensions[0], {1}, {1}, {1}, {1}}, {{p.inputDimensions[0]}}})
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        .addLayer("FakeQuantize", p._network_precision, &fake_quantize_params, {{p.inputDimensions[1], {1}, {1}, {1}, {1}}, {{p.inputDimensions[1]}}})
        .addLayer("Eltwise", p._network_precision, &eltwise_params, {{p.inputDimensions[0], p.inputDimensions[1]}, {{p.inputDimensions[0]}}}, 0, 0)
        .finish(&edges);
}

std::string EltwiseTestModel::getName() const {
    return std::string("EltwiseTestModel") +
        (cpuSpecific ? "_cpuSpecific" : "") +
        "_" + operation +
        (signedIntervals ? "_signedInterval" : "_notsignedInterval") +
        (minLevels != 2ul ? ("_minLevels" + std::to_string(minLevels)) : "");
}

bool EltwiseTestModel::transform(CNNNetwork& network, LayerTransformation::Params& params) const {
    LowPrecisionTransformations transformations = getLowPrecisionTransformations(params);
    if (!cpuSpecific) {
        THROW_IE_EXCEPTION << "not CPU/GPU specific Eltwise is not supported";
    }

    LayerTransformationPtr eltwiseTransformation = transformations.find("Eltwise");
    eltwiseTransformation->setMinQuantizationLevels(minLevels);

    LowPrecisionTransformer transformer(transformations);
    transformer.transform(network);

    if (params.quantizeOutputs) {
        if ((params.quantizedTensorAlignmentOnActivations == LayerTransformation::QuantizedTensorAlignment::UpdateLevel) && (minLevels != 2ul)) {
            const CNNLayerPtr eltwise = getLayer(network, "Eltwise12");
            if (eltwise->type != "Eltwise") {
                THROW_IE_EXCEPTION << "layer " << eltwise->type << " " << eltwise->name << " was quantized";
            }
        }

        if (params.updatePrecisions) {
            const CNNLayerPtr fakeQuantize1 = getLayer(network, "FakeQuantize6");
            const CNNLayerPtr fakeQuantize2 = getLayer(network, "FakeQuantize11");

            const Precision expectedPrecision = signedIntervals ? Precision::I8 : Precision::U8;
            if (fakeQuantize1->outData[0]->getPrecision() != expectedPrecision) {
                THROW_IE_EXCEPTION << "unexpected precision " << fakeQuantize1->outData[0]->getPrecision() << " for " << fakeQuantize1->type << " " << fakeQuantize1->name;
            }
            if (fakeQuantize2->outData[0]->getPrecision() != expectedPrecision) {
                THROW_IE_EXCEPTION << "unexpected precision " << fakeQuantize2->outData[0]->getPrecision() << " for " << fakeQuantize2->type << " " << fakeQuantize2->name;
            }
        }
    }
    return true;
}

void EltwiseTestModel::resetTransformation(CNNNetwork& network) const {
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
