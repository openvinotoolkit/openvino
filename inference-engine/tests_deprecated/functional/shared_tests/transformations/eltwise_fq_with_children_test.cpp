// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformer_single_layer_tests.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

std::string EltwiseFqWithChildrenTestModel::getModel(SingleLayerTransformationsTestParams& p) const {
    size_t type_size = sizeof(PrecisionTrait<Precision::FP32>::value_type);
    if (p._network_precision == "FP16")
        type_size = sizeof(PrecisionTrait<Precision::FP16>::value_type);

    std::map<std::string, std::string> constParams = {};
    std::map<std::string, std::string> fakeQuantizeParams = { {"levels", "256"} };
    std::map<std::string, std::string> eltwiseParams = { {"operation", operation} };
    std::map<std::string, std::string> poolingParams = { {"kernel", "1,1"}, {"pool-method", "max"}, {"exclude-pad", "false"} };

    std::vector<std::pair<std::string, std::string>> edges = {
        {"0,0", "5,5"}, {"5,10", "12,24"}, // Inputs
        {"1,1", "5,6"}, {"2,2", "5,7"}, {"3,3", "5,8"}, {"4,4", "5,9"}, // Const layers
        {"6,11", "10,16"}, {"7,12", "10,17"}, {"8,13", "10,18"}, {"9,14", "10,19"}, // Const layers
        {"5,10", "11,21"}, {"10,20", "11,22"}, // Fake quantize to Eltwise
        {"12,25", "10,15"},
    };

    return CommonTestUtils::DefaultNetBuilder::buildNetworkWithOneInput("EltwiseTestModel", p.inputDimensions[0], p._network_precision)
        // 1
        .addLayer("Const", p._network_precision, &constParams, {{}, {{1}}}, type_size, 0)
        // 2
        .addLayer("Const", p._network_precision, &constParams, {{}, {{1}}}, type_size, 0)
        // 3
        .addLayer("Const", p._network_precision, &constParams, {{}, {{1}}}, type_size, 0)
        // 4
        .addLayer("Const", p._network_precision, &constParams, {{}, {{1}}}, type_size, 0)
        // 5
        .addLayer("FakeQuantize", p._network_precision, &fakeQuantizeParams, {{p.inputDimensions[0], {1}, {1}, {1}, {1}}, {{p.inputDimensions[0]}}}, "fakeQuantize1")
        // 6
        .addLayer("Const", p._network_precision, &constParams, {{}, {{1}}}, type_size, 0)
        // 7
        .addLayer("Const", p._network_precision, &constParams, {{}, {{1}}}, type_size, 0)
        // 8
        .addLayer("Const", p._network_precision, &constParams, {{}, {{1}}}, type_size, 0)
        // 9
        .addLayer("Const", p._network_precision, &constParams, {{}, {{1}}}, type_size, 0)
        // 10
        .addLayer("FakeQuantize", p._network_precision, &fakeQuantizeParams, {{p.inputDimensions[0], {1}, {1}, {1}, {1}}, {{p.inputDimensions[0]}}}, "fakeQuantize2")
        // 11
        .addLayer("Eltwise", p._network_precision, &eltwiseParams, {{p.inputDimensions[0], p.inputDimensions[0]}, {{p.inputDimensions[0]}}}, 0, "eltwise")

        // 12
        .addLayer("Pooling", p._network_precision, &poolingParams, {p.inputDimensions, {p.inputDimensions}}, 0, "pooling")
        .finish(&edges);
}

std::string EltwiseFqWithChildrenTestModel::getName() const {
    return std::string("EltwiseFqWithChildrenTestModel") +
        (cpuSpecific ? "_cpuSpecific" : "") +
        "_" + operation +
        (signedIntervals ? "_signedInterval" : "_notsignedInterval") +
        (minLevels != 2ul ? ("_minLevels" + std::to_string(minLevels)) : "");
}

bool EltwiseFqWithChildrenTestModel::transform(CNNNetwork& network, LayerTransformation::Params& params) const {
    params.updatePrecisions = true;
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
            const CNNLayerPtr eltwise = getLayer(network, "eltwise");
            if (eltwise->type != "Eltwise") {
                THROW_IE_EXCEPTION << "layer " << eltwise->type << " " << eltwise->name << " was quantized";
            }
        }

        if (params.updatePrecisions) {
            {
                const CNNLayerPtr fakeQuantize1 = getLayer(network, "fakeQuantize1");
                const Precision defaultPrecision = signedIntervals ? Precision::I8 : Precision::U8;
                const Precision expectedPrecision = params.precisionsOnActivations.size() == 1 ? params.precisionsOnActivations[0] : defaultPrecision;
                if (fakeQuantize1->outData[0]->getPrecision() != expectedPrecision) {
                    THROW_IE_EXCEPTION << "unexpected precision " << fakeQuantize1->outData[0]->getPrecision() << " for " << fakeQuantize1->type << " " << fakeQuantize1->name;
                }
            }

            {
                const CNNLayerPtr fakeQuantize2 = getLayer(network, "fakeQuantize2");
                const CNNLayerPtr input = getLayer(network, "Input0");
                const Precision originalPrecision = input->outData[0]->getTensorDesc().getPrecision();
                if (fakeQuantize2->outData[0]->getPrecision() != originalPrecision) {
                    THROW_IE_EXCEPTION << "unexpected precision " << fakeQuantize2->outData[0]->getPrecision() << " for " << fakeQuantize2->type << " " << fakeQuantize2->name;
                }
            }
        }
    }
    return true;
}

void EltwiseFqWithChildrenTestModel::resetTransformation(CNNNetwork& network) const {
    const float low = signedIntervals ? -128 : 0.f;
    const float high = signedIntervals ? 127 : 255.f;

    fillData(getLayer(network, "Const1"), low / 4.f, "custom");
    fillData(getLayer(network, "Const2"), high / 4.f, "custom");
    fillData(getLayer(network, "Const3"), low / 4.f, "custom");
    fillData(getLayer(network, "Const4"), high / 4.f, "custom");

    fillData(getLayer(network, "Const6"), low / 2.f, "custom");
    fillData(getLayer(network, "Const7"), high / 2.f, "custom");
    fillData(getLayer(network, "Const8"), low / 2.f, "custom");
    fillData(getLayer(network, "Const9"), high / 2.f, "custom");
}
