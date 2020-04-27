// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformer_single_layer_tests.hpp"

std::string PoolingTestModel::getModel(SingleLayerTransformationsTestParams& p) const {
    size_t type_size = sizeof(InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type);
    if (p._network_precision == "FP16")
        type_size = sizeof(InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP16>::value_type);

    CommonTestUtils::pool_common_params pooling = { {1, 1}, {1, 1}, {0, 0}, {0, 0}, "valid", false, true };
    std::vector<size_t> poolOutShape(p.inputDimensions[0].size());
    getPoolOutShape(p.inputDimensions[0], pooling, poolOutShape);

    std::map<std::string, std::string> power_params = {
        {"power", "1"}, {"scale", "1"}, {"shift", "0"}
    };
    std::map<std::string, std::string> const_params = {};
    std::map<std::string, std::string> fake_quantize_params = {
        {"levels", "256"}
    };

    std::vector<std::pair<std::string, std::string>> edges = {
        {"0,0", "1,1"}, {"1,2", "6,7"}, // power
        {"2,3", "6,8"}, {"3,4", "6,9"}, {"4,5", "6,10"}, {"5,6", "6,11"}, // const
        {"6,12", "7,13"}, {"7,14", "8,15"} // pool, power
    };

    return CommonTestUtils::DefaultNetBuilder::buildNetworkWithOneInput(
            "Conv_ScaleShift_transformations", p.inputDimensions[0], p._network_precision)
        .addLayer("Power", p._network_precision, &power_params, {{p.inputDimensions[0]}, {p.inputDimensions[0]}})
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        .addLayer("FakeQuantize", p._network_precision, &fake_quantize_params, {{p.inputDimensions[0], {1}, {1}, {1}, {1}}, {{p.inputDimensions[0]}}})
        .poolingLayer(p._network_precision, {{p.inputDimensions[0]}, {poolOutShape}}, pooling)
        .addLayer("Power", p._network_precision, &power_params, {{poolOutShape}, {poolOutShape}})
        .finish(&edges);
}

void PoolingTestModel::resetTransformation(CNNNetwork& network) const {
    fillData(getLayer(network, "Const2"), 63.5, "custom");
    fillData(getLayer(network, "Const3"), 127.0, "custom");
    fillData(getLayer(network, "Const4"), 63.5, "custom");
    fillData(getLayer(network, "Const5"), 127.0, "custom");
}

std::string PoolingTestModel::getName() const {
    return "PoolingTestModel";
}

bool PoolingTestModel::transform(CNNNetwork& network, LayerTransformation::Params& params) const {
    LowPrecisionTransformer transformer(LowPrecisionTransformer::getAllTransformations(params));
    transformer.transform(network);

    const Precision precision = params.updatePrecisions ? Precision(Precision::U8) : network.getPrecision();

    CNNLayerPtr fakeQuantize = getLayer(network, "FakeQuantize6");
    if (fakeQuantize->outData[0]->getPrecision() != precision) {
        THROW_IE_EXCEPTION << fakeQuantize->name << " precision " << precision << " is not correct";
    }

    CNNLayerPtr pooling = getLayer(network, "Pooling7");
    if (pooling->outData[0]->getPrecision() != precision) {
        THROW_IE_EXCEPTION << pooling->name << " precision " << precision << " is not correct";
    }

    return true;
}
