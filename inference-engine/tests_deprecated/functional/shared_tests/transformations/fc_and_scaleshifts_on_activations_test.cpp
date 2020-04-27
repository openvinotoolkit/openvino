// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformer_single_layer_tests.hpp"

std::string FullyConnectedAndScaleShiftsOnActivationsTestModel::getModel(SingleLayerTransformationsTestParams& p) const {
    size_t type_size = sizeof(InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type);
    if (p._network_precision == "FP16")
        type_size = sizeof(InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP16>::value_type);

    std::vector<size_t> const_1_dims = {1000, 2048};
    std::vector<size_t> const_2_dims = {1000};
    std::map<std::string, std::string> scale_shift_params = {};
    std::map<std::string, std::string> const_params = {};
    std::map<std::string, std::string> fc_params = {
        { "out-size", "1000" }
    };

    std::vector<std::pair<std::string, std::string>> edges = {
        {"0,0", "1,1"}, {"1,2", "4,5"}, // ScaleShift
        {"2,3", "4,6"}, {"3,4", "4,7"}, // Const layers
    };

    return CommonTestUtils::DefaultNetBuilder::buildNetworkWithOneInput(
            "FCandScaleShift", p.inputDimensions[0], p._network_precision)
        .addLayer("ScaleShift", p._network_precision, &scale_shift_params, {{p.inputDimensions[0]}, {p.inputDimensions[0]}}, p.inputDimensions[0][1] * type_size, p.inputDimensions[0][1] * type_size)
        .addLayer("Const", p._network_precision, &const_params, {{}, {const_1_dims}},
                std::accumulate(const_1_dims.begin(), const_1_dims.end(), 1lu, std::multiplies<size_t>()) * type_size, 0)
        .addLayer("Const", p._network_precision, &const_params, {{}, {const_2_dims}},
                std::accumulate(const_2_dims.begin(), const_2_dims.end(), 1lu, std::multiplies<size_t>()) * type_size, 0)
        .addLayer("FullyConnected", p._network_precision, &fc_params, {{p.inputDimensions[0], const_1_dims, const_2_dims}, {{1, 1000}}})
        .finish(&edges);
}

std::string FullyConnectedAndScaleShiftsOnActivationsTestModel::getName() const {
    return "FullyConnectedAndScaleShiftsOnActivationsTestModel";
}

bool FullyConnectedAndScaleShiftsOnActivationsTestModel::transform(CNNNetwork& network, LayerTransformation::Params& params) const {
    LowPrecisionTransformer transformer = getLowPrecisionTransformer(params);
    transformer.transform(network);
    return true;
}

void FullyConnectedAndScaleShiftsOnActivationsTestModel::resetTransformation(CNNNetwork& network) const {
    fillData(getLayer(network, "ScaleShift1"), 0.4f, "weights");
    fillData(getLayer(network, "ScaleShift1"), 0.3f, "biases");

    fillDataWithInitValue(getLayer(network, "Const2"), "custom", 0.2f);
    fillDataWithInitValue(getLayer(network, "Const3"), "custom", 0.3f);
}
