// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformer_single_layer_tests.hpp"
#include "low_precision_transformations/eltwise.hpp"
#include "low_precision_transformations/concat_multi_channels.hpp"

std::string ConcatMultiChannelTestModel::getModel(SingleLayerTransformationsTestParams& p) const {
//    ASSERT_EQ(2, p.inputDimensions.size());
    size_t type_size = sizeof(InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type);
    if (p._network_precision == "FP16")
        type_size = sizeof(InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP16>::value_type);

    const size_t axis = 1; // should be passed in 'p' argument

    std::vector<size_t> concat_out_dims = p.inputDimensions[0];
    concat_out_dims[axis] += p.inputDimensions[1][axis];

    std::map<std::string, std::string> const_params = {};
    std::map<std::string, std::string> fake_quantize_params = {
        {"levels", "256"}
    };
    std::map<std::string, std::string> concat_params = {
        {"axis", "1"}
    };
    std::map<std::string, std::string> power_params = {
        {"power", "1"}, {"scale", "1"}, {"shift", "0"}
    };

    std::vector<std::pair<std::string, std::string>> edges = {
        {"0,0", "10,10"}, {"1,1", "11,16"}, // Inputs to FakeQuantize
        {"2,2", "10,11"}, {"3,3", "10,12"}, {"4,4", "10,13"}, {"5,5", "10,14"}, // Const layers
        {"6,6", "11,17"}, {"7,7", "11,18"}, {"8,8", "11,19"}, {"9,9", "11,20"}, // Const layers
        {"10,15", "12,22"}, {"11,21", "12,23"} // FakeQuantize to Concat
    };

    return CommonTestUtils::DefaultNetBuilder::buildNetworkWithOneInput(
            "Concat_transformations_", p.inputDimensions[0], p._network_precision)
        .addInputLayer(p._network_precision, p.inputDimensions[1])
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        .addLayer("FakeQuantize", p._network_precision, &fake_quantize_params, {{p.inputDimensions[0], {1}, {1}, {1}, {1}}, {{p.inputDimensions[0]}}})
        .addLayer("FakeQuantize", p._network_precision, &fake_quantize_params, {{p.inputDimensions[1], {1}, {1}, {1}, {1}}, {{p.inputDimensions[1]}}})
        .addLayer("Concat", p._network_precision, &concat_params, { {p.inputDimensions[0], p.inputDimensions[1]}, { concat_out_dims }})
        .finish(&edges);
}

std::string ConcatMultiChannelTestModel::getName() const {
    return "ConcatMultiChannelTestModel";
}

bool ConcatMultiChannelTestModel::transform(CNNNetwork& network, LayerTransformation::Params& params) const {
    LowPrecisionTransformer transformer(LowPrecisionTransformer::getAllTransformations(params).
        addBranchSpecific<ConcatMultiChannelsTransformation>(params, "Concat")
    );
    transformer.transform(network);
    return true;
}

void ConcatMultiChannelTestModel::resetTransformation(CNNNetwork& network) const {
    fillData(getLayer(network, "Const2"), 0.0, "custom");
    fillData(getLayer(network, "Const3"), 255.0 / 10.0, "custom");
    fillData(getLayer(network, "Const4"), 0.0, "custom");
    fillData(getLayer(network, "Const5"), 255.0 / 10.0, "custom");

    fillData(getLayer(network, "Const6"), -255.0 / 400.0, "custom");
    fillData(getLayer(network, "Const7"), 255.0 / 200.0, "custom");
    fillData(getLayer(network, "Const8"), -255.0 / 400.0, "custom");
    fillData(getLayer(network, "Const9"), 255.0 / 200.0, "custom");
}
