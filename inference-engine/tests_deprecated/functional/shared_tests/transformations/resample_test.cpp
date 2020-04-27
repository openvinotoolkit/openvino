// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformer_single_layer_tests.hpp"

std::string ResampleTestModel::getModel(SingleLayerTransformationsTestParams& p) const {
    size_t type_size = sizeof(InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type);
    if (p._network_precision == "FP16")
        type_size = sizeof(InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP16>::value_type);

    CommonTestUtils::conv_common_params conv = { {1, 1}, {3, 3}, {0, 0}, {0, 0}, {1, 1}, "valid", 1, 32, false, false };
    std::vector<size_t> convOutShape(p.inputDimensions[0].size());
    getConvOutShape(p.inputDimensions[0], conv, convOutShape);

    std::vector<size_t> weightsConstInputDims = { 32lu, 32lu, 3lu, 3lu };
    std::vector<size_t> biasesConvolutionConstDims = { conv.out_c };
    std::map<std::string, std::string> const_params = {};
    std::map<std::string, std::string> fake_quantize_params = {
        {"levels", "256"}
    };
    std::map<std::string, std::string> power_params = {
        {"power", "1"}, {"scale", "1"}, {"shift", "0"}
    };

    std::map<std::string, std::string> resampleParams = {
        {"antialias", "0"}, {"factor", "2"}, {"type", "caffe.ResampleParameter.NEAREST"}
    };

    std::vector<std::pair<std::string, std::string>> edges = {
        {"0,0", "5,5"}, // Power
        {"1,1", "5,6"}, {"2,2", "5,7"}, {"3,3", "5,8"}, {"4,4", "5,9"}, // Const layers
        {"5,10", "6,11"}
    };

    return CommonTestUtils::DefaultNetBuilder::buildNetworkWithOneInput("QuantizationOnWeights", p.inputDimensions[0], p._network_precision)
        // 1
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        // 2
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        // 3
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        // 4
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        // 5
        .addLayer("FakeQuantize", p._network_precision, &fake_quantize_params, {{p.inputDimensions[0], {1}, {1}, {1}, {1}}, {{p.inputDimensions[0]}}})
        // 6
        .addLayer("Resample", p._network_precision, &resampleParams, {{p.inputDimensions[0]}, {{p.inputDimensions[0]}}})
        .finish(&edges);
}

std::string ResampleTestModel::getName() const {
    return "ResampleTestModel";
}

bool ResampleTestModel::transform(CNNNetwork& network, LayerTransformation::Params& params) const {
    LowPrecisionTransformer transformer(LowPrecisionTransformer::getAllTransformations(params));
    transformer.transform(network);
    return true;
}

void ResampleTestModel::resetTransformation(CNNNetwork& network) const {
    fillData(getLayer(network, "Const1"), -128.0 / 20.0, "custom");
    fillData(getLayer(network, "Const2"), 127.0 / 20.0, "custom");
    fillData(getLayer(network, "Const3"), -128.0 / 20.0, "custom");
    fillData(getLayer(network, "Const4"), 127.0 / 20.0, "custom");
}
