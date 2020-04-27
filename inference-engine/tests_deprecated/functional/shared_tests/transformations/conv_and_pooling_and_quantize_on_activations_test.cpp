// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformer_single_layer_tests.hpp"

std::string ConvolutionAndPoolingAndQuantizeOnActivationsTestModel::getModel(SingleLayerTransformationsTestParams& p) const {
    size_t type_size = sizeof(InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type);
    if (p._network_precision == "FP16")
        type_size = sizeof(InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP16>::value_type);

    CommonTestUtils::pool_common_params pooling =
            { {2, 2}, {3, 3}, {0, 0}, {0, 0}, "valid", false, true };
    std::vector<size_t> poolOutShape(p.inputDimensions[0].size());
    getPoolOutShape(p.inputDimensions[0], pooling, poolOutShape);

    CommonTestUtils::conv_common_params conv =
            { {1, 1}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, "valid", 1, 80, true, true };
    std::vector<size_t> convOutShape(poolOutShape.size());
    getConvOutShape(poolOutShape, conv, convOutShape);

    std::map<std::string, std::string> power_params = {
        {"power", "1"}, {"scale", "1"}, {"shift", "0"}
    };
    std::map<std::string, std::string> const_params = {};
    std::map<std::string, std::string> fake_quantize_params = {
        {"levels", "256"}
    };

    std::vector<std::pair<std::string, std::string>> edges = {
        {"0,0", "5,5"},  // FQ
        {"1,1", "5,6"}, {"2,2", "5,7"}, {"3,3", "5,8"}, {"4,4", "5,9"}, // const
        {"5,10", "6,11"}, {"6,12", "7,13"} // Pool, Conv
    };

    return CommonTestUtils::DefaultNetBuilder::buildNetworkWithOneInput(
            "Conv_ScaleShift_transformations", p.inputDimensions[0], p._network_precision)
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        .addLayer("FakeQuantize", p._network_precision, &fake_quantize_params, {{p.inputDimensions[0], {1}, {1}, {1}, {1}}, {{p.inputDimensions[0]}}})
        .poolingLayer(p._network_precision, {{p.inputDimensions[0]}, {poolOutShape}}, pooling)
        .convolutionLayer(p._network_precision, {{poolOutShape}, {convOutShape}}, conv)
        .finish(&edges);
}

std::string ConvolutionAndPoolingAndQuantizeOnActivationsTestModel::getName() const {
    return "ConvolutionAndPoolingAndQuantizeOnActivationsTestModel";
}

bool ConvolutionAndPoolingAndQuantizeOnActivationsTestModel::transform(CNNNetwork& network, LayerTransformation::Params& params) const {
    LowPrecisionTransformer transformer = getLowPrecisionTransformer(params);
    transformer.transform(network);
    return true;
}

void ConvolutionAndPoolingAndQuantizeOnActivationsTestModel::resetTransformation(CNNNetwork& network) const {
    fillData(getLayer(network, "Const1"), -128.f / 20.f, "custom");
    fillData(getLayer(network, "Const2"), 127.f / 20.f, "custom");
    fillData(getLayer(network, "Const3"), -128.f / 20.f, "custom");
    fillData(getLayer(network, "Const4"), 127.f / 20.f, "custom");
    fillDataWithInitValue(getLayer(network, "Convolution7"), "weights", 1.234f);
    fillDataWithInitValue(getLayer(network, "Convolution7"), "biases", 5.678f);
}
