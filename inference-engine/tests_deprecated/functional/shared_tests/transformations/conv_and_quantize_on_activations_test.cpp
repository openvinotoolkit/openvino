// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformer_single_layer_tests.hpp"

std::string ConvolutionAndQuantizeOnActivationsTestModel::getModel(SingleLayerTransformationsTestParams& p) const {
    size_t type_size = sizeof(InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type);
    if (p._network_precision == "FP16")
        type_size = sizeof(InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP16>::value_type);

    CommonTestUtils::conv_common_params conv =
            { {2, 2}, {3, 3}, {0, 0}, {0, 0}, {1, 1}, "", 1, 32, true, true };
    std::vector<size_t> convOutShape(p.inputDimensions[0].size());
    getConvOutShape(p.inputDimensions[0], conv, convOutShape);

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
        {"5,10", "6,11"}, {"6,12", "7,13"} // Pool, Conv, power
    };

    return CommonTestUtils::DefaultNetBuilder::buildNetworkWithOneInput(
            "Conv_ScaleShift_transformations", p.inputDimensions[0], p._network_precision)
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        .addLayer("FakeQuantize", p._network_precision, &fake_quantize_params, {{p.inputDimensions[0], {1}, {1}, {1}, {1}}, {{p.inputDimensions[0]}}})
        .convolutionLayer(p._network_precision, {{p.inputDimensions[0]}, {convOutShape}}, conv)
        .addLayer("Power", p._network_precision, &power_params, {{convOutShape}, {convOutShape}})
        .finish(&edges);
}

std::string ConvolutionAndQuantizeOnActivationsTestModel::getName() const {
    return "ConvolutionAndQuantizeOnActivationsTestModel";
}

bool ConvolutionAndQuantizeOnActivationsTestModel::transform(CNNNetwork& network, LayerTransformation::Params& params) const {
    LowPrecisionTransformer transformer = getLowPrecisionTransformer(params);
    transformer.transform(network);
    return true;
}

void ConvolutionAndQuantizeOnActivationsTestModel::resetTransformation(CNNNetwork& network) const {
    fillData(getLayer(network, "Const1"), -128.0 / 20.0, "custom");
    fillData(getLayer(network, "Const2"), 127.0 / 20.0, "custom");
    fillData(getLayer(network, "Const3"), -128.0 / 20.0, "custom");
    fillData(getLayer(network, "Const4"), 127.0 / 20.0, "custom");
    fillDataWithInitValue(getLayer(network, "Convolution6"), "weights", 1.234);
    fillDataWithInitValue(getLayer(network, "Convolution6"), "biases", 5.678);
}
