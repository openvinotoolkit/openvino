// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformer_single_layer_tests.hpp"

std::string ConvolutionAndDequantizationScaleShiftsOnActivationsTestModel::getModel(SingleLayerTransformationsTestParams& p) const {
    size_t type_size = sizeof(InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type);
    if (p._network_precision == "FP16")
        type_size = sizeof(InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP16>::value_type);

    std::map<std::string, std::string> scale_shift_params = {};
    CommonTestUtils::conv_common_params conv =
            { {1, 1}, {3, 3}, {0, 0}, {0, 0}, {1, 1}, "", 1, 32, true, true };
    std::vector<size_t> convOutShape(p.inputDimensions[0].size());
    getConvOutShape(p.inputDimensions[0], conv, convOutShape);

    std::map<std::string, std::string> power_params = {
        {"power", "1"}, {"scale", "1"}, {"shift", "0"}
    };
    std::vector<std::pair<std::string, std::string>> edges = {
        {"0,0", "1,1"}, {"1,2", "2,3"}, {"2,4", "3,5"}
    };

    return CommonTestUtils::DefaultNetBuilder::buildNetworkWithOneInput(
            "Conv_ScaleShift_transformations", p.inputDimensions[0], p._network_precision)
        .addLayer("ScaleShift", p._network_precision, &scale_shift_params, {{p.inputDimensions[0]}, {p.inputDimensions[0]}}, p.inputDimensions[0][1] * type_size, p.inputDimensions[0][1] * type_size)
        .convolutionLayer(p._network_precision, {{p.inputDimensions[0]}, {convOutShape}}, conv)
        .addLayer("Power", p._network_precision, &power_params, {{convOutShape}, {convOutShape}})
        .finish(&edges);
}

std::string ConvolutionAndDequantizationScaleShiftsOnActivationsTestModel::getName() const {
    return "ConvolutionAndDequantizationScaleShiftsOnActivationsTestModel";
}

bool ConvolutionAndDequantizationScaleShiftsOnActivationsTestModel::transform(CNNNetwork& network, LayerTransformation::Params& params) const {
    LowPrecisionTransformer transformer = getLowPrecisionTransformer(params);
    transformer.transform(network);
    return true;
}

void ConvolutionAndDequantizationScaleShiftsOnActivationsTestModel::resetTransformation(CNNNetwork& network) const {
    fillData(getLayer(network, "ScaleShift1"), 3.f, "weights");
    fillData(getLayer(network, "ScaleShift1"), 4.f, "biases");

    fillDataWithInitValue(getLayer(network, "Convolution2"), "weights", 1.234f);
    fillDataWithInitValue(getLayer(network, "Convolution2"), "biases", 5.678f);
}
