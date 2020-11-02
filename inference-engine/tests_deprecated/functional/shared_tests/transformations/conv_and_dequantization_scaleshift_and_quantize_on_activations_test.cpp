// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformer_single_layer_tests.hpp"

std::string ConvolutionAndDequantizationScaleShiftAndQuantizeOnActivationsTestModel::getModel(SingleLayerTransformationsTestParams& p) const {
    size_t type_size = sizeof(float);
    if (p._network_precision == "FP16")
        type_size = sizeof(short);

    CommonTestUtils::conv_common_params conv =
            { {2, 2}, {3, 3}, {0, 0}, {0, 0}, {1, 1}, "", 1, 32, true, true };
    std::vector<size_t> convOutShape(p.inputDimensions[0].size());
    getConvOutShape(p.inputDimensions[0], conv, convOutShape);

    std::map<std::string, std::string> const_params = {};
    std::map<std::string, std::string> fake_quantize_params = {
        {"levels", "256"}
    };
    std::map<std::string, std::string> power_params = {
        {"power", "1"}, {"scale", "1"}, {"shift", "0"}
    };

    std::vector<std::pair<std::string, std::string>> edges = {
        {"0,0", "1,1"}, {"1,2", "6,7"},
        {"2,3", "6,8"}, {"3,4", "6,9"}, {"4,5", "6,10"}, {"5,6", "6,11"}, // Const layers
        {"6,12", "7,13"}, // Fake quantize to Convolution
        {"7,14", "8,15"} // Convolution to Power
    };

    return CommonTestUtils::DefaultNetBuilder::buildNetworkWithOneInput(
            "dequantizeScaleShift_", p.inputDimensions[0], p._network_precision)
        .addLayer("ScaleShift", p._network_precision, &const_params, {{p.inputDimensions[0]}, {p.inputDimensions[0]}}, p.inputDimensions[0][1] * type_size, p.inputDimensions[0][1] * type_size)
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        .addLayer("Const", p._network_precision, &const_params, {{}, {{1}}}, type_size, 0)
        .addLayer("FakeQuantize", p._network_precision, &fake_quantize_params, {{p.inputDimensions[0], {1}, {1}, {1}, {1}}, {{p.inputDimensions[0]}}})
        .convolutionLayer(p._network_precision, {{p.inputDimensions[0]}, {convOutShape}}, conv)
        .addLayer("Power", p._network_precision, &power_params, {{convOutShape}, {convOutShape}})
        .finish(&edges);
}

std::string ConvolutionAndDequantizationScaleShiftAndQuantizeOnActivationsTestModel::getName() const {
    return "ConvolutionAndDequantizationScaleShiftAndQuantizeOnActivationsTestModel";
}

bool ConvolutionAndDequantizationScaleShiftAndQuantizeOnActivationsTestModel::transform(CNNNetwork& network, LayerTransformation::Params& params) const {
    LowPrecisionTransformer transformer = getLowPrecisionTransformer(params);
    transformer.transform(network);
    return true;
}

void ConvolutionAndDequantizationScaleShiftAndQuantizeOnActivationsTestModel::resetTransformation(CNNNetwork& network) const {
    fillData(getLayer(network, "ScaleShift1"), 3, "weights");
    fillData(getLayer(network, "ScaleShift1"), 5, "biases");
    fillData(getLayer(network, "Const2"), -128.0, "custom");
    fillData(getLayer(network, "Const3"), 127.0, "custom");
    fillData(getLayer(network, "Const4"), -128.0, "custom");
    fillData(getLayer(network, "Const5"), 127.0, "custom");
    fillDataWithInitValue(getLayer(network, "Convolution7"), "weights", 1.234);
    fillDataWithInitValue(getLayer(network, "Convolution7"), "biases", 5.678);
}
