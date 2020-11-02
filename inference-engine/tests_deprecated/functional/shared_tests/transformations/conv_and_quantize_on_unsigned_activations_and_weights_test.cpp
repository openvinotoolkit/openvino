// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformer_single_layer_tests.hpp"

void ConvolutionAndQuantizeOnUnsignedActivationsAndWeightsTestModel::resetTransformation(CNNNetwork& network) const {
    fillData(getLayer(network, "Const2"), 63.5f, "custom");
    fillData(getLayer(network, "Const3"), 127.f, "custom");
    fillData(getLayer(network, "Const4"), 63.5f, "custom");
    fillData(getLayer(network, "Const5"), 127.f, "custom");

    fillDataWithInitValue(getLayer(network, "Const7"), "custom", 1.234f);

    fillData(getLayer(network, "Const8"), -1.275f / 2.f, "custom");
    fillData(getLayer(network, "Const9"), 1.275f, "custom");
    fillData(getLayer(network, "Const10"), -1.275f / 2.f, "custom");
    fillData(getLayer(network, "Const11"), 1.275f, "custom");

    fillDataWithInitValue(getLayer(network, "Const13"), "custom", 2.123f);
}

std::string ConvolutionAndQuantizeOnUnsignedActivationsAndWeightsTestModel::getName() const {
    return "ConvolutionAndQuantizeOnUnsignedActivationsAndWeightsTestModel";
}

bool ConvolutionAndQuantizeOnUnsignedActivationsAndWeightsTestModel::transform(CNNNetwork& network, LayerTransformation::Params& params) const {
    LowPrecisionTransformer transformer = getLowPrecisionTransformer(params);
    transformer.transform(network);

    if (params.quantizeOutputs) {
        const std::vector<CNNLayerPtr> layers = CNNNetSortTopologically(network);

        const CNNLayerPtr convolution = layers[layers.size() - 2];
        if ((convolution->type != "Convolution") || (convolution->name != "Convolution14_original")) {
            THROW_IE_EXCEPTION << "unexpected layer type '" << convolution->type << "' or name '" << convolution->name << "'";
        }

        const CNNLayerPtr dequantizationScaleShift = layers[layers.size() - 1];
        if ((dequantizationScaleShift->type != "ScaleShift") || (dequantizationScaleShift->name != "Convolution14")) {
            THROW_IE_EXCEPTION << "unexpected layer type '" << dequantizationScaleShift->type << "' or name '" << dequantizationScaleShift->name << "'";
        }
    }

    return true;
}
