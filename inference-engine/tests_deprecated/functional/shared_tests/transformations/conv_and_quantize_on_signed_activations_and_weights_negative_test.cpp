// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformer_single_layer_tests.hpp"

void ConvolutionAndQuantizeOnSignedActivationsAndWeightsNegativeTestModel::resetTransformation(CNNNetwork& network) const {
    fillData(getLayer(network, "Const2"), -128.f / 4.f, "custom");
    fillData(getLayer(network, "Const3"), 127.f / 4.f, "custom");
    fillData(getLayer(network, "Const4"), -128.f / 4.f, "custom");
    fillData(getLayer(network, "Const5"), 127.f / 4.f, "custom");

    fillDataWithInitValue(getLayer(network, "Const7"), "custom", 1.234);

    //fillData(getLayer(network, "Const8"), 0.f, "custom");
    //fillData(getLayer(network, "Const9"), 255.f / 40.f, "custom");
    //fillData(getLayer(network, "Const10"), 0.f, "custom");
    //fillData(getLayer(network, "Const11"), 255.f / 40.f, "custom");

    fillData(getLayer(network, "Const8"), -255.f / 40.f, "custom");
    fillData(getLayer(network, "Const9"), 0.f, "custom");
    fillData(getLayer(network, "Const10"), -255.f / 40.f, "custom");
    fillData(getLayer(network, "Const11"), 0.f, "custom");


    fillDataWithInitValue(getLayer(network, "Const13"), "custom", 2.123f);
}

std::string ConvolutionAndQuantizeOnSignedActivationsAndWeightsNegativeTestModel::getName() const {
    return "ConvolutionAndQuantizeOnSignedActivationsAndWeightsNegativeTestModel";
}

bool ConvolutionAndQuantizeOnSignedActivationsAndWeightsNegativeTestModel::transform(CNNNetwork& network, LayerTransformation::Params& params) const {
    LowPrecisionTransformer transformer = getLowPrecisionTransformer(params);
    transformer.transform(network);

    if (std::any_of(
        params.precisionsOnActivations.begin(),
        params.precisionsOnActivations.end(),
        [](const Precision precision) { return precision == Precision::U8; }) &&
        params.quantizeOutputs) {
        CNNLayerPtr scaleShfit = CNNNetworkHelper::getLayer(network, "Convolution14");
        if (scaleShfit->type != "ScaleShift") {
            THROW_IE_EXCEPTION << "unexpected last output dequantization layer type " << scaleShfit->name;
        }

        if (params.updateBiases) {
            const Blob::Ptr shiftsBlob = CNNNetworkHelper::getBlob(scaleShfit, "biases");
            std::shared_ptr<float> shiftsBuffer = CNNNetworkHelper::getFloatData(shiftsBlob);
            for (size_t i = 0ul; i < shiftsBlob->size(); ++i) {
                if (shiftsBuffer.get()[i] != 0.0) {
                    THROW_IE_EXCEPTION << "unexpected dequantization shift value";
                }
            }
        }
    }

    return true;
}
