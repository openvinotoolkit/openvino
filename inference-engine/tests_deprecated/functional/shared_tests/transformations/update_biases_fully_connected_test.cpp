// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformer_single_layer_tests.hpp"
#include "low_precision_transformations/fake_quantize.hpp"
#include "low_precision_transformations/convolution.hpp"
#include "low_precision_transformations/fully_connected.hpp"
#include "low_precision_transformations/scaleshift_to_convolution.hpp"

UpdateBiasesFullyConnectedTestModel::UpdateBiasesFullyConnectedTestModel(const bool addBiasesLayer) : FullyConnectedBaseTestModel(addBiasesLayer) {}

std::string UpdateBiasesFullyConnectedTestModel::getName() const {
    return std::string("UpdateBiasesFullyConnectedTestModel") +
        (addBiasesLayer ? "WithBiases" : "WithoutBiases");
}

void UpdateBiasesFullyConnectedTestModel::initInput(Blob::Ptr input) const {
    fillDataWithInitValue(input, -1.f);
}

bool UpdateBiasesFullyConnectedTestModel::transform(CNNNetwork& network, LayerTransformation::Params& params) const {
    // TODO: use getLowPrecisionTransformer(params) instead
    LowPrecisionTransformer transformer(LowPrecisionTransformer::getAllTransformations(params).
        add<FullyConnectedTransformation>(LayerTransformation::Params(params).setSupportAsymmetricQuantization(false), "FullyConnected").
        add<ConvolutionTransformation>(LayerTransformation::Params(params).setPrecisionsOnActivations({ Precision::U8 }), "Convolution").
        addCleanup<ScaleShiftToConvolutionTransformation>(
            LayerTransformation::Params(params).setPrecisionsOnActivations({ Precision::U8 }),
            "ScaleShift"));

    transformer.transform(network);

    if (params.quantizeOutputs) {
        const CNNLayerPtr dequantizationLayer = getLayer(network, "fullyConnected");
        if (dequantizationLayer->type != "ScaleShift") {
            THROW_IE_EXCEPTION << "was not quantized";
        }

        const Blob::Ptr biases = CNNNetworkHelper::getBiases(*dequantizationLayer);
        const std::shared_ptr<float> biasesData = CNNNetworkHelper::getFloatData(biases);
        if (params.updateBiases) {
            for (size_t i = 0ul; i < biases->size(); ++i) {
                if (biasesData.get()[i] != 0.f) {
                    THROW_IE_EXCEPTION << "biases value is not zero";
                }
            }
        } else {
            for (size_t i = 0ul; i < biases->size(); ++i) {
                if (biasesData.get()[i] == 0.f) {
                    THROW_IE_EXCEPTION << "biases value is zero";
                }
            }
        }
    }

    return true;
}
