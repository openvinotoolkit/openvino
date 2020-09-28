// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformer_single_layer_tests.hpp"
#include "low_precision_transformations/fake_quantize.hpp"
#include "low_precision_transformations/convolution.hpp"

UpdateBiasesConvolutionTestModel::UpdateBiasesConvolutionTestModel(const bool addBiasesLayer) : ConvolutionBaseTestModel(addBiasesLayer) {}

std::string UpdateBiasesConvolutionTestModel::getName() const {
    return std::string("UpdateBiasesConvolutionTestModel") +
        (addBiasesLayer ? "" : "_withoutBiases");
}

void UpdateBiasesConvolutionTestModel::initInput(Blob::Ptr input) const {
    fillDataWithInitValue(input, -1.f);
}

bool UpdateBiasesConvolutionTestModel::transform(CNNNetwork& network, LayerTransformation::Params& params) const {
    params.supportAsymmetricQuantization = false;

    LowPrecisionTransformer transformer = getLowPrecisionTransformer(params);
    transformer.transform(network);

    if (std::any_of(
        params.precisionsOnActivations.begin(),
        params.precisionsOnActivations.end(),
        [](const Precision precision) { return precision == Precision::U8; }) &&
        params.quantizeOutputs) {
        const CNNLayerPtr dequantizationLayer = getLayer(network, "Convolution");
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

            //CNNLayerPtr convolution = getCreatorLayer(dequantizationLayer->insData[0].lock()).lock();
            //CNNLayerPtr convolutionBiases = CNNNetworkHelper::getParent(*convolution, 2);
            //if (convolutionBiases == nullptr) {
            //    THROW_IE_EXCEPTION << "biases const layer was not added";
            //}
        }
    }

    return true;
}
