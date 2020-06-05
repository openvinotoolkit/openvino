// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/convolution.hpp"
#include "low_precision_transformations/network_helper.hpp"

#include <algorithm>
#include <details/caseless.hpp>
#include <memory>
#include <string>
#include <vector>

using namespace InferenceEngine;
using namespace InferenceEngine::details;

void ConvolutionTransformation::calculateDequantizationForAsymmetric(
    const CNNLayer& convolution,
    const std::vector<float>& originalDataDequantizationScales,
    const std::vector<float>& originalDataDequantizationShifts,
    const std::vector<float>& dataZeroPoints,
    const std::vector<float>& originalWeightsDequantizationScales,
    const std::vector<float>& originalWeightsDequantizationShifts,
    const std::vector<float>& weightsZeroPoints,
    std::vector<float>& dequantizationScales,
    std::vector<float>& dequantizationShifts) const {
    const size_t outputChannelCount = CNNNetworkHelper::getOutputChannelsCount(convolution);
    if (originalDataDequantizationScales.size() != outputChannelCount) {
        for (size_t i = 1ul; i < originalDataDequantizationScales.size(); ++i) {
            if (originalDataDequantizationScales[i - 1] != originalDataDequantizationScales[i])
            THROW_IE_EXCEPTION << "original dequantization scales on activations have different values";
        }
    }

    dequantizationScales.resize(outputChannelCount);
    for (size_t i = 0lu; i < dequantizationScales.size(); ++i) {
        const float originalWeightsDequantizationScale = (originalWeightsDequantizationScales.size() == 0) ?
            1.0 : (originalWeightsDequantizationScales.size() == 1 ? originalWeightsDequantizationScales[0] : originalWeightsDequantizationScales[i]);
        const float originalDataDequantizationScale = (originalDataDequantizationScales.size() != dequantizationScales.size()) ?
            originalDataDequantizationScales[0] : originalDataDequantizationScales[i];
        dequantizationScales[i] = originalDataDequantizationScale * originalWeightsDequantizationScale;
    }

    dequantizationShifts.resize(outputChannelCount);

    const Blob::Ptr convolutionBiasesBlob = CNNNetworkHelper::getBiases(convolution);
    if ((convolutionBiasesBlob != nullptr) &&
        convolutionBiasesBlob->getTensorDesc().getPrecision() != Precision::FP32 &&
        convolutionBiasesBlob->getTensorDesc().getPrecision() != Precision::FP16) {
        THROW_IE_EXCEPTION << "Unexpected convolution biases precision "
                           << convolutionBiasesBlob->getTensorDesc().getPrecision();
    }
    const auto convolutionBiasesBuffer = convolutionBiasesBlob == nullptr ? nullptr : CNNNetworkHelper::getFloatData(convolutionBiasesBlob);

    for (size_t outputChannel = 0lu; outputChannel < outputChannelCount; ++outputChannel) {
        const float originalWeightsDequantizationScale =
            originalWeightsDequantizationScales.size() == 0lu
                ? 1.0
                : (originalWeightsDequantizationScales.size() == 1
                       ? originalWeightsDequantizationScales[0]
                       : originalWeightsDequantizationScales[outputChannel]);

        const float originalDataDequantizationScale = (outputChannel < originalDataDequantizationScales.size()) ?
            originalDataDequantizationScales[outputChannel] :
            originalDataDequantizationScales[0];

        dequantizationShifts[outputChannel] =
            convolutionBiasesBuffer == nullptr
                ? 0.0
                : convolutionBiasesBuffer.get()[outputChannel] *
                  (1.0f - originalDataDequantizationScale * originalWeightsDequantizationScale);
    }
}

void ConvolutionTransformation::transform(TransformationContext& context, CNNLayer& layer) const {
    if (!WeightableLayerTransformation::canBeTransformed(context, layer)) {
        return;
    }

    if (!CaselessEq<std::string>()(layer.type, "Convolution")) {
        THROW_IE_EXCEPTION << "Layer '" << layer.name << "' has invalid type '" << layer.type << "'. Convolution is expected.";
    }

    const CNNLayerPtr scaleShiftOnData = CNNNetworkHelper::getParent(layer, 0);
    const CNNLayerPtr parentOnWeights = CNNNetworkHelper::getParent(layer, 1);
    if (parentOnWeights->type != "FakeQuantize") {
        return;
    }

    std::vector<float> originalDataDequantizationScales;
    std::vector<float> originalDataDequantizationShifts;
    fillFromDequantizationLayer(*scaleShiftOnData, originalDataDequantizationScales, originalDataDequantizationShifts);

    const bool isDepthwiseConvolution = isDepthwise(layer);
    if (!isDepthwiseConvolution) {
        // FIXME: it was already checked in WeightableLayerTransformation::canBeTransformed
        for (size_t i = 0lu; i < (originalDataDequantizationScales.size() - 1); ++i) {
            if (originalDataDequantizationScales[i] != originalDataDequantizationScales[i + 1]) {
                return;
            }
        }
    }

    std::vector<float> originalWeightsDequantizationScales;
    std::vector<float> originalWeightsDequantizationShifts;
    const CNNLayerPtr parentOnData = CNNNetworkHelper::getParent(layer, 0ul);

    const DataPrecision dataPrecisionOnWeights = fillDequantizationsForWeightsPath(
        layer,
        supportAsymmetricQuantization,
        originalWeightsDequantizationScales,
        originalWeightsDequantizationShifts);

#ifdef LPT_PRINT_DEQUANTIZATION_INFO
    printDequantizationValues(originalWeightsDequantizationScales, originalWeightsDequantizationShifts);
#endif

    std::vector<float> dequantizationScales;
    std::vector<float> dequantizationShifts;
    if (supportAsymmetricQuantization) {
        std::vector<float> dataZeroPoints(originalDataDequantizationShifts.size());
        for (size_t i = 0ul; i < originalDataDequantizationShifts.size(); ++i) {
            dataZeroPoints[i] = originalDataDequantizationShifts[i] / originalDataDequantizationScales[i];
        }

        std::vector<float> weightsZeroPoints(originalWeightsDequantizationShifts.size());
        for (size_t i = 0ul; i < originalWeightsDequantizationShifts.size(); ++i) {
            weightsZeroPoints[i] = originalWeightsDequantizationShifts[i] / originalWeightsDequantizationScales[i];
        }

        calculateDequantizationForAsymmetric(
            layer,
            originalDataDequantizationScales,
            originalDataDequantizationShifts,
            dataZeroPoints,
            originalWeightsDequantizationScales,
            originalWeightsDequantizationShifts,
            weightsZeroPoints,
            dequantizationScales,
            dequantizationShifts);

        const Precision weightsOriginalPrecision = parentOnWeights->outData[0]->getTensorDesc().getPrecision();
        const PrecisionsInfo dataPrecisionsInfo(
            scaleShiftOnData->outData[0]->getTensorDesc().getPrecision(),
            CNNNetworkHelper::getPrecisionParent(*scaleShiftOnData));

        std::vector<float> dataShifts(originalDataDequantizationShifts.size());
        for (size_t i = 0; i < dataShifts.size(); ++i) {
            dataShifts[i] = -originalDataDequantizationShifts[i] / originalDataDequantizationScales[i];
        }

        std::vector<float> weightsShifts(originalWeightsDequantizationShifts.size());
        for (size_t i = 0; i < weightsShifts.size(); ++i) {
            weightsShifts[i] = -originalWeightsDequantizationShifts[i] / originalWeightsDequantizationScales[i];
        }

        updateToSupportAsymmetricQuantization(
            context,
            layer,
            dataPrecisionsInfo,
            dataShifts,
            PrecisionsInfo(weightsOriginalPrecision, dataPrecisionOnWeights.precision),
            weightsShifts);
    } else {
        if (std::any_of(
            originalWeightsDequantizationShifts.begin(),
            originalWeightsDequantizationShifts.end(),
            [](const float value) { return value != 0.f; })) {
            return;
        }

        calculateDequantizationForSymmetric(
            layer,
            originalDataDequantizationScales,
            originalDataDequantizationShifts,
            originalWeightsDequantizationScales,
            originalWeightsDequantizationShifts,
            dequantizationScales,
            dequantizationShifts);
    }

    if (this->updateBiases) {
        std::vector<float> biasesShifts(dequantizationShifts.size(), 0.f);
        updateLayerBiases(context, layer, false, dequantizationScales, dequantizationShifts, biasesShifts);
    }

    CNNNetworkHelper::removeLayer(context.network, scaleShiftOnData);
    context.removeLayer(*scaleShiftOnData);

    if (parentOnWeights->type == "ScaleShift") {
        CNNNetworkHelper::removeLayer(context.network, parentOnWeights);
        context.removeLayer(*parentOnWeights);
    } else if (parentOnWeights->type == "FakeQuantize") {
        if (weightsToConst) {
            const Blob::Ptr weights = updatePrecisions ?
                CNNNetworkHelper::quantizeWeights(*parentOnWeights, roundQuantizedValues, dataPrecisionOnWeights.precision) :
                CNNNetworkHelper::quantizeWeights(*parentOnWeights, roundQuantizedValues);

            const std::vector<CNNLayerPtr> constLayers = CNNNetworkHelper::transformFakeQuantizeToConst(
                context,
                parentOnWeights,
                weights,
                CNNNetworkHelper::getParent(*parentOnWeights, 0)->name);

            if (updatePrecisions) {
                for (const CNNLayerPtr constLayer : constLayers) {
                    CNNNetworkHelper::setOutDataPrecision(*constLayer, dataPrecisionOnWeights.precision);
                }
            }
        }
    } else {
        THROW_IE_EXCEPTION << "unexpected parent layer type on weights: " << parentOnWeights->type;
    }

    addDequantizationLayer(context, layer, dequantizationScales, dequantizationShifts);
}
