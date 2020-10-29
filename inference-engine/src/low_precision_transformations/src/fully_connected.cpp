// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/fully_connected.hpp"

#include <algorithm>
#include <blob_factory.hpp>
#include <cmath>
#include <details/caseless.hpp>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <details/ie_cnn_network_tools.h>
#include <ie_common.h>
#include "cnn_network_impl.hpp"
#include "ie_util_internal.hpp"
#include "low_precision_transformations/common/ie_lpt_exception.hpp"
#include "low_precision_transformations/network_helper.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

void FullyConnectedTransformation::transform(TransformationContext& context, CNNLayer& fullyConnected) const {
    if (!WeightableLayerTransformation::canBeTransformed(context, fullyConnected)) {
        return;
    }

    if (!CaselessEq<std::string>()(fullyConnected.type, "FullyConnected") &&
        !CaselessEq<std::string>()(fullyConnected.type, "GEMM")) {
        THROW_IE_EXCEPTION << "layer '" << fullyConnected.name << "' is not correct";
    }

    if ((fullyConnected.insData.size() != 1) && (fullyConnected.insData.size() != 2) &&
        (fullyConnected.insData.size() != 3)) {
        THROW_IE_EXCEPTION << "layer inputs '" << fullyConnected.insData.size() << "' is not correct";
    }

    const CNNLayerPtr scaleShiftOnData = CNNNetworkHelper::getParent(fullyConnected, 0);
    if (scaleShiftOnData->type != "ScaleShift") {
        return;
    }

    const CNNLayerPtr parentOnWeights = CNNNetworkHelper::getParent(fullyConnected, 1);
    const QuantizationDetails originalQuantizationDetails = parentOnWeights != nullptr ?
        QuantizationDetails::getDetails(*parentOnWeights) :
        QuantizationDetails();

    if (fullyConnected.outData.size() != 1) {
        THROW_IE_EXCEPTION << "layer outputs '" << fullyConnected.outData.size() << "' is not correct";
    }

    std::vector<float> originalDataDequantizationScales;
    std::vector<float> originalDataDequantizationShifts;
    fillFromDequantizationLayer(*scaleShiftOnData, originalDataDequantizationScales, originalDataDequantizationShifts);

    std::vector<float> originalWeightsDequantizationScales;
    std::vector<float> originalWeightsDequantizationShifts;

    if (parentOnWeights != nullptr) {
        if (parentOnWeights->type == "FakeQuantize") {
            fillDequantizationsForWeightsPath(
                fullyConnected,
                supportAsymmetricQuantization,
                originalWeightsDequantizationScales,
                originalWeightsDequantizationShifts);

        } else if (parentOnWeights->type == "Const") {
            originalWeightsDequantizationScales.push_back(1.0);
            originalWeightsDequantizationShifts.push_back(0.0);
        } else {
            THROW_IE_EXCEPTION << "Unexpected dequantization layer type " << parentOnWeights->type;
        }
    }

    std::vector<float> dequantizationScales;
    std::vector<float> dequantizationShifts;
    std::vector<float> biasesShifts;
    if (supportAsymmetricQuantization) {
        std::vector<float> dataShifts(originalDataDequantizationShifts.size());
        for (size_t i = 0; i < dataShifts.size(); ++i) {
            dataShifts[i] = -originalDataDequantizationShifts[i] / originalDataDequantizationScales[i];
        }
        std::vector<float> weightsShifts(originalWeightsDequantizationShifts.size());
        for (size_t i = 0; i < weightsShifts.size(); ++i) {
            weightsShifts[i] = -originalWeightsDequantizationShifts[i] / originalWeightsDequantizationScales[i];
        }

        std::vector<float> dataZeroPoints(originalDataDequantizationShifts.size());
        for (size_t i = 0ul; i < originalDataDequantizationShifts.size(); ++i) {
            dataZeroPoints[i] = originalDataDequantizationShifts[i] / originalDataDequantizationScales[i];
        }

        calculateDequantizationForAsymmetric(
            fullyConnected,
            dataZeroPoints,
            originalWeightsDequantizationScales,
            dequantizationScales,
            dequantizationShifts);

        biasesShifts.resize(dequantizationShifts.size());

        Precision weightsOriginalPrecision;
        Precision weightsLowPrecision;
        if (parentOnWeights->type == "FakeQuantize") {
            weightsOriginalPrecision = parentOnWeights->outData[0]->getTensorDesc().getPrecision();
            weightsLowPrecision = getDataPrecision(
                *parentOnWeights,
                QuantizationDetails::getDetails(*parentOnWeights),
                true,
                supportAsymmetricQuantization).precision;
        } else {
            THROW_IE_EXCEPTION << "unexpected layer type on weights " << parentOnWeights->type;
        }

        const PrecisionsInfo dataPrecisionsInfo(
            scaleShiftOnData->outData[0]->getTensorDesc().getPrecision(),
            CNNNetworkHelper::getPrecisionParent(*scaleShiftOnData));

        updateToSupportAsymmetricQuantization(
            context,
            fullyConnected,
            dataPrecisionsInfo,
            dataShifts,
            PrecisionsInfo(weightsOriginalPrecision, weightsLowPrecision),
            weightsShifts);
    } else {
        if (std::any_of(
            originalWeightsDequantizationShifts.begin(),
            originalWeightsDequantizationShifts.end(),
            [](const float value) { return value != 0.f; })) {
            return;
        }

        calculateDequantizationForSymmetric(
            fullyConnected,
            originalWeightsDequantizationScales,
            originalWeightsDequantizationShifts,
            dequantizationScales,
            dequantizationShifts,
            biasesShifts);
    }

    if (this->updateBiases) {
        updateLayerBiases(context, fullyConnected, dequantizationScales, dequantizationShifts, biasesShifts);
    }

    if ((parentOnWeights != nullptr) && (parentOnWeights->type == "FakeQuantize")) {
        const DataPrecision dataPrecision = getDataPrecision(
            *parentOnWeights,
            originalQuantizationDetails,
            true,
            supportAsymmetricQuantization);

        // disabled, looks like not necessary more - use asymmetric quantization instead
        // std::vector<float> outputLowValues(originalQuantizationDetails.outputIntervalsCount, dataPrecision.min);
        // std::vector<float> outputHighValues(originalQuantizationDetails.outputIntervalsCount, dataPrecision.max);
        // updateWeights(parentOnWeights, outputLowValues, outputHighValues);

        if (weightsToConst) {
            const QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(*parentOnWeights);
            const DataPrecision dataPrecision = getDataPrecision(
                *parentOnWeights,
                quantizationDetails,
                true,
                supportAsymmetricQuantization);

            const Blob::Ptr weights =
                updatePrecisions
                    ? CNNNetworkHelper::quantizeWeights(*parentOnWeights, roundQuantizedValues, dataPrecision.precision)
                    : CNNNetworkHelper::quantizeWeights(*parentOnWeights, roundQuantizedValues);
            const std::vector<CNNLayerPtr> constLayers = CNNNetworkHelper::transformFakeQuantizeToConst(
                context, parentOnWeights, weights, CNNNetworkHelper::getParent(*parentOnWeights, 0)->name);

            if (updatePrecisions) {
                for (const CNNLayerPtr constLayer : constLayers) {
                    CNNNetworkHelper::setOutDataPrecision(*constLayer, dataPrecision.precision);
                }
            }
        }
    }

    CNNNetworkHelper::removeLayer(context.network, scaleShiftOnData);
    context.removeLayer(*scaleShiftOnData);

    const std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildren(fullyConnected);
    const size_t outputChannelsCount = CNNNetworkHelper::getOutputChannelsCount(fullyConnected);
    if (children.size() == 0) {
        const std::string originalName = fullyConnected.name;
        CNNNetworkHelper::renameLayer(
            context.network,
            fullyConnected.name,
            fullyConnected.name + LayerTransformation::lastLayerPrefix);

        CNNLayerPtr dequantizationLayer = CNNNetworkHelper::addScaleShiftBetween(
            context,
            std::make_shared<CNNLayer>(fullyConnected),
            nullptr,
            DequantizationDetails(dequantizationScales, dequantizationShifts, outputChannelsCount),
            originalName);
        context.dequantizationLayersNames.insert(dequantizationLayer->name);
    } else {
        for (const CNNLayerPtr& child : children) {
            CNNLayerPtr dequantizationLayer = CNNNetworkHelper::addScaleShiftBetween(
                context,
                std::make_shared<CNNLayer>(fullyConnected),
                child,
                DequantizationDetails(dequantizationScales, dequantizationShifts, outputChannelsCount));
            context.dequantizationLayersNames.insert(dequantizationLayer->name);
        }
    }
}

void FullyConnectedTransformation::calculateDequantizationForSymmetric(
    const CNNLayer& fullyConnected,
    const std::vector<float>& originalWeightsDequantizationScales,
    const std::vector<float>& originalWeightsDequantizationShifts,
    std::vector<float>& dequantizationScales,
    std::vector<float>& dequantizationShifts,
    std::vector<float>& biasesShifts) const {
    for (size_t i = 0; i < originalWeightsDequantizationShifts.size(); ++i) {
        if (originalWeightsDequantizationShifts[i] != 0.0) {
            THROW_IE_EXCEPTION << "shift values on weights for '" << fullyConnected.type << "' layer '" << fullyConnected.name << "' are not supported";
        }
    }

    const DataPtr inputData = fullyConnected.insData[0].lock();
    if (inputData == nullptr) {
        THROW_IE_EXCEPTION << "input data is absent for layer " << fullyConnected.name;
    }
    const Layout inputLayout = inputData->getLayout();
    if (inputLayout != Layout::NC) {
        THROW_IE_EXCEPTION << "Unexpected input layout " << inputLayout;
    }

    const DataPtr insData = fullyConnected.insData[0].lock();
    if (insData == nullptr) {
        THROW_IE_LPT_EXCEPTION(fullyConnected) << "insert data ia absent";
    }
    const size_t inputChannelsCount = insData->getDims()[1];

    const Layout outputLayout = fullyConnected.outData[0]->getLayout();
    if (outputLayout != Layout::NC) {
        THROW_IE_EXCEPTION << "Unexpected output layout " << outputLayout;
    }
    const size_t outputChannelsCount = fullyConnected.outData[0]->getDims()[1];
    dequantizationScales.resize(outputChannelsCount);
    dequantizationShifts.resize(outputChannelsCount);
    biasesShifts.resize(outputChannelsCount);

    CNNLayerPtr scaleShift = CNNNetworkHelper::getParent(fullyConnected);
    if (scaleShift->type != "ScaleShift") {
        THROW_IE_EXCEPTION << "Unexpected layer type to calculate quantization values " << scaleShift->type;
    }

    const Blob::Ptr prevDequantizationScaleBlob = CNNNetworkHelper::getBlob(scaleShift, "weights");
    const auto prevDequantizationScaleBuffer = CNNNetworkHelper::getFloatData(prevDequantizationScaleBlob);
    const Blob::Ptr prevDequantizationShiftBlob = CNNNetworkHelper::getBlob(scaleShift, "biases");
    const auto prevDequantizationShiftBuffer = CNNNetworkHelper::getFloatData(prevDequantizationShiftBlob);

    const Blob::Ptr weightsBlob = CNNNetworkHelper::getWeights(fullyConnected, roundQuantizedValues);
    const auto weightsBuffer = CNNNetworkHelper::getFloatData(weightsBlob);
    const Blob::Ptr biasesBlob = CNNNetworkHelper::getBiases(fullyConnected);
    const auto biasesBuffer = biasesBlob == nullptr ? nullptr : CNNNetworkHelper::getFloatData(biasesBlob);

    const float prevDequantizationScale = prevDequantizationScaleBuffer.get()[0];
    for (size_t i = 0; i < outputChannelsCount; ++i) {
        dequantizationScales[i] = prevDequantizationScale *
            (originalWeightsDequantizationScales.size() == 0 ?
                1.0 :
                (originalWeightsDequantizationScales.size() == 1 ? originalWeightsDequantizationScales[0] : originalWeightsDequantizationScales[i]));
    }

    for (size_t channel = 0lu; channel < outputChannelsCount; ++channel) {
        float sum = 0.0;
        const float weightsDequantizationScale = originalWeightsDequantizationScales.size() == 0 ?
            1.0 :
            (originalWeightsDequantizationScales.size() == 1 ? originalWeightsDequantizationScales[0] : originalWeightsDequantizationScales[channel]);

        for (size_t inputChannel = 0; inputChannel < inputChannelsCount; ++inputChannel) {
            const float w = weightsBuffer.get()[channel * inputChannelsCount + inputChannel];
            sum += w * prevDequantizationShiftBuffer.get()[inputChannel] * weightsDequantizationScale;
        }

        dequantizationShifts[channel] = biasesBuffer == nullptr ?
            sum :
            (sum + biasesBuffer.get()[channel] - prevDequantizationScale * biasesBuffer.get()[channel] * weightsDequantizationScale);
        biasesShifts[channel] = sum;
    }
}

void FullyConnectedTransformation::calculateDequantizationForAsymmetric(
    const CNNLayer& fullyConnected,
    const std::vector<float>& dataZeroPoints,
    const std::vector<float>& originalWeightsDequantizationScales,
    std::vector<float>& dequantizationScales,
    std::vector<float>& dequantizationShifts) const {
    const DataPtr inputData = fullyConnected.insData[0].lock();
    if (inputData == nullptr) {
        THROW_IE_EXCEPTION << "input data is absent for layer " << fullyConnected.name;
    }
    const Layout inputLayout = inputData->getLayout();
    if (inputLayout != Layout::NC) {
        THROW_IE_EXCEPTION << "Unexpected input layout " << inputLayout;
    }
    const DataPtr insData = fullyConnected.insData[0].lock();
    if (insData == nullptr) {
        THROW_IE_LPT_EXCEPTION(fullyConnected) << "insert data is absent";
    }
    const size_t inputChannelsCount = insData->getDims()[1];

    const Layout outputLayout = fullyConnected.outData[0]->getLayout();
    if (outputLayout != Layout::NC) {
        THROW_IE_EXCEPTION << "Unexpected output layout " << outputLayout;
    }
    const size_t outputChannelsCount = fullyConnected.outData[0]->getDims()[1];
    dequantizationScales.resize(outputChannelsCount);
    dequantizationShifts.resize(outputChannelsCount);

    CNNLayerPtr scaleShift = CNNNetworkHelper::getParent(fullyConnected);
    if (scaleShift->type != "ScaleShift") {
        THROW_IE_EXCEPTION << "Unexpected layer type to calculate quantization values " << scaleShift->type;
    }

    const Blob::Ptr prevDequantizationScaleBlob = CNNNetworkHelper::getBlob(scaleShift, "weights");
    const auto prevDequantizationScaleBuffer = CNNNetworkHelper::getFloatData(prevDequantizationScaleBlob);
    const Blob::Ptr prevDequantizationShiftBlob = CNNNetworkHelper::getBlob(scaleShift, "biases");
    const auto prevDequantizationShiftBuffer = CNNNetworkHelper::getFloatData(prevDequantizationShiftBlob);

    const Blob::Ptr weightsBlob = CNNNetworkHelper::getWeights(fullyConnected, roundQuantizedValues);
    const auto weightsBuffer = CNNNetworkHelper::getFloatData(weightsBlob);
    const Blob::Ptr biasesBlob = CNNNetworkHelper::getBiases(fullyConnected);
    const auto biasesBuffer = biasesBlob == nullptr ? nullptr : CNNNetworkHelper::getFloatData(biasesBlob);

    for (size_t i = 0; i < outputChannelsCount; ++i) {
        dequantizationScales[i] =
            prevDequantizationScaleBuffer.get()[0] *
            (originalWeightsDequantizationScales.size() == 0
                 ? 1.0
                 : (originalWeightsDequantizationScales.size() == 1 ? originalWeightsDequantizationScales[0]
                                                                    : originalWeightsDequantizationScales[i]));
    }

    for (size_t channel = 0lu; channel < outputChannelsCount; ++channel) {
        float sum1 = 0.0;
        float sum2 = 0.0;
        const float weightsDequantizationScale =
            originalWeightsDequantizationScales.size() == 0
                ? 1.0
                : (originalWeightsDequantizationScales.size() == 1 ? originalWeightsDequantizationScales[0]
                                                                   : originalWeightsDequantizationScales[channel]);

        for (size_t w = 0; w < inputChannelsCount; ++w) {
            const float kernel = weightsBuffer.get()[channel * inputChannelsCount + w];
            sum1 += kernel * prevDequantizationShiftBuffer.get()[channel] * weightsDequantizationScale;
            sum2 += kernel * dataZeroPoints[w] * weightsDequantizationScale;
        }

        dequantizationShifts[channel] = biasesBuffer == nullptr
                                            ? sum1
                                            : (sum1 + biasesBuffer.get()[channel] -
                                               prevDequantizationScaleBuffer.get()[channel] *
                                                   biasesBuffer.get()[channel] * weightsDequantizationScale);
    }
}
