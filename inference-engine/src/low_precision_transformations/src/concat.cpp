// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/concat.hpp"

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

#include <ie_common.h>
#include <details/ie_cnn_network_tools.h>
#include "cnn_network_impl.hpp"
#include "ie_util_internal.hpp"

#include "low_precision_transformations/common/ie_lpt_exception.hpp"
#include "low_precision_transformations/network_helper.hpp"
#include "low_precision_transformations/quantization_details.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

bool ConcatTransformation::getQuantizeLayers(
    CNNLayerPtr layer,
    std::vector<std::string>& childNameOurAfterQuantizeLayers,
    std::vector<CNNLayerPtr>& quantizeLayers,
    std::vector<std::vector<std::pair<CNNLayerPtr, CNNLayerPtr>>>& intermediateLayers,
    std::vector<CNNLayerPtr>& concatLayers,
    CNNLayerPtr child,
    std::vector<CNNLayerPtr>& sideOutputLayers,
    std::vector<std::string>& childrenNameSideOutputLayers) {
    if (!CaselessEq<std::string>()(layer->type, "FakeQuantize") &&
        !CaselessEq<std::string>()(layer->type, "Quantize")) {
        do {
            if (CaselessEq<std::string>()(layer->type, "Pooling") || CaselessEq<std::string>()(layer->type, "Resample")) {
                intermediateLayers.back().push_back(std::pair<CNNLayerPtr, CNNLayerPtr>(
                    layer,
                    concatLayers.empty() ? child : concatLayers.back()));
                child = layer;
                layer = CNNNetworkHelper::getParent(*layer, 0);
            } else if (CaselessEq<std::string>()(layer->type, "Concat")) {
                concatLayers.push_back(layer);

                if (layer->outData[0]->getInputTo().size() != 1) {
                    sideOutputLayers.push_back(layer);
                    childrenNameSideOutputLayers.push_back(child->name);
                }
                int size = layer->insData.size();
                child = layer;
                for (int i = 0; i < size; i++) {
                    CNNLayerPtr layer1 = CNNNetworkHelper::getParent(*layer, i);
                    intermediateLayers.push_back({});
                    if (!getQuantizeLayers(
                        layer1,
                        childNameOurAfterQuantizeLayers,
                        quantizeLayers,
                        intermediateLayers,
                        concatLayers,
                        child,
                        sideOutputLayers,
                        childrenNameSideOutputLayers)) {
                        return false;
                    }
                }
                return true;
            } else {
                return false;
            }
        } while (!CaselessEq<std::string>()(layer->type, "FakeQuantize") &&
                 !CaselessEq<std::string>()(layer->type, "Quantize"));
    }

    childNameOurAfterQuantizeLayers.push_back(child->name);
    quantizeLayers.push_back(layer);
    return true;
}

void ConcatTransformation::transform(TransformationContext& context, CNNLayer& concat) const {
    if (!canBeTransformed(context, concat)) {
        return;
    }

    if (!CaselessEq<std::string>()(concat.type, "Concat")) {
        THROW_IE_EXCEPTION << "layer type '" << concat.name << "' is not correct";
    }

    if (concat.GetParamAsUInt("axis", 1) != 1) {
        return;
    }

    if ((concat.insData.size() < 2)) {
        THROW_IE_EXCEPTION << "layer inputs '" << concat.insData.size() << "' is not correct";
    }

    std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildren(concat);
    if (CNNNetworkHelper::IsChild(children, { "Concat" }, { "Pooling", "Resample" })) {
        return;
    }

    std::vector<CNNLayerPtr> quantizeLayers;
    std::vector<std::vector<std::pair<CNNLayerPtr, CNNLayerPtr>>> intermediateLayers;
    std::vector<CNNLayerPtr> concatLayers;
    const auto inputDataNumber = concat.insData.size();
    std::vector<std::string> childNameOurAfterQuantizeLayers;
    std::vector<QuantizationDetails> quantizationLayersDetails;
    std::vector<CNNLayerPtr> sideOutputLayers;
    std::vector<std::string> childrenNameSideOutputLayers;
    for (size_t index = 0lu; index < inputDataNumber; index++) {
        DataPtr quantizeOnData = concat.insData[index].lock();
        if (quantizeOnData == nullptr) {
            THROW_IE_EXCEPTION << "input is absent";
        }
        auto parentLayer = quantizeOnData->getCreatorLayer().lock();
        intermediateLayers.push_back({});
        if (!getQuantizeLayers(
            parentLayer,
            childNameOurAfterQuantizeLayers,
            quantizeLayers,
            intermediateLayers,
            concatLayers,
            std::make_shared<CNNLayer>(concat),
            sideOutputLayers,
            childrenNameSideOutputLayers)) {
            return;
        }
    }

    if (quantizeLayers.empty()) {
        return;
    }

    for (const CNNLayerPtr& quantizeLayer : quantizeLayers) {
        if (!QuantizationDetails::outputLayoutIsSupported(*quantizeLayer)) {
            return;
        }
    }

    DataPrecision dataPrecision = getDataPrecision(*quantizeLayers[0], QuantizationDetails::getDetails(*quantizeLayers[0]), false, false);
    if (dataPrecision.precision == Precision::UNSPECIFIED) {
        return;
    }

    const QuantizationDetails& quantizationDetails1 = QuantizationDetails::getDetails(*quantizeLayers[0]);
    const QuantizationDetails& quantizationDetails2 = QuantizationDetails::getDetails(*quantizeLayers[1]);

    size_t quantizationLevels = 0lu;
    for (int i = 0; i < quantizeLayers.size(); i++) {
        const QuantizationDetails& quantizationDetails = QuantizationDetails::getDetails(*quantizeLayers[i]);
        if (!QuantizationDetails::isSupportedLevel(quantizationDetails.levels)) continue;
        if (quantizationLevels == 0lu) {
            quantizationLevels = quantizationDetails.levels;
        } else if (quantizationLevels != quantizationDetails.levels) {
            THROW_IE_EXCEPTION << "different quantization levels " << quantizationLevels << " are not supported";
        }

        quantizationLayersDetails.push_back(quantizationDetails);

        const DataPrecision dataPrecision2 = getDataPrecision(*quantizeLayers[i], quantizationDetails, false, false);
        if (dataPrecision2.precision == Precision::UNSPECIFIED) {
            return;
        }

        if (dataPrecision.precision != dataPrecision2.precision) {
            // quantization levels are the same, difference can be in sign
            // wider interval (precision) is preferable: use signed if least one interval is signed
            dataPrecision = dataPrecision.precision.isSigned() ? dataPrecision : dataPrecision2;
        }
    }

    if (dataPrecision.precision == Precision::UNSPECIFIED) {
        return;
    }

    // per tensor scale is supported only
    if (quantizationLayersDetails.empty() || (quantizationLayersDetails[0].inputHighValues.size() != 1ul)) {
        return;
    }

    std::vector<float> dequantizationScales;
    std::vector<float> dequantizationShifts;
    const size_t outputChannelsCount = CNNNetworkHelper::getOutputChannelsCount(concat);

    dequantizationScales.resize(outputChannelsCount);
    dequantizationShifts.resize(outputChannelsCount);
    const auto parentsCount = quantizeLayers.size();

    std::unordered_map<std::string, std::vector<float>> dequantizationScalesLayers;
    std::unordered_map<std::string, std::vector<float>> dequantizationShiftsLayers;

    if ((quantizationLayersDetails[0].inputHighValues.size() == 1)) {
        float outputLowValue = quantizationLayersDetails[0].outputLowValues[0];
        float outputHighValue = quantizationLayersDetails[0].outputHighValues[0];
        for (size_t index = 0lu; index < parentsCount; index++) {
            if (outputLowValue > quantizationLayersDetails[index].outputLowValues[0]) {
                outputLowValue = quantizationLayersDetails[index].outputLowValues[0];
            }
            if (outputHighValue < quantizationLayersDetails[index].outputHighValues[0]) {
                outputHighValue = quantizationLayersDetails[index].outputHighValues[0];
            }
        }

        if ((outputLowValue == 0.f) && (outputHighValue == 0.f)) {
            return;
        }

        const float maxOutputInterval = outputHighValue - outputLowValue;
        if (quantizedTensorAlignmentOnActivations == QuantizedTensorAlignment::UpdateLevel) {
            const size_t minLevels = getMinQuantizationLevels(
                dataPrecision,
                maxOutputInterval,
                quantizationLayersDetails,
                outputLowValue,
                outputHighValue);
            if (minLevels < this->minQuantizationLevels) {
                return;
            }
        }


        const float dequantizationScale = maxOutputInterval / (dataPrecision.max - dataPrecision.min);
        const float max = maxOutputInterval / ((dataPrecision.max - dataPrecision.min) / dataPrecision.max);
        const float min = maxOutputInterval / ((dataPrecision.max - dataPrecision.min) / dataPrecision.min);
        const float dequantizationShift = outputLowValue - min;

        const float quantizationScale = 1.f / dequantizationScale;
        const float quantizationShift = - dequantizationShift * quantizationScale;

        for (int index = 0; index < parentsCount; index++) {
            if (quantizeLayers[index] == nullptr)
                continue;
            CNNLayer& fakeQuantizeLayer = *quantizeLayers[index];
            const QuantizationDetails quantizationDetails = quantizationLayersDetails[index];

            switch (quantizedTensorAlignmentOnActivations) {
            case QuantizedTensorAlignment::None: {
                const float updatedOutputLowValue = quantizationDetails.outputLowValues[0] * quantizationScale + quantizationShift;
                CNNNetworkHelper::updateBlobs(fakeQuantizeLayer, 3, updatePrecisions ? roundf(updatedOutputLowValue) : updatedOutputLowValue);

                const float updatedOutputHighValue = quantizationDetails.outputHighValues[0] * quantizationScale + quantizationShift;
                CNNNetworkHelper::updateBlobs(fakeQuantizeLayer, 4, updatePrecisions ? roundf(updatedOutputHighValue) : updatedOutputHighValue);

                break;
            }
            case QuantizedTensorAlignment::UpdateIntervals: {
                const float inputLowValue = quantizationDetails.outputLowValues[0] != 0.0
                                                ? (quantizationDetails.inputLowValues[0] *
                                                   (outputLowValue / quantizationDetails.outputLowValues[0]))
                                                : outputLowValue;
                const float inputHighValue = quantizationDetails.outputHighValues[0] != 0.0
                                                 ? (quantizationDetails.inputHighValues[0] *
                                                    (outputHighValue / quantizationDetails.outputHighValues[0]))
                                                 : outputHighValue;

                CNNNetworkHelper::updateBlobs(fakeQuantizeLayer, 1, inputLowValue);
                CNNNetworkHelper::updateBlobs(fakeQuantizeLayer, 2, inputHighValue);
                CNNNetworkHelper::updateBlobs(fakeQuantizeLayer, 3, dataPrecision.min);
                CNNNetworkHelper::updateBlobs(fakeQuantizeLayer, 4, dataPrecision.max);
                break;
            }
            case QuantizedTensorAlignment::UpdateLevel: {
                const float updatedOutputLowValue = quantizationDetails.outputLowValues[0] * quantizationScale + quantizationShift;
                CNNNetworkHelper::updateBlobs(fakeQuantizeLayer, 3, updatePrecisions ? roundf(updatedOutputLowValue) : updatedOutputLowValue);

                const float updatedOutputHighValue = quantizationDetails.outputHighValues[0] * quantizationScale + quantizationShift;
                CNNNetworkHelper::updateBlobs(fakeQuantizeLayer, 4, updatePrecisions ? roundf(updatedOutputHighValue) : updatedOutputHighValue);

                const int levels = static_cast<int>(fabs(roundf(updatedOutputHighValue) - roundf(updatedOutputLowValue)) + 1.0);
                fakeQuantizeLayer.params["levels"] = std::to_string(levels);
                QuantizeLayer* layer = dynamic_cast<QuantizeLayer*>(&fakeQuantizeLayer);
                if (layer == nullptr) {
                    THROW_IE_EXCEPTION << "incorrect type for layer " << fakeQuantizeLayer.name;
                }
                layer->levels = levels;

                break;
            }
            default: {
                THROW_IE_EXCEPTION << "unexpected value " << quantizedTensorAlignmentOnActivations;
            }
            }

            if (updatePrecisions) {
                CNNNetworkHelper::setOutDataPrecision(fakeQuantizeLayer, dataPrecision.precision);
            }

            const size_t fakeQuantizeOutputChannelsCount = CNNNetworkHelper::getOutputChannelsCount(fakeQuantizeLayer);

            const std::vector<float> fakeQuantizeDequantizationScales(fakeQuantizeOutputChannelsCount, dequantizationScale);
            dequantizationScalesLayers[fakeQuantizeLayer.name] = fakeQuantizeDequantizationScales;

            const std::vector<float> fakeQuantizeDequantizationShifts(fakeQuantizeOutputChannelsCount, dequantizationShift);
            dequantizationShiftsLayers[fakeQuantizeLayer.name] = fakeQuantizeDequantizationShifts;
        }

        dequantizationScales.resize(outputChannelsCount);
        std::fill(dequantizationScales.begin(), dequantizationScales.end(), dequantizationScale);

        dequantizationShifts.resize(outputChannelsCount);
        std::fill(dequantizationShifts.begin(), dequantizationShifts.end(), dequantizationShift);
    } else {
        return;
    }

    addDequantizationForQuantize(
        context,
        concat,
        quantizeLayers,
        intermediateLayers,
        childNameOurAfterQuantizeLayers,
        dequantizationScalesLayers,
        dequantizationShiftsLayers);

    if (updatePrecisions) {
        for (const std::vector<std::pair<CNNLayerPtr, CNNLayerPtr>>& intermediateLayersList : intermediateLayers) {
            for (const std::pair<CNNLayerPtr, CNNLayerPtr>& pair : intermediateLayersList) {
                CNNLayerPtr intermediateLayer = pair.first;
                CNNNetworkHelper::setOutDataPrecision(*intermediateLayer, dataPrecision.precision);
            }
        }

        CNNNetworkHelper::setOutDataPrecision(concat, dataPrecision.precision);
        for (const CNNLayerPtr& concatLayer : concatLayers) {
            // TODO: check if the same precision is used: U8 or S8 for all concat layers
            CNNNetworkHelper::setOutDataPrecision(*concatLayer, dataPrecision.precision);
        }
    }

    // Add scaleshift at outputs of our layers
    children = CNNNetworkHelper::getChildren(concat);
    if (children.size() == 0) {
        const std::string originalName = concat.name;
        CNNNetworkHelper::renameLayer(context.network, concat.name, concat.name + LayerTransformation::lastLayerPrefix);

        CNNLayerPtr dequantizationLayer = CNNNetworkHelper::addScaleShiftBetween(
            context,
            std::make_shared<CNNLayer>(concat),
            nullptr,
            DequantizationDetails(dequantizationScales, dequantizationShifts, outputChannelsCount), originalName);
        context.dequantizationLayersNames.insert(dequantizationLayer->name);
    } else {
        for (const CNNLayerPtr& child : children) {
            CNNLayerPtr dequantizationLayer = CNNNetworkHelper::addScaleShiftBetween(
                context,
                std::make_shared<CNNLayer>(concat),
                child,
                DequantizationDetails(dequantizationScales, dequantizationShifts, outputChannelsCount));
            context.dequantizationLayersNames.insert(dequantizationLayer->name);
        }
    }

    // Add scaleshift at outputs of side branches
    for (int index = 0; index < sideOutputLayers.size(); index++) {
        const size_t outputChannelsCount = CNNNetworkHelper::getOutputChannelsCount(*sideOutputLayers[index]);
        std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildren(*sideOutputLayers[index], childrenNameSideOutputLayers[index]);
        for (int i = 0; i < children.size(); i++) {
            std::vector<float> dequantizationScales1(outputChannelsCount, dequantizationScales[0]);
            std::vector<float> dequantizationShifts1(outputChannelsCount, dequantizationShifts[0]);
            CNNLayerPtr dequantizationLayer = CNNNetworkHelper::addScaleShiftBetween(
                context,
                std::make_shared<CNNLayer>(*sideOutputLayers[index]),
                children[i],
                DequantizationDetails(dequantizationScales1, dequantizationShifts1, outputChannelsCount));
            context.dequantizationLayersNames.insert(dequantizationLayer->name);
        }
    }
}

void ConcatTransformation::addDequantizationForQuantize(
    TransformationContext& context,
    const CNNLayer& concat,
    const std::vector<CNNLayerPtr>& quantizeLayers,
    const std::vector<std::vector<std::pair<CNNLayerPtr, CNNLayerPtr>>>& intermediateLayers,
    const std::vector<std::string>& childNameOurAfterQuantizeLayers,
    const std::unordered_map<std::string, std::vector<float>>& dequantizationScalesLayers,
    const std::unordered_map<std::string, std::vector<float>>& dequantizationShiftsLayers) const {
    const size_t parentsCount = quantizeLayers.size();
    for (int index = 0; index < parentsCount; index++) {
        CNNLayer& fakeQuantize = *quantizeLayers[index];
        context.quantizedFakeQuantizeNames.insert(quantizeLayers[index]->name);
        if (quantizeLayers[index]->outData[0]->getInputTo().size() != 1) {
            std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildren(*quantizeLayers[index], childNameOurAfterQuantizeLayers[index]);

            for (int i = 0; i < children.size(); i++) {
                const size_t outputChannelsCount = CNNNetworkHelper::getOutputChannelsCount(*quantizeLayers[index]);

                auto dequantizationScalesIt = dequantizationScalesLayers.find(fakeQuantize.name);
                if (dequantizationScalesIt == dequantizationScalesLayers.end()) {
                    THROW_IE_EXCEPTION << "dequantization scales not found for layer " << fakeQuantize.name;
                }

                auto dequantizationShiftIt = dequantizationShiftsLayers.find(fakeQuantize.name);
                if (dequantizationShiftIt == dequantizationShiftsLayers.end()) {
                    THROW_IE_EXCEPTION << "dequantization shifts not found for layer " << fakeQuantize.name;
                }

                CNNLayerPtr dequantizationLayer = CNNNetworkHelper::addScaleShiftBetween(
                    context,
                    std::make_shared<CNNLayer>(*quantizeLayers[index]),
                    children[i],
                    DequantizationDetails(dequantizationScalesIt->second, dequantizationShiftIt->second, outputChannelsCount));
                context.dequantizationLayersNames.insert(dequantizationLayer->name);
            }
        }
    }

    for (const std::vector<std::pair<CNNLayerPtr, CNNLayerPtr>>& intermediateLayersList : intermediateLayers) {
        for (auto it = intermediateLayersList.rbegin(); it != intermediateLayersList.rend(); ++it) {
            const std::pair<CNNLayerPtr, CNNLayerPtr> intermediateLayerPair = *it;
            const CNNLayerPtr intermediateLayer = intermediateLayerPair.first;
            const CNNLayerPtr concatLayer = intermediateLayerPair.second;

            const CNNLayerPtr nextIntermediateLayer = (it + 1) != intermediateLayersList.rend() ? (*(it + 1)).first : concatLayer;
            const std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildren(*intermediateLayer, nextIntermediateLayer->name);
            if (!children.empty()) {
                CNNLayerPtr layer = intermediateLayer;
                while (layer->type != "FakeQuantize") {
                    std::vector<CNNLayerPtr> parents = CNNNetworkHelper::getParents(*layer);
                    if (parents.empty()) {
                        THROW_IE_LPT_EXCEPTION(*intermediateLayer) << "intermediate layer doesn't have parents";
                    }
                    if (parents.size() > 1ul) {
                        THROW_IE_LPT_EXCEPTION(*intermediateLayer) << "intermediate layer has several parents";
                    }

                    layer = parents[0];
                }
                const CNNLayerPtr fakeQuantize = layer;

                for (int childIndex = 0; childIndex < children.size(); childIndex++) {
                    const size_t outputChannelsCount = CNNNetworkHelper::getOutputChannelsCount(*intermediateLayer);

                    const auto dequantizationScalesIt = dequantizationScalesLayers.find(fakeQuantize->name);
                    if (dequantizationScalesIt == dequantizationScalesLayers.end()) {
                        THROW_IE_EXCEPTION << "dequantization scales not found for layer " << fakeQuantize->name;
                    }

                    const auto dequantizationShiftIt = dequantizationShiftsLayers.find(fakeQuantize->name);
                    if (dequantizationShiftIt == dequantizationShiftsLayers.end()) {
                        THROW_IE_EXCEPTION << "dequantization shifts not found for layer " << fakeQuantize->name;
                    }

                    CNNLayerPtr dequantizationLayer = CNNNetworkHelper::addScaleShiftBetween(
                        context,
                        intermediateLayer,
                        children[childIndex],
                        DequantizationDetails(dequantizationScalesIt->second, dequantizationShiftIt->second, outputChannelsCount));
                    context.dequantizationLayersNames.insert(dequantizationLayer->name);
                }
            }
        }
    }
}

size_t ConcatTransformation::getMinQuantizationLevels(
    const DataPrecision& dataPrecision,
    const float maxOutputInterval,
    const std::vector<QuantizationDetails>& quantizationLayersDetails,
    const float outputLowValue,
    const float outputHighValue) const {
    size_t minLevels = std::numeric_limits<std::size_t>::max();
    for (const QuantizationDetails quantizationDetails : quantizationLayersDetails) {
        // if there is negative part then calculation is based on `outputLowValue` if not then on `outputHighValue` only
        const float updatedOutputLowValue = outputLowValue != 0.f ?
            (quantizationDetails.outputLowValues[0] / outputLowValue) * dataPrecision.min :
            (quantizationDetails.outputLowValues[0] / outputHighValue) * dataPrecision.max;

        // if there is positive part then calculation is based on `outputHighValue` if not then on `outputLowValue` only
        const float updatedOutputHighValue = outputHighValue != 0.f ?
            (quantizationDetails.outputHighValues[0] / outputHighValue) * dataPrecision.max :
            (quantizationDetails.outputHighValues[0] / outputLowValue) * dataPrecision.min;

        const int levels = static_cast<int>(fabs(roundf(updatedOutputHighValue) - roundf(updatedOutputLowValue)) + 1.0);
        if (minLevels > levels) {
            minLevels = levels;
        }
    }
    return minLevels;
}
