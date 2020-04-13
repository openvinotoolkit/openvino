// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/concat.hpp"
#include "low_precision_transformations/network_helper.hpp"
#include "low_precision_transformations/quantization_details.hpp"

#include <details/ie_cnn_network_tools.h>
#include <ie_common.h>

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

#include "cnn_network_impl.hpp"
#include "ie_util_internal.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

bool ConcatTransformation::getQuantizeLayers(
    CNNLayerPtr layer,
    std::vector<std::string>& childNameOurAfterQuantizeLayers,
    std::vector<CNNLayerPtr>& quantizeLayers,
    std::vector<std::vector<CNNLayerPtr>>& intermediateLayers,
    std::vector<CNNLayerPtr>& concatLayers,
    std::string childName,
    std::vector<CNNLayerPtr>& sideOutputLayers,
    std::vector<std::string>& childrenNameSideOutputLayers) {
    if (!CaselessEq<std::string>()(layer->type, "FakeQuantize") &&
        !CaselessEq<std::string>()(layer->type, "Quantize")) {
        do {
            if (CaselessEq<std::string>()(layer->type, "Pooling")) {
                intermediateLayers.back().push_back(layer);
                childName = layer->name;
                layer = CNNNetworkHelper::getParent(*layer, 0);
            } else if (CaselessEq<std::string>()(layer->type, "Concat")) {
                concatLayers.push_back(layer);

                if (layer->outData[0]->getInputTo().size() != 1) {
                    sideOutputLayers.push_back(layer);
                    childrenNameSideOutputLayers.push_back(childName);
                }
                int size = layer->insData.size();
                childName = layer->name;
                for (int i = 0; i < size; i++) {
                    CNNLayerPtr layer1 = CNNNetworkHelper::getParent(*layer, i);
                    intermediateLayers.push_back({});
                    if (!getQuantizeLayers(
                        layer1,
                        childNameOurAfterQuantizeLayers,
                        quantizeLayers,
                        intermediateLayers,
                        concatLayers,
                        childName,
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

    childNameOurAfterQuantizeLayers.push_back(childName);
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
    if (CNNNetworkHelper::IsChild(children, {"Concat"}, {"Pooling"})) {
        return;
    }

    std::vector<CNNLayerPtr> quantizeLayers;
    std::vector<std::vector<CNNLayerPtr>> intermediateLayers;
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
            concat.name,
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
    std::vector<std::vector<float>> dequantizationShiftsLayers;
    dequantizationShiftsLayers.resize(parentsCount);

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

        const float generalScaleDequantize = maxOutputInterval / (dataPrecision.max - dataPrecision.min);

        for (int index = 0; index < parentsCount; index++) {
            if (quantizeLayers[index] == nullptr)
                continue;
            CNNLayer& fakeQuantizeLayer = *quantizeLayers[index];
            const QuantizationDetails quantizationDetails = quantizationLayersDetails[index];

            // TODO: copy/paste, refactor: extract to MultiBranchTransformation::updateQuantizationRange
            switch (quantizedTensorAlignmentOnActivations) {
            case QuantizedTensorAlignment::None: {
                const float quantizationScale = (dataPrecision.max - dataPrecision.min) / maxOutputInterval;

                // if there is negative part then calculation is based on `outputLowValue` if not then on `outputHighValue` only
                const float updatedOutputLowValue = dataPrecision.precision.isSigned() ?
                    (outputLowValue != 0.f ?
                    (quantizationDetails.outputLowValues[0] / outputLowValue) * dataPrecision.min :
                        (quantizationDetails.outputLowValues[0] / outputHighValue) * dataPrecision.max) :
                        (quantizationDetails.outputLowValues[0] + fabs(outputLowValue)) * quantizationScale;
                CNNNetworkHelper::updateBlobs(fakeQuantizeLayer, 3, updatePrecisions ? roundf(updatedOutputLowValue) : updatedOutputLowValue);

                // if there is positive part then calculation is based on `outputHighValue` if not then on `outputLowValue` only
                const float updatedOutputHighValue = dataPrecision.precision.isSigned() ?
                    (outputHighValue != 0.f ?
                    (quantizationDetails.outputHighValues[0] / outputHighValue) * dataPrecision.max :
                        (quantizationDetails.outputHighValues[0] / outputLowValue) * dataPrecision.min) :
                        (quantizationDetails.outputHighValues[0] + fabs(outputLowValue)) * quantizationScale;
                CNNNetworkHelper::updateBlobs(fakeQuantizeLayer, 4, updatePrecisions ? roundf(updatedOutputHighValue) : updatedOutputHighValue);
                break;
            }
            case QuantizedTensorAlignment::UpdateIntervals: {
                const float inputLowValue = quantizationDetails.outputLowValues[0] != 0.f ?
                    (quantizationDetails.inputLowValues[0] * (outputLowValue / quantizationDetails.outputLowValues[0])) :
                    outputLowValue;
                const float inputHighValue = quantizationDetails.outputHighValues[0] != 0.f ?
                    (quantizationDetails.inputHighValues[0] * (outputHighValue / quantizationDetails.outputHighValues[0])) :
                    outputHighValue;

                CNNNetworkHelper::updateBlobs(fakeQuantizeLayer, 1, inputLowValue);
                CNNNetworkHelper::updateBlobs(fakeQuantizeLayer, 2, inputHighValue);
                CNNNetworkHelper::updateBlobs(fakeQuantizeLayer, 3, dataPrecision.min);
                CNNNetworkHelper::updateBlobs(fakeQuantizeLayer, 4, dataPrecision.max);
                break;
            }
            case QuantizedTensorAlignment::UpdateLevel: {
                const float quantizationScale = (dataPrecision.max - dataPrecision.min) / maxOutputInterval;

                // if there is negative part then calculation is based on `outputLowValue` if not then on `outputHighValue` only
                const float updatedOutputLowValue = dataPrecision.precision.isSigned() ?
                    (outputLowValue != 0.f ?
                        (quantizationDetails.outputLowValues[0] / outputLowValue) * dataPrecision.min :
                        (quantizationDetails.outputLowValues[0] / outputHighValue) * dataPrecision.max) :
                    (quantizationDetails.outputLowValues[0] + fabs(outputLowValue)) * quantizationScale;
                CNNNetworkHelper::updateBlobs(fakeQuantizeLayer, 3, updatePrecisions ? roundf(updatedOutputLowValue) : updatedOutputLowValue);

                // if there is positive part then calculation is based on `outputHighValue` if not then on `outputLowValue` only
                const float updatedOutputHighValue = dataPrecision.precision.isSigned() ?
                    (outputHighValue != 0.f ?
                        (quantizationDetails.outputHighValues[0] / outputHighValue) * dataPrecision.max :
                        (quantizationDetails.outputHighValues[0] / outputLowValue) * dataPrecision.min) :
                    (quantizationDetails.outputHighValues[0] + fabs(outputLowValue)) * quantizationScale;
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

                const std::vector<CNNLayerPtr>& intermediateLayersList = intermediateLayers[index];
                for (const CNNLayerPtr intermediateLayer : intermediateLayersList) {
                    CNNNetworkHelper::setOutDataPrecision(*intermediateLayer, dataPrecision.precision);
                }
            }

            dequantizationShiftsLayers[index].push_back(outputLowValue);
        }

        dequantizationScales.resize(outputChannelsCount);
        std::fill(dequantizationScales.begin(), dequantizationScales.end(), generalScaleDequantize);

        dequantizationShifts.resize(outputChannelsCount);
        // assumption: if interval is signed then shift is not required
        std::fill(dequantizationShifts.begin(), dequantizationShifts.end(), dataPrecision.precision.isSigned() ? 0.f : outputLowValue);
    } else {
        return;
    }

    // Add scaleshift at other outputs of the Quantize layer
    for (int index = 0; index < parentsCount; index++) {
        context.quantizedFakeQuantizeNames.insert(quantizeLayers[index]->name);
        if (quantizeLayers[index]->outData[0]->getInputTo().size() != 1) {
            std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildren(*quantizeLayers[index], childNameOurAfterQuantizeLayers[index]);

            for (int i = 0; i < children.size(); i++) {
                const size_t outputChannelsCount = CNNNetworkHelper::getOutputChannelsCount(*quantizeLayers[index]);
                const std::vector<float> branchDequantizationScales(outputChannelsCount, dequantizationScales[0]);
                const std::vector<float> branchDequantizationShifts(outputChannelsCount, dequantizationShiftsLayers[index][0]);
                CNNLayerPtr dequantizationLayer = CNNNetworkHelper::addScaleShiftBetween(
                    context,
                    std::make_shared<CNNLayer>(*quantizeLayers[index]),
                    children[i],
                    DequantizationDetails(branchDequantizationScales, branchDequantizationShifts, outputChannelsCount));
                context.dequantizationLayersNames.insert(dequantizationLayer->name);
            }
        }

        const std::vector<CNNLayerPtr>& intermediateLayersList = intermediateLayers[index];
        for (size_t i = 0ul; i < intermediateLayersList.size(); ++i) {
            const CNNLayerPtr intermediateLayer = intermediateLayersList[i];
            const CNNLayer& nextIntermediateLayer = (i + 1) < intermediateLayersList.size() ? *intermediateLayersList[i] : concat;
            std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildren(*intermediateLayer, nextIntermediateLayer.name);

            for (int i = 0; i < children.size(); i++) {
                const size_t outputChannelsCount = CNNNetworkHelper::getOutputChannelsCount(*quantizeLayers[index]);
                const std::vector<float> branchDequantizationScales(outputChannelsCount, dequantizationScales[0]);
                const std::vector<float> branchDequantizationShifts(outputChannelsCount, dequantizationShiftsLayers[index][0]);
                CNNLayerPtr dequantizationLayer = CNNNetworkHelper::addScaleShiftBetween(
                    context,
                    intermediateLayer,
                    children[i],
                    DequantizationDetails(branchDequantizationScales, branchDequantizationShifts, outputChannelsCount));
                context.dequantizationLayersNames.insert(dequantizationLayer->name);
            }
        }
    }

    if (updatePrecisions) {
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
