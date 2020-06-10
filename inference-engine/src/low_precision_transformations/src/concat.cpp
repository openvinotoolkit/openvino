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
#include "low_precision_transformations/quantization_details.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

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

    Subgraph subgraph = CNNNetworkHelper::getSubgraph(concat);
    if (subgraph.empty()) {
        return;
    }

    for (const CNNLayerPtr& quantizationLayer : subgraph.quantizationLayers) {
        if (context.quantizedFakeQuantizeNames.find(quantizationLayer->name) != context.quantizedFakeQuantizeNames.end()) {
            return;
        }
    }

    DataPrecision dataPrecision = getDataPrecision(
        *subgraph.quantizationLayers[0],
        QuantizationDetails::getDetails(*subgraph.quantizationLayers[0]), false, false);
    if (dataPrecision.precision == Precision::UNSPECIFIED) {
        return;
    }


    // TODO: FQ output I8 but Convolution U8 before <- we should handle that avoid asymmetric quantization

    std::vector<QuantizationDetails> quantizationLayersDetails;
    size_t quantizationLevels = 0lu;
    for (int i = 0; i < subgraph.quantizationLayers.size(); i++) {
        const QuantizationDetails& quantizationDetails = QuantizationDetails::getDetails(*subgraph.quantizationLayers[i]);
        if (!QuantizationDetails::isSupportedLevel(quantizationDetails.levels)) continue;
        if (quantizationLevels == 0lu) {
            quantizationLevels = quantizationDetails.levels;
        } else if (quantizationLevels != quantizationDetails.levels) {
            THROW_IE_EXCEPTION << "different quantization levels " << quantizationLevels << " are not supported";
        }

        quantizationLayersDetails.push_back(quantizationDetails);

        const DataPrecision dataPrecision2 = getDataPrecision(*subgraph.quantizationLayers[i], quantizationDetails, false, false);
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


    float dequantizationScale;
    float dequantizationShift;

    if ((quantizationLayersDetails[0].inputHighValues.size() == 1)) {
        float outputLowValue = quantizationLayersDetails[0].outputLowValues[0];
        float outputHighValue = quantizationLayersDetails[0].outputHighValues[0];

        for (size_t index = 0lu; index < subgraph.quantizationLayers.size(); index++) {
            const QuantizationDetails& quantizationDetails = quantizationLayersDetails[index];
            if (outputLowValue > quantizationDetails.outputLowValues[0]) {
                outputLowValue = quantizationDetails.outputLowValues[0];
            }
            if (outputHighValue < quantizationDetails.outputHighValues[0]) {
                outputHighValue = quantizationDetails.outputHighValues[0];
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


        dequantizationScale = maxOutputInterval / (dataPrecision.max - dataPrecision.min);
        const float max = maxOutputInterval / ((dataPrecision.max - dataPrecision.min) / dataPrecision.max);
        const float min = maxOutputInterval / ((dataPrecision.max - dataPrecision.min) / dataPrecision.min);
        dequantizationShift = outputLowValue - min;

        const float quantizationScale = 1.f / dequantizationScale;
        const float quantizationShift = - dequantizationShift * quantizationScale;

        for (int index = 0; index < subgraph.quantizationLayers.size(); index++) {
            CNNLayer& fakeQuantizeLayer = *subgraph.quantizationLayers[index];
            const QuantizationDetails& quantizationDetails = quantizationLayersDetails[index];

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
        }
    } else {
        return;
    }

    if (updatePrecisions) {
        for (const auto it : subgraph.layers) {
            const CNNLayer* layer = it.second;
            CNNNetworkHelper::setOutDataPrecision(*layer, dataPrecision.precision);
        }
    }

    auto dequantizationValuesCallback = [&](
        const CNNLayer& layer,
        const std::string& originalLayerName,
        std::vector<float>& layerDequantizationScales,
        std::vector<float>& layerDequantizationShifts
        ) {
        const size_t outputChannelsCount = CNNNetworkHelper::getOutputChannelsCount(layer);

        layerDequantizationScales.resize(outputChannelsCount);
        std::fill(layerDequantizationScales.begin(), layerDequantizationScales.end(), dequantizationScale);

        layerDequantizationShifts.resize(outputChannelsCount);
        std::fill(layerDequantizationShifts.begin(), layerDequantizationShifts.end(), dequantizationShift);
    };

    addDequantizationLayers(context, subgraph, dequantizationValuesCallback);

    for (const CNNLayerPtr& quantizationLayer : subgraph.quantizationLayers) {
        context.quantizedFakeQuantizeNames.insert(quantizationLayer->name);
    }
}

void ConcatTransformation::addDequantizationLayers(
    TransformationContext& context,
    Subgraph& subgraph,
    std::function<void(
        const CNNLayer& layer,
        const std::string& originalLayerName,
        std::vector<float>& dequantizationScales,
        std::vector<float>& dequantizationShifts)> getLayerDequantizationCallback) const {
    OutputsDataMap outputs;
    context.network.getOutputsInfo(outputs);

    std::unordered_map<std::string, CNNLayer*> notHandledSubgraphLayers = subgraph.layers;
    while (notHandledSubgraphLayers.size() != 0ul) {
        const auto layerIt = notHandledSubgraphLayers.begin();
        CNNLayer* layer = layerIt->second;
        notHandledSubgraphLayers.erase(layerIt);

        std::vector<float> layerDequantizationScales;
        std::vector<float> layerDequantizationShifts;

        const std::vector<CNNLayerPtr>& children = CNNNetworkHelper::getChildren(*layer);
        for (const CNNLayerPtr& child : children) {
            if (subgraph.layers.find(child->name) == subgraph.layers.end()) {
                if (layerDequantizationScales.size() == 0ul) {
                    getLayerDequantizationCallback(*layer, layer->name, layerDequantizationScales, layerDequantizationShifts);
                }

                CNNLayerPtr dequantizationLayer = CNNNetworkHelper::addScaleShiftBetween(
                    context,
                    std::make_shared<CNNLayer>(*layer),
                    child,
                    DequantizationDetails(layerDequantizationScales, layerDequantizationShifts, layerDequantizationScales.size()));
                context.dequantizationLayersNames.insert(dequantizationLayer->name);
            }
        }

        const auto it = outputs.find(layer->name);
        if (it != outputs.end()) {
            const std::string originalName = layer->name;
            const std::string newName = layer->name + LayerTransformation::lastLayerPostfix;
            CNNNetworkHelper::renameLayer(context.network, originalName, newName);

            layer->name = newName;
            subgraph.layers[layer->name] = layer;

            if (layerDequantizationScales.size() == 0ul) {
                getLayerDequantizationCallback(*layer, originalName, layerDequantizationScales, layerDequantizationShifts);
            }

            CNNLayerPtr dequantizationLayer = CNNNetworkHelper::addScaleShiftBetween(
                context,
                std::make_shared<CNNLayer>(*layer),
                nullptr,
                DequantizationDetails(layerDequantizationScales, layerDequantizationShifts, layerDequantizationScales.size()),
                originalName);
            context.dequantizationLayersNames.insert(dequantizationLayer->name);
            subgraph.layers[dequantizationLayer->name] = dequantizationLayer.get();
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
