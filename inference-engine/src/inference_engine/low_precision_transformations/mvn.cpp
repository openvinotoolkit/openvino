// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mvn.hpp"

#include <algorithm>
#include <details/caseless.hpp>
#include <memory>
#include <string>
#include <vector>

#include "network_helper.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

void MvnTransformation::transform(TransformationContext& context, CNNLayer& layer) const {
    if (!LayerTransformation::canBeTransformed(context, layer)) {
        return;
    }

    if (!CaselessEq<std::string>()(layer.type, "MVN")) {
        THROW_IE_EXCEPTION << "Layer '" << layer.name << "' has invalid type '" << layer.type << "'. Convolution is expected.";
    }

    const CNNLayerPtr scaleShiftOnData = CNNNetworkHelper::getParent(layer, 0);
    if (scaleShiftOnData->type != "ScaleShift") {
        return;
    }

    std::vector<float> originalDataDequantizationScales;
    std::vector<float> originalDataDequantizationShifts;
    fillFromDequantizationLayer(*scaleShiftOnData, originalDataDequantizationScales, originalDataDequantizationShifts);
    if (std::any_of(originalDataDequantizationShifts.begin(), originalDataDequantizationShifts.end(), [](const float value) { return value != 0.f; })) {
        return;
    }

    const size_t acrossChannels = layer.GetParamAsUInt("across_channels", 0ul);
    if ((acrossChannels == 1ul) &&
        std::any_of(
        originalDataDequantizationScales.begin(),
        originalDataDequantizationScales.end(),
        [&](const float value) { return value != originalDataDequantizationScales[0]; })) {
        return;
    }

    const size_t normalizeVariance = layer.GetParamAsUInt("normalize_variance", 0ul);

    std::vector<float> dequantizationScales(originalDataDequantizationScales.size());
    std::vector<float> dequantizationShifts(originalDataDequantizationShifts.size(), 0.f);

    for (size_t channel = 0ul; channel < dequantizationScales.size(); ++channel) {
        dequantizationScales[channel] = normalizeVariance == 0ul ?
            originalDataDequantizationScales[channel] :
            std::signbit(originalDataDequantizationScales[channel]) ? -1.f : 1.f;
    }

    CNNNetworkHelper::removeLayer(context.network, scaleShiftOnData);
    context.removeLayer(*scaleShiftOnData);

    const size_t outputChannelsCount = CNNNetworkHelper::getOutputChannelsCount(layer);
    const std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildren(layer);
    if (children.size() == 0) {
        const std::string originalName = layer.name;
        CNNNetworkHelper::renameLayer(context.network, layer.name, layer.name + LayerTransformation::lastLayerPrefix);

        const CNNLayerPtr dequantizationLayer = CNNNetworkHelper::addScaleShiftBetween(
            context,
            std::make_shared<CNNLayer>(layer),
            nullptr,
            DequantizationDetails(dequantizationScales, dequantizationShifts, outputChannelsCount),
            originalName);
        context.dequantizationLayersNames.insert(dequantizationLayer->name);
    } else {
        for (const CNNLayerPtr& child : children) {
            const CNNLayerPtr dequantizationLayer = CNNNetworkHelper::addScaleShiftBetween(
                context,
                std::make_shared<CNNLayer>(layer),
                child,
                DequantizationDetails(dequantizationScales, dequantizationShifts, outputChannelsCount));
            context.dequantizationLayersNames.insert(dequantizationLayer->name);
        }
    }
}

bool MvnTransformation::isPrecisionPreserved(const CNNLayer& layer) const noexcept {
    return false;
}
