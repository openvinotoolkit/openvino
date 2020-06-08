// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/normalize.hpp"

#include <algorithm>
#include <string>
#include <memory>
#include <vector>

#include <details/caseless.hpp>
#include "low_precision_transformations/common/ie_lpt_exception.hpp"
#include "low_precision_transformations/network_helper.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

bool NormalizeTransformation::canBeTransformed(const TransformationContext& context, const CNNLayer& layer) const {
    if (!LayerTransformation::canBeTransformed(context, layer)) {
        return false;
    }

    if (layer.insData.size() != 1) {
        THROW_IE_LPT_EXCEPTION(layer) << "layer inputs '" << layer.insData.size() << "' is not correct";
    }

    if (!CaselessEq<std::string>()(layer.type, "Normalize")) {
        THROW_IE_LPT_EXCEPTION(layer) << "layer '" << layer.name << "' is not correct";
    }

    const CNNLayerPtr parent = CNNNetworkHelper::getParent(layer, 0);
    return (parent->type == "ScaleShift");
}

void NormalizeTransformation::transform(TransformationContext& context, CNNLayer& layer) const {
    if (!canBeTransformed(context, layer)) {
        return;
    }

    const CNNLayerPtr scaleShift = CNNNetworkHelper::getParent(layer, 0);
    std::vector<float> originalDequantizationScales;
    std::vector<float> originalDequantizationShifts;
    fillFromDequantizationLayer(*scaleShift, originalDequantizationScales, originalDequantizationShifts);

    const bool acrossSpatial = layer.GetParamAsBool("across_spatial");
    if (std::any_of(originalDequantizationShifts.begin(), originalDequantizationShifts.end(), [](const float value) { return value != 0.f; })) {
        return;
    }

    if (acrossSpatial &&
        std::any_of(
            originalDequantizationScales.begin(),
            originalDequantizationScales.end(),
            [&](const float value) { return value != originalDequantizationScales[0]; })) {
        return;
    }

    std::vector<float> dequantizationScales(originalDequantizationScales.size());
    std::vector<float> dequantizationShifts(originalDequantizationShifts.size(), 0.f);
    for (size_t channel = 0ul; channel < dequantizationScales.size(); ++channel) {
        dequantizationScales[channel] = std::signbit(originalDequantizationScales[channel]) ? -1.f : 1.f;
    }

    CNNNetworkHelper::removeLayer(context.network, scaleShift);
    context.removeLayer(*scaleShift);

    addDequantizationLayer(context, layer, dequantizationScales, dequantizationShifts);
}

bool NormalizeTransformation::isPrecisionPreserved(const CNNLayer& layer) const noexcept {
    return false;
}
