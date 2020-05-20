// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/interp.hpp"

#include <algorithm>
#include <details/caseless.hpp>
#include <string>
#include <memory>
#include <vector>

#include "low_precision_transformations/common/ie_lpt_exception.hpp"
#include "low_precision_transformations/network_helper.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

bool InterpTransformation::canBeTransformed(const TransformationContext& context, const CNNLayer& layer) const {
    if (!LayerTransformation::canBeTransformed(context, layer)) {
        return false;
    }

    if (layer.insData.size() != 1) {
        THROW_IE_LPT_EXCEPTION(layer) << "layer inputs '" << layer.insData.size() << "' is not correct";
    }

    if (!CaselessEq<std::string>()(layer.type, "Interp")) {
        THROW_IE_LPT_EXCEPTION(layer) << "layer '" << layer.name << "' is not correct";
    }

    const CNNLayerPtr parent = CNNNetworkHelper::getParent(layer, 0);
    return !(parent->type != "ScaleShift");
}

void InterpTransformation::transform(TransformationContext& context, CNNLayer& layer) const {
    if (!canBeTransformed(context, layer)) {
        return;
    }

    const CNNLayerPtr parent = CNNNetworkHelper::getParent(layer, 0);

    std::vector<float> dequantizationScales;
    std::vector<float> dequantizationShifts;
    fillFromDequantizationLayer(*parent, dequantizationScales, dequantizationShifts);

    CNNNetworkHelper::removeLayer(context.network, parent);
    context.removeLayer(*parent);

    addDequantizationLayer(context, layer, dequantizationScales, dequantizationShifts);
}
