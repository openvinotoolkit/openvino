// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/permute.hpp"

#include <algorithm>
#include <caseless.hpp>
#include <string>
#include <vector>

#include "low_precision_transformations/network_helper.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

void PermuteTransformation::transform(TransformationContext& context, CNNLayer& layer) const {
    if (!canBeTransformed(context, layer)) {
        return;
    }

    if (layer.insData.size() != 1) {
        THROW_IE_EXCEPTION << "layer inputs '" << layer.insData.size() << "' is not correct";
    }

    if (!CaselessEq<std::string>()(layer.type, "Permute")) {
        THROW_IE_EXCEPTION << "layer '" << layer.name << "' is not correct";
    }

    if (!layer.CheckParamPresence("order")) {
        THROW_IE_EXCEPTION << "Permute parameter 'order' is absent";
    }

    const CNNLayerPtr scaleShift = CNNNetworkHelper::getParent(layer);
    if ((scaleShift == nullptr) || (scaleShift->type != "ScaleShift")) {
        return;
    }

    std::vector<float> dequantizationScales;
    std::vector<float> dequantizationShifts;
    fillFromDequantizationLayer(*scaleShift, dequantizationScales, dequantizationShifts);

    if (!DequantizationDetails::isPerTensor(dequantizationScales, dequantizationShifts)) {
        std::vector<unsigned int> orders = layer.GetParamAsUInts("order");
        if ((orders.size() < 2) || (orders[0] != 0U) || (orders[1] != 1U)) {
            return;
        }
    }

    TransparentBaseTransformation::transform(context, layer);
}
