// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/resample.hpp"
#include "low_precision_transformations/network_helper.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

using namespace InferenceEngine;
using namespace InferenceEngine::details;

void ResampleTransformation::transform(TransformationContext& context, CNNLayer& layer) const {
    if (!LayerTransformation::canBeTransformed(context, layer)) {
        return;
    }

    const std::vector<CNNLayerPtr> parents = CNNNetworkHelper::getParents(layer);
    if (parents.size() != 1ul) {
        THROW_IE_EXCEPTION << "unexpected input layers count " << parents.size();
    }

    if (parents[0]->type != "ScaleShift") {
        return;
    }

    const std::string type = layer.GetParamAsString("type", "");
    if (type != "caffe.ResampleParameter.NEAREST") {
        return;
    }

    const Precision precision = getPrecisionBeforeParentDequantizationScaleShift(layer);

    std::vector<float> dequantizationScales;
    std::vector<float> dequantizationShifts;
    fillFromDequantizationLayer(*parents[0], dequantizationScales, dequantizationShifts);

    // transparent base transformation
    CNNNetworkHelper::removeLayer(context.network, parents[0]);
    context.removeLayer(*parents[0]);

    if (updatePrecisions) {
        CNNNetworkHelper::setOutDataPrecision(layer, precision);
    }

    addDequantizationLayer(context, layer, dequantizationScales, dequantizationShifts);
}
