// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/transparent_base_transformation.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "low_precision_transformations/common/ie_lpt_exception.hpp"
#include "low_precision_transformations/network_helper.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

void TransparentBaseTransformation::transform(TransformationContext& context, CNNLayer& layer) const {
    const CNNLayerPtr scaleShift = CNNNetworkHelper::getParent(layer, 0);
    if (scaleShift == nullptr) {
        return;
    }

    if (scaleShift->type == "Concat") {
        if (updatePrecisions) {
            // TODO: looks like as workaround for Concat -> Pooling -> Concat: refactor later
            CNNNetworkHelper::setOutDataPrecision(layer, CNNNetworkHelper::getPrecisionParent(layer, 0ul));
        }
    } else if (scaleShift->type == "ScaleShift") {
        if (updatePrecisions) {
            CNNNetworkHelper::setOutDataPrecision(layer, getPrecisionBeforeParentDequantizationScaleShift(layer));
        }

        std::vector<float> scales;
        std::vector<float> shifts;
        fillFromDequantizationLayer(*scaleShift, scales, shifts);

        const size_t outputChannelsCount = CNNNetworkHelper::getOutputChannelsCount(layer);
        if (outputChannelsCount != CNNNetworkHelper::getInputChannelsCount(layer)) {
            if (!DequantizationDetails::isPerTensor(scales, shifts)) {
                THROW_IE_LPT_EXCEPTION(layer) << "input and output channels count values are different for per channel quantization";
            }
            scales = std::vector<float>(outputChannelsCount, scales[0]);
            shifts = std::vector<float>(outputChannelsCount, shifts[0]);
        }

        CNNNetworkHelper::removeLayer(context.network, scaleShift);
        context.removeLayer(*scaleShift);

        addDequantizationLayer(context, layer, scales, shifts);
    }
}
