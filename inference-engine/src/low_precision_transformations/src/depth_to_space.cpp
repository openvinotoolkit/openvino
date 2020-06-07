// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/depth_to_space.hpp"

#include <algorithm>
#include <details/caseless.hpp>
#include <memory>
#include <string>
#include <vector>

#include "low_precision_transformations/common/ie_lpt_exception.hpp"
#include "low_precision_transformations/network_helper.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

void DepthToSpaceTransformation::transform(TransformationContext& context, CNNLayer& layer) const {
    if (!canBeTransformed(context, layer)) {
        return;
    }

    if ((layer.insData.size() == 0) || layer.insData.size() > 2) {
        THROW_IE_EXCEPTION << "layer inputs '" << layer.insData.size() << "' is not correct";
    }

    if (!CaselessEq<std::string>()(layer.type, "DepthToSpace")) {
        THROW_IE_EXCEPTION << "layer '" << layer.name << "' is not correct";
    }

    TransparentBaseTransformation::transform(context, layer);
}

bool DepthToSpaceTransformation::isPrecisionPreserved(const CNNLayer& layer) const noexcept {
    return true;
}

bool DepthToSpaceTransformation::canBeTransformed(const TransformationContext& context, const CNNLayer& layer) const {
    if (!TransparentBaseTransformation::canBeTransformed(context, layer)) {
        return false;
    }

    const std::vector<CNNLayerPtr> parents = CNNNetworkHelper::getParents(layer);
    if (parents.size() != 1) {
        return false;
    }

    if (parents[0]->type != "ScaleShift") {
        return false;
    }

    const std::vector<size_t> inputDims = parents[0]->outData[0]->getDims();
    if (inputDims.size() < 3) {
        return false;
    }

    const size_t inputChannels = CNNNetworkHelper::getInputChannelsCount(layer);
    const size_t outputChannels = CNNNetworkHelper::getOutputChannelsCount(layer);
    if (inputChannels != outputChannels) {
        std::vector<float> scales;
        std::vector<float> shifts;
        fillFromDequantizationLayer(*parents[0], scales, shifts);

        if (!DequantizationDetails::isPerTensor(scales, shifts)) {
            return false;
        }
    }

    return true;
}
