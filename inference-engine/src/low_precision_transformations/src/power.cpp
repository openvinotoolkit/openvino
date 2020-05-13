// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/power.hpp"

#include <algorithm>
#include <details/caseless.hpp>
#include <string>
#include <memory>
#include <vector>

#include "low_precision_transformations/common/ie_lpt_exception.hpp"
#include "low_precision_transformations/network_helper.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

bool PowerTransformation::canBeTransformed(const TransformationContext& context, const CNNLayer& layer) const {
    if (!LayerTransformation::canBeTransformed(context, layer)) {
        return false;
    }

    if (layer.insData.size() != 1) {
        THROW_IE_LPT_EXCEPTION(layer) << "layer inputs '" << layer.insData.size() << "' is not correct";
    }

    if (!CaselessEq<std::string>()(layer.type, "Power")) {
        THROW_IE_LPT_EXCEPTION(layer) << "layer '" << layer.name << "' is not correct";
    }

    const PowerLayer* powerLayer = dynamic_cast<const PowerLayer*>(&layer);
    if (powerLayer == nullptr) {
        THROW_IE_LPT_EXCEPTION(layer) << "unexpected Power layer type";
    }
    if (powerLayer->power != 1.f) {
        return false;
    }

    const CNNLayerPtr parent = CNNNetworkHelper::getParent(layer, 0);
    return !(parent->type != "ScaleShift");
}

void PowerTransformation::transform(TransformationContext& context, CNNLayer& layer) const {
    if (!canBeTransformed(context, layer)) {
        return;
    }

    const PowerLayer* powerLayer = dynamic_cast<const PowerLayer*>(&layer);
    if (powerLayer == nullptr) {
        THROW_IE_LPT_EXCEPTION(layer) << "unexpected Power layer type";
    }

    const CNNLayerPtr parent = CNNNetworkHelper::getParent(layer, 0);

    Blob::Ptr weightsBlob = CNNNetworkHelper::getBlob(parent, "weights");
    auto wBuffer = weightsBlob->buffer().as<float*>();
    for (size_t channel = 0ul; channel < weightsBlob->size(); ++channel) {
        wBuffer[channel] = wBuffer[channel] * powerLayer->scale;
    }

    Blob::Ptr shiftsBlob = CNNNetworkHelper::getBlob(parent, "biases");
    auto sBuffer = shiftsBlob->buffer().as<float*>();
    for (size_t channel = 0ul; channel < shiftsBlob->size(); ++channel) {
        sBuffer[channel] = sBuffer[channel] * powerLayer->scale + powerLayer->offset;
    }

    const std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildren(layer);
    CNNNetworkHelper::removeLayer(context.network, std::make_shared<CNNLayer>(layer));
    context.removeLayer(layer);
    if (children.empty()) {
        const std::string originalName = layer.name;
        CNNNetworkHelper::renameLayer(context.network, parent->name, layer.name);
    }
}
