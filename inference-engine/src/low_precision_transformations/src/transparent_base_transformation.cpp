// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/transparent_base_transformation.hpp"
#include "low_precision_transformations/network_helper.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

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

        const Blob::Ptr weights_blob = CNNNetworkHelper::getBlob(scaleShift, "weights");
        auto weights = CNNNetworkHelper::getFloatData(weights_blob);
        const std::vector<float> scales = std::vector<float>(weights.get(), weights.get() + weights_blob->size());

        const Blob::Ptr biases_blob = CNNNetworkHelper::getBlob(scaleShift, "biases");
        auto biases = CNNNetworkHelper::getFloatData(biases_blob);
        const std::vector<float> shifts = std::vector<float>(biases.get(), biases.get() + biases_blob->size());

        CNNNetworkHelper::removeLayer(context.network, scaleShift);
        context.removeLayer(*scaleShift);

        const std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildren(layer);
        if (children.size() == 0) {
            const std::string originalName = layer.name;
            CNNNetworkHelper::renameLayer(context.network, layer.name, layer.name + LayerTransformation::lastLayerPrefix);

            CNNLayerPtr dequantizationLayer = CNNNetworkHelper::addScaleShiftBetween(
                context,
                std::make_shared<CNNLayer>(layer),
                nullptr,
                DequantizationDetails(scales, shifts),
                originalName);
            context.dequantizationLayersNames.insert(dequantizationLayer->name);
        } else {
            for (const CNNLayerPtr& child : children) {
                CNNLayerPtr dequantizationLayer = CNNNetworkHelper::addScaleShiftBetween(
                    context,
                    std::make_shared<CNNLayer>(layer),
                    child,
                    DequantizationDetails(scales, shifts));
                context.dequantizationLayersNames.insert(dequantizationLayer->name);
            }
        }
    }
}
