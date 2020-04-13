// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/activation.hpp"
#include "low_precision_transformations/network_helper.hpp"

#include <algorithm>
#include <details/caseless.hpp>
#include <memory>
#include <string>
#include <vector>

using namespace InferenceEngine;
using namespace InferenceEngine::details;

void ActivationTransformation::transform(TransformationContext& context, CNNLayer& layer) const {
    if (!CaselessEq<std::string>()(layer.type, "ReLU")) {
        THROW_IE_EXCEPTION << "layer type '" << layer.name << "' is not correct";
    }

    const CNNLayerPtr scaleShift = CNNNetworkHelper::getParent(layer, 0);
    if ((scaleShift == nullptr) || (scaleShift->type != "ScaleShift")) {
        return;
    }

    // TODO: temporary limitation
    if (scaleShift->insData.size() != 1) {
        return;
    }

    const Blob::Ptr weightsBlob = CNNNetworkHelper::getBlob(scaleShift, "weights");
    auto weights = CNNNetworkHelper::getFloatData(weightsBlob);
    const std::vector<float> scales = std::vector<float>(weights.get(), weights.get() + weightsBlob->size());

    const Blob::Ptr biasesBlob = CNNNetworkHelper::getBlob(scaleShift, "biases");
    auto biases = CNNNetworkHelper::getFloatData(biasesBlob);
    const std::vector<float> shifts = std::vector<float>(biases.get(), biases.get() + biasesBlob->size());

    CNNLayerPtr activationLayer;
    if ((std::all_of(shifts.begin(), shifts.end(),
                     [](float value) {
                         return value == 0.0;
                     })) &&
        (std::all_of(scales.begin(), scales.end(), [](float value) {
            return value >= 0.0;
        }))) {
        activationLayer = std::make_shared<CNNLayer>(layer);
    } else {
        const float negativeSlope = layer.GetParamAsFloat("negative_slope", 0.0);
        if (negativeSlope != 0.0) {
            return;
        }

        if (!(std::equal(shifts.begin() + 1, shifts.end(), shifts.begin())) ||
            !(std::equal(scales.begin() + 1, scales.end(), scales.begin()))) {
            return;
        }

        const Precision precision = getPrecisionBeforeParentDequantizationScaleShift(layer);

        std::vector<CNNLayerPtr> parents = CNNNetworkHelper::getParents(*scaleShift);
        if (parents.size() != 1) {
            return;
        }

        LayerParams layerParams {layer.name + "_Clamp", "Clamp", precision};
        activationLayer = std::make_shared<ClampLayer>(layerParams);

        ClampLayer* clampLayer = dynamic_cast<ClampLayer*>(activationLayer.get());
        if (std::all_of(scales.begin(), scales.end(), [](float value) {
                return value >= 0.0;
            })) {
            clampLayer->min_value = -shifts[0] / scales[0];
            clampLayer->max_value = DataPrecision::getMaxValue(precision);
            clampLayer->params["min"] = CNNLayer::ie_serialize_float(clampLayer->min_value);
            clampLayer->params["max"] = CNNLayer::ie_serialize_float(clampLayer->max_value);
        } else {
            // TODO: workaround: only U8 on activations
            clampLayer->min_value = DataPrecision::getMinValue(precision, 256);
            clampLayer->max_value = -shifts[0] / scales[0];
            clampLayer->params["min"] = CNNLayer::ie_serialize_float(clampLayer->min_value);
            clampLayer->params["max"] = CNNLayer::ie_serialize_float(clampLayer->max_value);
        }

        std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildren(layer);
        if (children.size() != 1) {
            return;
        }

        for (CNNLayerPtr child : children) {
            CNNNetworkHelper::addLayer(context, std::make_shared<CNNLayer>(layer), child, activationLayer);
        }

        CNNNetworkHelper::removeLayer(context.network, std::make_shared<CNNLayer>(layer));
        context.removeLayer(layer);
    }

    if (updatePrecisions) {
        CNNNetworkHelper::setOutDataPrecision(layer, getPrecisionBeforeParentDequantizationScaleShift(layer));
    }

    CNNNetworkHelper::removeLayer(context.network, scaleShift);
    context.removeLayer(*scaleShift);

    const std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildren(*activationLayer);
    for (const CNNLayerPtr& child : children) {
        CNNLayerPtr dequantizationLayer = CNNNetworkHelper::addScaleShiftBetween(context, activationLayer, child,
                                                                                 DequantizationDetails(scales, shifts));
        context.dequantizationLayersNames.insert(dequantizationLayer->name);
    }
}
