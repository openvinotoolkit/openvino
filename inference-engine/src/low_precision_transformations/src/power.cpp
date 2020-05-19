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

    auto scale_and_shift_blob = [] (Blob::Ptr &&blob, float scale, float shift) {
        auto float_data = CNNNetworkHelper::getFloatData(blob);
        auto float_data_ptr = float_data.get();
        auto float_data_size = blob->size();

        for (size_t i = 0ul; i < float_data_size; i++) {
            float_data_ptr[i] = float_data_ptr[i] * scale + shift;
        }

        CNNNetworkHelper::fillBlobByFP32(blob, float_data_ptr);
    };

    const CNNLayerPtr parent = CNNNetworkHelper::getParent(layer, 0);

    scale_and_shift_blob(CNNNetworkHelper::getBlob(parent, "weights"), powerLayer->scale, 0.0f);
    scale_and_shift_blob(CNNNetworkHelper::getBlob(parent, "biases") , powerLayer->scale, powerLayer->offset);

    const std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildren(layer);
    CNNNetworkHelper::removeLayer(context.network, std::make_shared<CNNLayer>(layer));
    context.removeLayer(layer);
    if (children.empty()) {
        const std::string originalName = layer.name;
        CNNNetworkHelper::renameLayer(context.network, parent->name, layer.name);
    }
}
