// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/const.hpp"
#include "low_precision_transformations/network_helper.hpp"

#include <ie_common.h>

#include <algorithm>
#include <blob_factory.hpp>
#include <cmath>
#include <caseless.hpp>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <legacy/cnn_network_impl.hpp>
#include <legacy/ie_util_internal.hpp>

using namespace InferenceEngine;
using namespace InferenceEngine::details;

void ConstTransformation::transform(TransformationContext& context, CNNLayer& layer) const {
    if (!canBeTransformed(context, layer)) {
        return;
    }

    if (!CaselessEq<std::string>()(layer.type, "Const")) {
        THROW_IE_EXCEPTION << "layer type '" << layer.name << "' is not correct";
    }

    if ((layer.insData.size() != 0) || (layer.outData.size() != 1)) {
        return;
    }

    const std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildren(layer);
    if (!CNNNetworkHelper::IsChild(children, {"FakeQuantize"})) {
        return;
    }
    if (children.size() != 1) {
        THROW_IE_EXCEPTION << "unexpected children count " << children.size();
    }

    const auto fakeQuantize = children[0];
    const CNNLayerPtr inputLayer = CNNNetworkHelper::getParent(*fakeQuantize, 0);
    if (inputLayer == nullptr) {
        THROW_IE_EXCEPTION << "input data layer for FakeQuantize " << fakeQuantize->name << " is nullable";
    }
    if (inputLayer->name != layer.name) {
        return;
    }

    const Blob::Ptr weights = CNNNetworkHelper::quantizeWeights(*fakeQuantize, roundQuantizedValues);
    CNNNetworkHelper::transformFakeQuantizeToConst(context, fakeQuantize, weights, layer.name);
}
