// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <assert.h>
#include "graph_transformer.h"

namespace InferenceEngine {

void replaceLayerWithNewLayer(ICNNNetwork &network, const CNNLayerPtr &layer, const CNNLayerPtr &newLayer) {
    assert(layer->name == newLayer->name);

    // Redirect srd data
    for (auto& src : layer->insData) {
        src.lock()->getInputTo()[layer->name] = newLayer;
    }
    newLayer->insData = layer->insData;

    // Redirect dst data
    for (auto& dst : layer->outData) {
        dst->creatorLayer = newLayer;
    }
    newLayer->outData = layer->outData;

    network.addLayer(newLayer);
}

}  // namespace InferenceEngine
