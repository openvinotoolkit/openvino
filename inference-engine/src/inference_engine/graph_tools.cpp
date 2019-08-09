// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#if defined(_WIN32)
# define NOMINMAX      // to avoid defining min/max macros
#endif  // _WIN32
#include "graph_tools.hpp"
#include "details/ie_cnn_network_tools.h"
#include <vector>
#include <limits>
#include <string>

using namespace std;

namespace InferenceEngine {
namespace details {

std::vector<CNNLayerPtr> CNNNetSortTopologically(const ICNNNetwork & network) {
    std::vector<CNNLayerPtr> stackOfVisited;
    bool res = CNNNetForestDFS(CNNNetGetAllInputLayers(network), [&](CNNLayerPtr  current){
        stackOfVisited.push_back(current);
    }, false);

    if (!res) {
        THROW_IE_EXCEPTION << "Sorting not possible, due to existed loop.";
    }

    std::reverse(std::begin(stackOfVisited), std::end(stackOfVisited));

    return stackOfVisited;
}

}   // namespace details

void CNNNetSubstituteLayer(InferenceEngine::ICNNNetwork &network,
                           const InferenceEngine::CNNLayerPtr &layer,
                           const InferenceEngine::CNNLayerPtr &newLayer) {
    IE_ASSERT(layer->name == newLayer->name);

    // Redirect srd data
    for (auto& src : layer->insData) {
        src.lock()->getInputTo()[layer->name] = newLayer;
    }
    newLayer->insData = layer->insData;

    // Redirect dst data
    for (auto& dst : layer->outData) {
        dst->getCreatorLayer() = newLayer;
    }
    newLayer->outData = layer->outData;

    network.addLayer(newLayer);
}


size_t invalid_data_idx = std::numeric_limits<size_t>::max();
}  // namespace InferenceEngine
