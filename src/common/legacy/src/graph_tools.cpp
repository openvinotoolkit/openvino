// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <limits>
#include <string>
#include <vector>

#include "legacy/graph_tools.hpp"
#include "legacy/details/ie_cnn_network_tools.h"

using namespace std;

namespace InferenceEngine {
namespace details {

std::vector<CNNLayerPtr> CNNNetSortTopologically(const CNNNetwork& network) {
    std::vector<CNNLayerPtr> stackOfVisited;
    bool res = CNNNetForestDFS(
        CNNNetGetAllInputLayers(network),
        [&](CNNLayerPtr current) {
            stackOfVisited.push_back(current);
        },
        false);

    if (!res) {
        IE_THROW() << "Sorting not possible, due to existed loop.";
    }

    std::reverse(std::begin(stackOfVisited), std::end(stackOfVisited));

    return stackOfVisited;
}

}  // namespace details

}  // namespace InferenceEngine
