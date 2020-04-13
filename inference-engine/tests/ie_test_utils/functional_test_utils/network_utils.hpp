// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpp/ie_cnn_network.h"

namespace FuncTestUtils {

void compareCNNNetworks(const InferenceEngine::CNNNetwork &network, const InferenceEngine::CNNNetwork &refNetwork, bool sameNetVersions = true);

void compareCNNNLayers(const InferenceEngine::CNNLayerPtr &layer, const InferenceEngine::CNNLayerPtr &refLayer, bool sameNetVersions);

IE_SUPPRESS_DEPRECATED_START
template <class T>
inline void compareLayerByLayer(const T& network, const T& refNetwork, bool sameNetVersions = true) {
    auto iterator = network.begin();
    auto refIterator = refNetwork.begin();
    if (network.size() != refNetwork.size())
        THROW_IE_EXCEPTION << "CNNNetworks have different number of layers: " << network.size() << " vs " << refNetwork.size();
    for (; iterator != network.end() && refIterator != refNetwork.end(); iterator++, refIterator++) {
        InferenceEngine::CNNLayerPtr layer = *iterator;
        InferenceEngine::CNNLayerPtr refLayer = *refIterator;
        compareCNNNLayers(layer, refLayer, sameNetVersions);
    }
}
IE_SUPPRESS_DEPRECATED_END

}  // namespace FuncTestUtils