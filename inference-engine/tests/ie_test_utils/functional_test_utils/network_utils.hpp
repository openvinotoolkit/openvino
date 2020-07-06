// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpp/ie_cnn_network.h"
#include "cnn_network_impl.hpp"
#include "details/ie_cnn_network_iterator.hpp"

namespace FuncTestUtils {

void compareCNNNetworks(const InferenceEngine::CNNNetwork &network, const InferenceEngine::CNNNetwork &refNetwork, bool sameNetVersions = true);

void compareCNNNLayers(const InferenceEngine::CNNLayerPtr &layer, const InferenceEngine::CNNLayerPtr &refLayer, bool sameNetVersions);

IE_SUPPRESS_DEPRECATED_START
template <class T>
inline void compareLayerByLayer(const T& network, const T& refNetwork, bool sameNetVersions = true) {
    InferenceEngine::details::CNNNetworkIterator iterator, refIterator, end;
    std::shared_ptr<InferenceEngine::details::CNNNetworkImpl> convertedNetwork, convertedRefNetwork;

    auto convertNetwork = [] (const InferenceEngine::CNNNetwork & net,
        std::shared_ptr<InferenceEngine::details::CNNNetworkImpl> & convertedNet,
        InferenceEngine::details::CNNNetworkIterator & it) {
        if (net.getFunction()) {
            convertedNet = std::make_shared<InferenceEngine::details::CNNNetworkImpl>(net);
            it = InferenceEngine::details::CNNNetworkIterator(convertedNet.get());
        } else {
            auto & inetwork = static_cast<const InferenceEngine::ICNNNetwork&>(net);
            it = InferenceEngine::details::CNNNetworkIterator(&inetwork);
        }
    };

    convertNetwork(network, convertedNetwork, iterator);
    convertNetwork(refNetwork, convertedRefNetwork, refIterator);

    size_t layerCount = convertedNetwork ? convertedNetwork->layerCount() : network.layerCount();
    size_t layerRefCount = convertedRefNetwork ? convertedRefNetwork->layerCount() : refNetwork.layerCount();

    if (layerCount != layerRefCount)
        THROW_IE_EXCEPTION << "CNNNetworks have different number of layers: " << network.layerCount() << " vs " << refNetwork.layerCount();
    for (; iterator != end && refIterator != end; iterator++, refIterator++) {
        InferenceEngine::CNNLayerPtr layer = *iterator;
        InferenceEngine::CNNLayerPtr refLayer = *refIterator;
        compareCNNNLayers(layer, refLayer, sameNetVersions);
    }
}

template <>
inline void compareLayerByLayer(const std::vector<InferenceEngine::CNNLayerPtr>& network,
                                const std::vector<InferenceEngine::CNNLayerPtr>& refNetwork,
                                bool sameNetVersions) {
    auto iterator = network.begin();
    auto refIterator = refNetwork.begin();
    if (network.size() != refNetwork.size())
        THROW_IE_EXCEPTION << "CNNNetworks have different number of layers: " <<
            network.size() << " vs " << refNetwork.size();
    for (; iterator != network.end() && refIterator != refNetwork.end(); iterator++, refIterator++) {
        InferenceEngine::CNNLayerPtr layer = *iterator;
        InferenceEngine::CNNLayerPtr refLayer = *refIterator;
        compareCNNNLayers(layer, refLayer, sameNetVersions);
    }
}

IE_SUPPRESS_DEPRECATED_END

}  // namespace FuncTestUtils