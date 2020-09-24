// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpp/ie_cnn_network.h"
#include <legacy/cnn_network_impl.hpp>
#include <legacy/details/ie_cnn_network_iterator.hpp>

namespace FuncTestUtils {

void compareCNNNetworks(const InferenceEngine::CNNNetwork &network, const InferenceEngine::CNNNetwork &refNetwork, bool sameNetVersions = true);

void compareCNNNLayers(const InferenceEngine::CNNLayerPtr &layer, const InferenceEngine::CNNLayerPtr &refLayer, bool sameNetVersions);

void compareLayerByLayer(const InferenceEngine::CNNNetwork& network,
                         const InferenceEngine::CNNNetwork& refNetwork,
                         bool sameNetVersions = true);

void compareLayerByLayer(const std::vector<InferenceEngine::CNNLayerPtr>& network,
                         const std::vector<InferenceEngine::CNNLayerPtr>& refNetwork,
                         bool sameNetVersions = true);

}  // namespace FuncTestUtils