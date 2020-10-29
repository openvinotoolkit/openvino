// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_blob.h>
#include <legacy/ie_layers.h>

#include <string>
#include <functional>
#include <unordered_set>
#include <vector>
#include <utility>

namespace InferenceEngine {
class ICNNNetwork;

using LayersSet = std::unordered_set<CNNLayerPtr>;

/// Split network on subgraphs based on layer affinity
///
/// @param network - source network
/// @param checkers - list of supported plugins
///
/// @return list of subgraphs
std::vector<LayersSet>
splitGraph(ICNNNetwork& network,
           const std::vector<std::string>& plugins);

/// Sort sugraphs topologically, behaviour is undefined if there are circular
/// refences between subgraps
///
/// @param subgraphs - list of subgraphs
void
sortSubgraphs(std::vector<LayersSet>& subgraphs);

}  // namespace InferenceEngine

