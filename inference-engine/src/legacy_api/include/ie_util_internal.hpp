// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpp/ie_cnn_network.h>

#include <ie_icnn_network.hpp>
#include <cnn_network_impl.hpp>
#include <file_utils.h>
#include <deque>
#include <functional>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace InferenceEngine {

/**
 * @brief Creates data object copy unconnected to any graph
 * @param source - source data object
 * @return Shared pointer to new data object
 */
DataPtr cloneData(const Data& source);

IE_SUPPRESS_DEPRECATED_START

/**
 * @brief Creates layer object copy, unconnected to any grapoh
 * @param source - source layer object
 * @return Shared pointer to new layer object
 */
INFERENCE_ENGINE_API_CPP(CNNLayerPtr) clonelayer(const CNNLayer& source);

/**
 * @brief Clones selected set of nodes into separate network
 * only connections between passed nodes will be duplicated
 *
 * @param layers Layers to clone, must all be in same network
 * @param networkStats A network statistic to clone
 *
 * @return Cloned network
 */
INFERENCE_ENGINE_API_CPP(InferenceEngine::details::CNNNetworkImplPtr)
cloneNet(const std::vector<InferenceEngine::CNNLayerPtr>& layers);

IE_SUPPRESS_DEPRECATED_END

/**
 * @brief Clones the whole network without conversion to CNNNetworkImpl. All layers and data objects will be cloned
 * @note Blobs inside layers are reused
 * @param network A network to clone
 * @return A cloned object
 */
INFERENCE_ENGINE_API_CPP(std::shared_ptr<InferenceEngine::ICNNNetwork>)
cloneNetwork(const InferenceEngine::ICNNNetwork& network);

/**
 * @brief Clones the whole network. All layers and data objects will be cloned
 * @note Blobs inside layers are reused
 * @param network A network to clone
 * @return A cloned object
 */
INFERENCE_ENGINE_API_CPP(InferenceEngine::details::CNNNetworkImplPtr)
cloneNet(const InferenceEngine::ICNNNetwork& network);

using ordered_properties = std::vector<std::pair<std::string, std::string>>;
using printer_callback =
    std::function<void(const InferenceEngine::CNNLayerPtr, ordered_properties&, ordered_properties&)>;

/**
 * @brief Visualize network in GraphViz (.dot) format and write to output stream
 *
 * @param network - graph to visualize
 * @param out - output stream for saving graph
 * @param layer_cb - callback function, that called on every printed layer node
 */
INFERENCE_ENGINE_API_CPP(void)
saveGraphToDot(InferenceEngine::ICNNNetwork& network, std::ostream& out, printer_callback layer_cb = nullptr);

/**
  @brief Return root data objects, i.e. objects came from input or const layers

  @param network - network to process

  @return set of root data objects,
  */
INFERENCE_ENGINE_API_CPP(std::unordered_set<DataPtr>)
getRootDataObjects(ICNNNetwork& network);

}  // namespace InferenceEngine
