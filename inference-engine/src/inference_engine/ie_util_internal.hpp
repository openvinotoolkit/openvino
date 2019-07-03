// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef IE_UTIL_HPP
#define IE_UTIL_HPP

#include <vector>
#include <functional>
#include <deque>
#include <unordered_set>
#include <utility>
#include <string>

#include <cpp/ie_cnn_network.h>
#include <cnn_network_impl.hpp>
#include <tuple>
#include <type_traits>


namespace InferenceEngine {

/**
 * @brief Simple helper function to check element presence in container
 * container must provede stl-compliant find member function
 *
 * @param container - Container to check
 * @param element - element to check
 *
 * @return true if element present in container
 */
template<typename C, typename T>
bool contains(const C& container, const T& element) {
    return container.find(element) != container.end();
}

/**
 * @brief checks that given type is one of specified in variadic template list
 * @tparam ...
 */
template <typename...>
struct is_one_of {
    static constexpr bool value = false;
};

/**
 * @brief checks that given type is one of specified in variadic template list
 * @tparam ...
 */
template <typename F, typename S, typename... T>
struct is_one_of<F, S, T...> {
    static constexpr bool value =
        std::is_same<F, S>::value || is_one_of<F, T...>::value;
};



/**
 * @brief Split graph into subgraphs using provided splitter object
 *
 * @param network - Source network
 * @param splitter - Splitter object, take two adjacent layers, must return true
 * if layers must go to different subgraphs
 *
 * @return list of subgraphs
 */
INFERENCE_ENGINE_API_CPP(std::vector<std::vector<InferenceEngine::CNNLayerPtr>>)
groupSubgraphs(InferenceEngine::ICNNNetwork& network,
               std::function<bool(const InferenceEngine::CNNLayerPtr&,
                                  const InferenceEngine::CNNLayerPtr&)> splitter);

/**
 * @brief Creates data object copy unconnected to any graph
 * @param source - source data object
 * @return Shared pointer to new data object
 */
INFERENCE_ENGINE_API_CPP(DataPtr) cloneData(const Data& source);

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
 * @param layers - layers to clone, must all be in same network
 * @param layerCloner - layer cloning functor
 *
 * @return Cloned network
 */
INFERENCE_ENGINE_API_CPP(InferenceEngine::details::CNNNetworkImplPtr)
cloneNet(const std::vector<InferenceEngine::CNNLayerPtr>& layers,
         const ICNNNetworkStats* networkStats,
         std::function<CNNLayerPtr(const CNNLayer&)> layerCloner = clonelayer);

/**
 * Clones the whole network. All layers and data objects will be cloned
 *
 * Blobs inside layers are reused
 * */
INFERENCE_ENGINE_API_CPP(InferenceEngine::details::CNNNetworkImplPtr)
cloneNet(const InferenceEngine::ICNNNetwork &network);

namespace traverse {

INFERENCE_ENGINE_API_CPP(void)
forward(const InferenceEngine::CNNLayerPtr& layer, std::deque<InferenceEngine::CNNLayerPtr>& layers);

INFERENCE_ENGINE_API_CPP(void)
backward(const InferenceEngine::CNNLayerPtr& layer, std::deque<InferenceEngine::CNNLayerPtr>& layers);

template<class T>
void traverse(T& inputs,
              std::function<void(InferenceEngine::CNNLayerPtr& layer)> apply,
              std::function<void(const InferenceEngine::CNNLayerPtr& layer, std::deque<InferenceEngine::CNNLayerPtr>& layers)> expand = forward) {
    std::unordered_set<InferenceEngine::CNNLayerPtr> visitedObjects;
    std::deque<InferenceEngine::CNNLayerPtr>         layersToCheck;

    layersToCheck.insert(layersToCheck.end(), inputs.begin(), inputs.end());

    while (!layersToCheck.empty()) {
        auto& layer = layersToCheck.front();
        if (visitedObjects.insert(layer).second) {
            apply(layer);
            expand(layer, layersToCheck);
        }
        layersToCheck.pop_front();
    }
}

INFERENCE_ENGINE_API_CPP(void)
traverse(InferenceEngine::ICNNNetwork& network,
         std::function<void(InferenceEngine::CNNLayerPtr& layer)> apply,
         std::function<void(const InferenceEngine::CNNLayerPtr& layer,
         std::deque<InferenceEngine::CNNLayerPtr>& layers)> expand = forward);

}  // namespace traverse

using ordered_properties = std::vector<std::pair<std::string, std::string>>;
using printer_callback = std::function<void(const InferenceEngine::CNNLayerPtr,
                                            ordered_properties &,
                                            ordered_properties &)>;

/**
 * @brief Visualize network in GraphViz (.dot) format and write to output stream
 *
 * @param network - graph to visualize
 * @param out - output stream for saving graph
 * @param layer_cb - callback function, that called on every printed layer node
 */
INFERENCE_ENGINE_API_CPP(void) saveGraphToDot(InferenceEngine::ICNNNetwork &network, std::ostream &out, printer_callback layer_cb = nullptr);

/**
  @brief Return root data objects, i.e. objects came from input or const layers

  @param network - network to process

  @return set of root data objects,
  */
INFERENCE_ENGINE_API_CPP(std::unordered_set<DataPtr>)
getRootDataObjects(ICNNNetwork &network);

}  // namespace InferenceEngine

#endif  // IE_UTIL_HPP
