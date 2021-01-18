// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for the CNNNetworkIterator class
 *
 * @file ie_cnn_network_iterator.hpp
 */
#pragma once
#include <iterator>
#include <list>
#include <deque>
#include <unordered_set>
#include <utility>

#include "ie_api.h"
#include "cpp/ie_cnn_network.h"
#include "ie_locked_memory.hpp"

#include <legacy/ie_layers.h>
#include <legacy/cnn_network_impl.hpp>

namespace InferenceEngine {
namespace details {

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief This class enables range loops for CNNNetwork objects
 */
class INFERENCE_ENGINE_INTERNAL("Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1")
CNNNetworkIterator {
    IE_SUPPRESS_DEPRECATED_START

    std::unordered_set<CNNLayer*> visited {};
    std::list<CNNLayerPtr> nextLayersToVisit {};
    InferenceEngine::CNNLayerPtr currentLayer = nullptr;
    const ICNNNetwork* network = nullptr;

    void init(const ICNNNetwork* net) {
        network = net;
        if (network == nullptr) THROW_IE_EXCEPTION << "ICNNNetwork object is nullptr";

        OutputsDataMap outputs;
        network->getOutputsInfo(outputs);

        InputsDataMap inputs;
        network->getInputsInfo(inputs);

        auto get_consumers = [](const CNNLayerPtr& node) -> std::vector<CNNLayerPtr> {
            std::vector<CNNLayerPtr> consumers;
            for (const auto & output : node->outData) {
                for (const auto &consumer : getInputTo(output)) {
                    consumers.push_back(consumer.second);
                }
            }
            return consumers;
        };
        auto bfs = [&](const CNNLayerPtr& start_node, bool traverse_via_outputs = false) {
            if (!start_node || visited.count(start_node.get())) return;
            std::deque<CNNLayerPtr> q;
            q.push_front(start_node);
            while (!q.empty()) {
                auto node = q.front();
                q.pop_front();
                if (visited.insert(node.get()).second) {
                    nextLayersToVisit.push_front(node);
                }

                // Traverse via inputs
                for (const auto & input : node->insData) {
                    auto locked_input = input.lock();
                    if (!locked_input) {
                        THROW_IE_EXCEPTION << "insData for " << node->name << " is not valid.";
                    }
                    if (auto next_node = getCreatorLayer(locked_input).lock()) {
                        if (!visited.count(next_node.get())) {
                            // Check that all consumers were visited
                            bool all_consumers_used(true);
                            for (const auto & consumer : get_consumers(next_node)) {
                                if (!visited.count(consumer.get())) all_consumers_used = false;
                            }
                            if (all_consumers_used) {
                                q.push_front(next_node);
                            }
                        }
                    }
                }

                // Traverse via outputs
                if (traverse_via_outputs) {
                    for (const auto &consumer : get_consumers(node)) {
                        if (!visited.count(consumer.get())) {
                            q.push_front(consumer);
                        }
                    }
                }
            }
        };

        // First we run bfs starting from outputs that provides deterministic graph traverse
        for (const auto & output : outputs) {
            bfs(getCreatorLayer(output.second).lock());
        }

        // For cases when graph has no outputs we start bfs from inputs to ensure topological sort
        for (const auto & input : inputs) {
            const auto data_ptr = input.second->getInputData();
            for (const auto & consumer : getInputTo(data_ptr))
                bfs(consumer.second, true);
        }
        currentLayer = nextLayersToVisit.front();
    }


public:
    /**
     * iterator trait definitions
     */
    typedef std::forward_iterator_tag iterator_category;
    typedef CNNLayerPtr value_type;
    typedef int difference_type;
    typedef CNNLayerPtr pointer;
    typedef CNNLayerPtr reference;

    /**
     * @brief Default constructor
     */
    CNNNetworkIterator() = default;
    /**
     * @brief Constructor. Creates an iterator for specified CNNNetwork instance.
     * @param network Network to iterate. Make sure the network object is not destroyed before iterator goes out of
     * scope.
     */
    explicit CNNNetworkIterator(const ICNNNetwork* network) {
        init(network);
    }

    explicit CNNNetworkIterator(const CNNNetwork & network) {
        const auto & inetwork = static_cast<const InferenceEngine::ICNNNetwork&>(network);
        init(&inetwork);
    }

    /**
     * @brief Performs pre-increment
     * @return This CNNNetworkIterator instance
     */
    CNNNetworkIterator& operator++() {
        currentLayer = next();
        return *this;
    }

    /**
     * @brief Performs post-increment.
     * Implementation does not follow the std interface since only move semantics is used
     */
    void operator++(int) {
        currentLayer = next();
    }

    /**
     * @brief Checks if the given iterator is not equal to this one
     * @param that Iterator to compare with
     * @return true if the given iterator is not equal to this one, false - otherwise
     */
    bool operator!=(const CNNNetworkIterator& that) const {
        return !operator==(that);
    }

    /**
     * @brief Gets const layer pointer referenced by this iterator
     */
    const CNNLayerPtr& operator*() const {
        if (nullptr == currentLayer) {
            THROW_IE_EXCEPTION << "iterator out of bound";
        }
        return currentLayer;
    }

    /**
     * @brief Gets a layer pointer referenced by this iterator
     */
    CNNLayerPtr& operator*() {
        if (nullptr == currentLayer) {
            THROW_IE_EXCEPTION << "iterator out of bound";
        }
        return currentLayer;
    }
    /**
     * @brief Compares the given iterator with this one
     * @param that Iterator to compare with
     * @return true if the given iterator is equal to this one, false - otherwise
     */
    bool operator==(const CNNNetworkIterator& that) const {
        return currentLayer == that.currentLayer &&
            (network == that.network ||
             ((network == nullptr || that.network == nullptr) && currentLayer == nullptr));
    }

private:
    /**
     * @brief implementation based on BFS
     */
    CNNLayerPtr next() {
        if (nextLayersToVisit.empty()) {
            return nullptr;
        }

        nextLayersToVisit.pop_front();

        return nextLayersToVisit.empty() ? nullptr : nextLayersToVisit.front();
    }

    IE_SUPPRESS_DEPRECATED_END
};
}  // namespace details
}  // namespace InferenceEngine
