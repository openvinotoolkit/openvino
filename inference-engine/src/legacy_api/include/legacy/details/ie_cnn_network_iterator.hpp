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

    std::unordered_set<CNNLayer*> visited;
    std::list<CNNLayerPtr> nextLayersToVisit;
    InferenceEngine::CNNLayerPtr currentLayer;
    ICNNNetwork* network = nullptr;

    void init(const ICNNNetwork* network) {
        if (network == nullptr) THROW_IE_EXCEPTION << "ICNNNetwork object is nullptr";
        OutputsDataMap outputs;
        network->getOutputsInfo(outputs);
        std::list<CNNLayerPtr> layersQueue;

        for (const auto& output : outputs) {
            auto layer = getCreatorLayer(output.second).lock();
            if (layer) {
                layersQueue.push_back(layer);
            }
        }

        while (!layersQueue.empty()) {
            auto layer = layersQueue.front();
            layersQueue.pop_front();
            if (visited.find(layer.get()) != visited.end())
                continue;
            nextLayersToVisit.push_front(layer);
            visited.insert(layer.get());
            for (const auto& input : layer->insData) {
                const auto inData = input.lock();
                if (inData) {
                    auto prevLayer = getCreatorLayer(inData).lock();
                    if (prevLayer) {
                        layersQueue.push_back(prevLayer);
                    }
                }
            }
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
        return network == that.network && currentLayer == that.currentLayer;
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
