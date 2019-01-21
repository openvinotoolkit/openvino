// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for the CNNNetworkIterator class
 * @file ie_cnn_network_iterator.hpp
 */
#pragma once
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <list>
#include <iterator>
#include <memory>
#include <vector>

#include <ie_inetwork.hpp>

namespace InferenceEngine {
namespace details {

template<class NT, class LT>
class INetworkIterator: public std::iterator<std::input_iterator_tag, std::shared_ptr<LT>> {
public:
    explicit INetworkIterator(NT * network, bool toEnd = false): network(network), currentIdx(0) {
        if (!network || toEnd)
            return;
        const auto& inputs = network->getInputs();

        std::vector<std::shared_ptr<LT>> allInputs;
        for (const auto& input : inputs) {
            allInputs.push_back(std::dynamic_pointer_cast<LT>(input));
        }

        bool res = forestDFS(allInputs, [&](std::shared_ptr<LT> current) {
            sortedLayers.push_back(current);
        }, false);

        if (!res) {
            THROW_IE_EXCEPTION << "Sorting not possible, due to existed loop.";
        }

        std::reverse(std::begin(sortedLayers), std::end(sortedLayers));
        currentLayer = getNextLayer();
    }
    bool operator!=(const INetworkIterator& that) const {
        return !operator==(that);
    }
    bool operator==(const INetworkIterator& that) const {
        return network == that.network && currentLayer == that.currentLayer;
    }
    typename INetworkIterator::reference operator*() {
        if (nullptr == currentLayer) {
            THROW_IE_EXCEPTION << "iterator out of bound";
        }
        return currentLayer;
    }

    INetworkIterator& operator++() {
        currentLayer = getNextLayer();
        return *this;
    }

    const INetworkIterator<NT, LT> operator++(int) {
        INetworkIterator<NT, LT> retval = *this;
        ++(*this);
        return retval;
    }

private:
    std::vector<std::shared_ptr<LT>> sortedLayers;
    std::shared_ptr<LT> currentLayer;
    size_t currentIdx;
    NT *network = nullptr;

    std::shared_ptr<LT> getNextLayer() {
        return (sortedLayers.size() > currentIdx) ? sortedLayers[currentIdx++] : nullptr;
    }

    template<class T>
    inline bool forestDFS(const std::vector<std::shared_ptr<LT>>& heads, const T &visit, bool bVisitBefore) {
        if (heads.empty()) {
            return true;
        }

        std::unordered_map<idx_t, bool> visited;
        for (auto & layer : heads) {
            if (!DFS(visited, layer, visit, bVisitBefore)) {
                return false;
            }
        }
        return true;
    }

    template<class T>
    inline bool DFS(std::unordered_map<idx_t, bool> &visited,
                    const std::shared_ptr<LT> &layer,
                    const T &visit,
                    bool visitBefore) {
        if (layer == nullptr) {
            return true;
        }

        if (visitBefore)
            visit(layer);

        visited[layer->getId()] = false;
        for (const auto &connection : network->getLayerConnections(layer->getId())) {
            if (connection.to().layerId() == layer->getId()) {
                continue;
            }
            const auto outLayer = network->getLayer(connection.to().layerId());
            auto i = visited.find(outLayer->getId());
            if (i != visited.end()) {
                /**
                 * cycle detected we entered still not completed node
                 */
                if (!i->second) {
                    return false;
                }
                continue;
            }

            if (!DFS(visited, outLayer, visit, visitBefore)) {
                return false;
            }
        }
        if (!visitBefore)
            visit(layer);
        visited[layer->getId()] = true;
        return true;
    }
};

}  // namespace details
}  // namespace InferenceEngine
