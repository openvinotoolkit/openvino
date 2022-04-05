// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <map>
#include <vector>
#include <list>
#include <string>
#include "gna_graph_tools.hpp"

namespace GNAPluginNS {
/**
 * @brief implements upstream search for BFS routine
 */
class UpstreamLayersIterator {
    using iterator = std::vector<InferenceEngine::DataWeakPtr> ::iterator;
    InferenceEngine::details::OutLayersIterator standardIterator;
    InferenceEngine::CNNLayer* origin = nullptr;
    iterator  currentLayer;
    iterator  endLayer;

 public:
    UpstreamLayersIterator() = default;
    explicit UpstreamLayersIterator(InferenceEngine::CNNLayer* origin, iterator beg) : origin(origin) {
        currentLayer = beg;
        endLayer = origin->insData.end();
    }

    void operator ++() {
        currentLayer++;
    }

    bool operator == (UpstreamLayersIterator that) const {
        // probing for end
        if (origin != nullptr && that.origin == nullptr) {
            return currentLayer == endLayer;
        }

        if (origin == nullptr && that.origin != nullptr) {
            return that.currentLayer == that.endLayer;
        }

        if (origin != that.origin) {
            THROW_GNA_EXCEPTION << "iterator not comparable for layers: " << origin->name << ", and " << that.origin->name;
        }

        return currentLayer == that.currentLayer;
    }

    bool operator != (UpstreamLayersIterator that) const {
        return !this->operator==(that);
    }

    InferenceEngine::CNNLayerPtr operator *() const {
        if (origin == nullptr) {
            return nullptr;
        }
        auto data = currentLayer->lock();
        if (!data) {
            THROW_GNA_EXCEPTION << "Cannot lock insData for layer: " << origin->name;
        }
        auto parent = getCreatorLayer(data).lock();
        if (!parent) {
            THROW_GNA_EXCEPTION << "Cannot getParent for layer: " << origin->name;
        }
        return parent;
    }
};

class UpstreamLayersContainer {
    InferenceEngine::CNNLayer* origin;
    int startIdx = -1;

 public:
    explicit UpstreamLayersContainer(InferenceEngine::CNNLayer* origin, int startIdx = -1) : origin(origin), startIdx(startIdx) {}

    UpstreamLayersIterator begin() {
        if (origin == nullptr) {
            return end();
        }
        auto beg = origin->insData.begin();
        if (startIdx > 0) {
            std::advance(beg, startIdx);
        }
        return UpstreamLayersIterator(origin, beg);
    }

    UpstreamLayersIterator end() {
        if (origin == nullptr) {
            return UpstreamLayersIterator();
        }
        auto end = origin->insData.end();
        if (startIdx != -1) {
            end = ++makeBeginIterator();
        }

        return UpstreamLayersIterator(origin, end);
    }

 protected:
    std::vector<InferenceEngine::DataWeakPtr>::iterator makeBeginIterator() {
        auto beg = origin->insData.begin();
        if (startIdx > 0) {
            std::advance(beg, startIdx);
        }
        return beg;
    }
};

inline UpstreamLayersContainer make_upstream_order(InferenceEngine::CNNLayer* origin, int startIdx = -1) {
    UpstreamLayersContainer fusedCnt(origin, startIdx);
    return fusedCnt;
}


}  // namespace GNAPluginNS
