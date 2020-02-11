// Copyright (C) 2018-2020 Intel Corporation
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
 * @brief Modify child layers walking order to maintain strict ordering required for gna_fuse logic
 */
class FuzedLayersIterator {
    friend class  FuzedLayersContainer;
    using iterator = std::map<std::string, InferenceEngine::CNNLayerPtr>::iterator;
    std::list<iterator> allOutputs;
    bool needReorder = false;
    InferenceEngine::details::OutLayersIterator standardIterator;
    std::list<iterator>::iterator reorderedIterator;

 public:
    FuzedLayersIterator() = default;
    explicit FuzedLayersIterator(InferenceEngine::CNNLayer* origin) {
        bool hasActivation = false;
        for (auto && data : origin->outData) {
            auto & inputTo = data->getInputTo();
            for (auto i = inputTo.begin(); i != inputTo.end(); i++) {
                LayerInfo info(i->second);
                if (info.isActivation()) {
                    hasActivation = true;
                    allOutputs.push_back(i);
                } else {
                    allOutputs.push_front(i);
                }
            }
        }

        if (hasActivation && allOutputs.size() > 1) {
            needReorder = true;
        }

        if (!needReorder) {
            standardIterator = InferenceEngine::details::OutInfoWrapper(origin).begin();
        } else {
            reorderedIterator = allOutputs.begin();
        }
    }

    void operator ++() {
        if (!needReorder) {
            standardIterator.operator++();
            return;
        }
        reorderedIterator++;
    }

    bool operator == (FuzedLayersIterator that) const {
        if (!needReorder) {
            return standardIterator == that.standardIterator;
        }
        if (that.reorderedIterator == std::list<iterator>::iterator()) {
            return reorderedIterator == allOutputs.end();
        }

        return that.reorderedIterator == reorderedIterator;
    }

    bool operator != (FuzedLayersIterator that) const {
        return !this->operator==(that);
    }

    InferenceEngine::CNNLayerPtr operator *() const {
        if (!needReorder) {
            return *standardIterator;
        }
        return (*reorderedIterator)->second;
    }
};

class FuzedLayersContainer {
    InferenceEngine::CNNLayer* origin;
 public:
    explicit FuzedLayersContainer(InferenceEngine::CNNLayer* origin) : origin(origin) {}
    FuzedLayersIterator begin() {
        return FuzedLayersIterator(origin);
    }

    FuzedLayersIterator end() {
        return FuzedLayersIterator();
    }
};

inline FuzedLayersContainer make_fuzed_order(InferenceEngine::CNNLayer* origin) {
    FuzedLayersContainer fusedCnt(origin);
    return fusedCnt;
}

}  // namespace GNAPluginNS
