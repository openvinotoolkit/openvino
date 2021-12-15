// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ie_algorithm.hpp"
#include "cpp/ie_cnn_network.h"

#include <legacy/layer_transform.hpp>
#include <legacy/cnn_network_impl.hpp>

IE_SUPPRESS_DEPRECATED_START

namespace InferenceEngine {
namespace details {

/**
 * @brief Iterate over all layers followed by certain CNNLayer layer, and suitable to use ranged loops for output layers
 */
class OutLayersIterator {
    std::vector<DataPtr>::iterator dataCntIteratorCurrent;
    std::vector<DataPtr>::iterator dataCntIteratorEnd;

    using OutdataIterator = std::map<std::string, CNNLayerPtr>::iterator;
    bool pointingToEnd = true;
    OutdataIterator currentIterator;

public:
    OutLayersIterator() = default;

    static OutLayersIterator make_begin(std::vector<DataPtr>& origin) {
        if (origin.empty()) {
            return {};
        }
        OutLayersIterator it;

        it.dataCntIteratorCurrent = origin.begin();
        it.dataCntIteratorEnd = origin.end();
        it.moveToNextNonEmptyData();

        return it;
    }

    bool operator==(const OutLayersIterator& it) const {
        if (pointingToEnd || it.pointingToEnd) {
            return pointingToEnd && it.pointingToEnd;
        }
        return it.dataCntIteratorCurrent == dataCntIteratorCurrent && it.currentIterator == currentIterator;
    }

    bool operator!=(const OutLayersIterator& it) const {
        return !this->operator==(it);
    }

    void operator++() {
        if (dataCntIteratorCurrent == dataCntIteratorEnd) {
            return;
        }

        if (pointingToEnd) {
            return;
        }
        currentIterator++;
        if (currentIterator != getInputTo(dataCntIteratorCurrent->get()).end()) {
            return;
        }

        dataCntIteratorCurrent++;
        moveToNextNonEmptyData();
    }

    CNNLayerPtr operator*() const {
        return currentIterator->second;
    }

protected:
    void moveToNextNonEmptyData() {
        pointingToEnd = true;
        for (; dataCntIteratorCurrent != dataCntIteratorEnd; dataCntIteratorCurrent++) {
            if (!getInputTo(dataCntIteratorCurrent->get()).empty()) {
                currentIterator = getInputTo(dataCntIteratorCurrent->get()).begin();
                pointingToEnd = false;
                break;
            }
        }
    }
};

class OutInfoWrapper {
    CNNLayer* origin = nullptr;

public:
    explicit OutInfoWrapper(CNNLayer* origin): origin(origin) {}
    OutLayersIterator begin() const {
        return OutLayersIterator::make_begin(origin->outData);
    }

    OutLayersIterator end() const {
        return {};
    }
};

inline OutInfoWrapper default_order(CNNLayer* layer) {
    return OutInfoWrapper(layer);
}

/**
 * @brief implementation of DFS with visiting checking to avoid multientry
 * @param visited - set to store visited layers
 * @param layer - current layer to start DFS from
 * @param visit - user callback on visited node
 * @param visitBefore - indicates when callback is happened before all child nodes or after
 * @return false if cycle detected
 */
template <class T, class Ordering = std::function<OutInfoWrapper(CNNLayer*)>>
inline bool DFS(std::unordered_map<CNNLayer*, bool>& visited, const InferenceEngine::CNNLayerPtr& layer, const T& visit,
                bool visitBefore, const Ordering& order = &default_order) {
    if (layer == nullptr) {
        return true;
    }

    if (visitBefore) visit(layer);
    visited[layer.get()] = false;
    for (auto outLayerPtr : order(layer.get())) {
        auto i = visited.find(outLayerPtr.get());
        if (i != visited.end()) {
            /**
             * cycle detected we entered still not completed node
             */
            if (!i->second) {
                return false;
            }
            continue;
        }
        if (!DFS(visited, outLayerPtr, visit, visitBefore, order)) {
            return false;
        }
    }

    if (!visitBefore) visit(layer);
    visited[layer.get()] = true;
    return true;
}

/**
 * @brief implementation of DFS in unordered graph, mean next layers not just child but also parents
 * @param visited - set to store visited layers
 * @param layer - current layer to start UnorderedDFS from
 * @param visit - user callback on visited node
 * @param visitBefore - indicates when callback is happened before all child nodes or after
 */
template <class T>
inline void UnorderedDFS(std::unordered_set<CNNLayer*>& visited, const InferenceEngine::CNNLayerPtr& layer,
                         const T& visit, bool visitBefore) {
    std::queue<InferenceEngine::CNNLayerPtr> layers;
    auto cycleDFS = [&]() {
        if (layers.empty()) return;
        auto cnnLayer = layers.front();
        layers.pop();

        if (cnnLayer == nullptr) {
            return;
        }
        if (visited.end() != visited.find(cnnLayer.get())) {
            return;
        }

        if (visitBefore) visit(cnnLayer);
        visited.insert(cnnLayer.get());

        // visit children
        for (auto& od : cnnLayer->outData) {
            for (auto nl : getInputTo(od)) {
                layers.push(nl.second);
            }
        }

        // visit parents
        for (size_t i = 0; i < cnnLayer->insData.size(); i++) {
            auto& input = cnnLayer->insData[i];
            if (!input.lock()) {
                IE_THROW() << "Data " << i << " inserted into layer " << cnnLayer->name << " is nullptr";
            } else {
                auto creatorLayer = getCreatorLayer(input.lock()).lock();
                if (creatorLayer) {
                    layers.push(creatorLayer);
                }
            }
        }

        if (!visitBefore) visit(cnnLayer);
    };
    layers.push(layer);
    while (!layers.empty()) {
        cycleDFS();
    }
}

/**
 * @brief implementation of DFS with visiting checking to avoid multyentry
 * @param visited - set to store visited layers
 * @param layer - current layer to start DFS from
 * @param visit - user callback on visited node
 */
template <class T>
inline void BFS(InferenceEngine::CNNLayerPtr layer, const T& visit, int maxDepth) {
    std::set<InferenceEngine::CNNLayer*> visited;
    std::list<InferenceEngine::CNNLayerPtr> nextLayers;
    nextLayers.push_back(layer);

    int layersOnLevel = 1;
    for (; !nextLayers.empty() && maxDepth != 0;) {
        visit(*nextLayers.begin());
        for (auto& od : (*nextLayers.begin())->outData) {
            for (auto nl : getInputTo(od)) {
                if (visited.find(nl.second.get()) == visited.end()) {
                    nextLayers.push_back(nl.second);
                    visited.insert(nl.second.get());
                }
            }
        }
        nextLayers.pop_front();
        // move to nextLayer
        if (!--layersOnLevel) {
            layersOnLevel = nextLayers.size();
            maxDepth--;
        }
    }
}

}  // namespace details

/**
 * Generic DFS algorithm traverser
 * @param layer - starting layer
 * @param visit - callback to be called upon visiting
 * @param visitBefore - indicates when callback is happened before all child nodes or after
 */
template <class T, class Ordering = std::function<details::OutInfoWrapper(CNNLayer*)>>
inline bool CNNNetDFS(const InferenceEngine::CNNLayerPtr& layer, const T& visit, bool visitBefore = true,
                      const Ordering& order = &details::default_order) {
    if (layer == nullptr) {
        return true;
    }

    std::unordered_map<CNNLayer*, bool> visited;
    return details::DFS(visited, layer, visit, visitBefore, order);
}
/**
 * DFS algorithm with multiple starting data
 * @param layer - starting data
 * @param visit - callback to be called upon visiting
 * @param visitBefore - indicates when callback is happened before all child nodes or after
 */
template <class T>
inline bool CNNNetForestDFS(const std::vector<DataPtr>& heads, const T& visit, bool bVisitBefore) {
    std::unordered_map<CNNLayer*, bool> visited;
    for (const auto& in : heads) {
        for (const auto& to : getInputTo(in)) {
            if (visited.find(to.second.get()) != visited.end()) continue;
            if (!details::DFS(visited, to.second, visit, bVisitBefore)) {
                return false;
            }
        }
    }
    return true;
}

/**
 * DFS algorithm with multiple starting nodes
 * @param layer - starting layer
 * @param visit - callback to be called upon visiting
 * @param visitBefore - indicates when callback is happened before all child nodes or after
 */
template <class Forest, class T>
inline bool CNNNetForestDFS(const Forest& heads, const T& visit, bool bVisitBefore) {
    if (heads.empty()) {
        return true;
    }

    std::unordered_map<CNNLayer*, bool> visited;
    for (auto& layer : heads) {
        if (!details::DFS(visited, layer, visit, bVisitBefore)) {
            return false;
        }
    }
    return true;
}

/**
 * DFS algorithm with multiple starting nodes
 * @param layer - starting layer
 * @param visit - callback to be called upon visiting
 * @param visitBefore - indicates when callback is happened before all child nodes or after
 */
template <class Ordering, class Forest, class T>
inline bool CNNNetForestDFS(const Forest& heads, const T& visit, bool bVisitBefore, const Ordering& order) {
    if (heads.empty()) {
        return true;
    }

    std::unordered_map<CNNLayer*, bool> visited;
    for (auto& layer : heads) {
        if (!details::DFS(visited, layer, visit, bVisitBefore, order)) {
            return false;
        }
    }
    return true;
}

/**
 * Generic BFS algorithm traverser
 * @param layer - starting layer
 * @param visit - callback to be called upon visiting
 */
template <class T>
inline void CNNNetBFS(const InferenceEngine::CNNLayerPtr& layer, const T& visit) {
    if (!layer) {
        return;
    }
    details::BFS(layer, visit, -1);
}

/**
 * @brief pointer of previous layers
 * @param idx - index in previous layer collection
 * @param layer
 */
inline bool CNNNetHasPrevLayer(const InferenceEngine::CNNLayer* layer, int idx = 0) {
    IE_ASSERT(layer != nullptr);
    if (layer->insData.empty() || static_cast<int>(layer->insData.size()) <= idx) {
        return false;
    }
    auto prevData = layer->insData[idx].lock();
    return !!getCreatorLayer(prevData).lock();
}

/**
 * @brief to allow storing of LayersSP in collections ordered by  names
 */

class LayerNameLess {
public:
    bool operator()(const CNNLayerPtr& lhs, const CNNLayerPtr& rhs) const {
        return std::less<std::string>()(lhs->name, rhs->name);
    }
};

using CNNLayerSet = std::set<CNNLayerPtr, LayerNameLess>;

/**
 * @brief returns all layers that are input or memory
 * @param network
 * @return set of input layers
 */
inline CNNLayerSet CNNNetGetAllInputLayers(const CNNNetwork& network) {
    InputsDataMap inputs = network.getInputsInfo();
    OutputsDataMap outputs = network.getOutputsInfo();

    std::vector<DataPtr> entryDataSet;
    entryDataSet.reserve(inputs.size() + outputs.size());
    for (const auto &kvp : inputs)
        entryDataSet.push_back(kvp.second->getInputData());
    for (const auto &kvp : outputs)
        entryDataSet.push_back(kvp.second);

    CNNLayerSet inputLayers;
    std::unordered_set<CNNLayer*> allLayers;

    if (entryDataSet.empty()) return inputLayers;

    // define any layer connected to provided Data object (consumer or creator)
    auto findConnectedLayer = [] (const DataPtr& data) -> CNNLayerPtr {
        auto consumerLayers = getInputTo(data);
        if (!consumerLayers.empty())
            return consumerLayers.begin()->second;

        auto creator = getCreatorLayer(data).lock();
        if (creator != nullptr)
            return creator;

        return nullptr;
    };

    for (const auto& data : entryDataSet) {
        auto entryLayer = findConnectedLayer(data);

        if (entryLayer == nullptr) continue;

        details::UnorderedDFS(
            allLayers, entryLayer,
            [&inputLayers](const CNNLayerPtr& layer) {
                if (layer->insData.empty()) {
                    inputLayers.insert(layer);
                }
            },
            false);
    }
    return inputLayers;
}

inline CNNLayerSet CNNNetGetAllInputLayers(ICNNNetwork * network) {
    std::shared_ptr<ICNNNetwork> pointer(network, [](ICNNNetwork* p) { });
    return CNNNetGetAllInputLayers(CNNNetwork(pointer));
}

/**
 * @brief returns all layers that are input or memory , search started from arbitrary location in network
 * @param start layer
 * @return set of input layers
 */
inline CNNLayerSet CNNNetGetAllInputLayers(CNNLayer* layer) {
    CNNLayerSet inputLayers;
    std::unordered_set<CNNLayer*> allLayers;

    CNNLayerPtr layerPtr(layer, [](CNNLayer*) {});

    details::UnorderedDFS(
        allLayers, layerPtr,
        [&](CNNLayerPtr layer) {
            if (layer->insData.empty()) {
                inputLayers.insert(layer);
            }
        },
        false);
    return inputLayers;
}

/**
 * @brief Sorts CNNNetork graph in topological order, while uses custom ordering when walking among child nodes
 * @param network - input CNNNetwork
 * @param ordering - callback that returns output layers for given CNNLayer pointer, see default_order function
 * @return sorted CNNNetwork layers
 */
template <class LayerOrdering>
std::vector<CNNLayerPtr> CNNNetSortTopologicallyEx(const CNNNetwork& network, LayerOrdering ordering) {
    std::vector<CNNLayerPtr> stackOfVisited;
    bool res = CNNNetForestDFS(
        CNNNetGetAllInputLayers(network),
        [&](CNNLayerPtr current) {
            stackOfVisited.push_back(current);
        },
        false, ordering);

    if (!res) {
        IE_THROW() << "Sorting not possible, due to existed loop.";
    }

    std::reverse(std::begin(stackOfVisited), std::end(stackOfVisited));

    return stackOfVisited;
}

using CNNNetPtr = std::shared_ptr<ICNNNetwork>;
using CNNNetCPtr = std::shared_ptr<const ICNNNetwork>;

/**
 * @brief deep copy of the entire network, structure using custom copier for layers
 * @param input - source network
 * @param cp - custom copier object, ex: [](CNNLayerPtr lp) { return injectData<EmptyStruct>(lp); }
 * @return copied network
 */
template <class Copier>
inline CNNNetwork CNNNetCopy(const CNNNetwork& input, const Copier& cp) {
    auto net = std::make_shared<details::CNNNetworkImpl>();

    net->setName(input.getName());

    // rest info is layer dependent so have to create graph clone
    std::unordered_map<CNNLayer*, CNNLayerPtr> oldToNewLayers;

    auto starters = CNNNetGetAllInputLayers(input);

    // 1st pass node creation
    bool res = CNNNetForestDFS(
        starters,
        [&](CNNLayerPtr current) {
            auto newLayer = cp(current);
            oldToNewLayers[current.get()] = newLayer;
            net->addLayer(newLayer);
        },
        true);

    if (!res) {
        IE_THROW() << "Copying of network not possible, due to existed loop.";
    }

    // internal utility to locate out data idx in layer
    auto findOutDataIdx = [&](DataPtr sourceData) {
        int dataIdx = -1;
        auto sourceLayer = getCreatorLayer(sourceData).lock();
        if (!sourceLayer) {
            IE_THROW() << "Data " << sourceData->getName() << " has no creator layer";
        }
        for (size_t j = 0; j < sourceLayer->outData.size(); j++) {
            if (sourceData.get() == sourceLayer->outData[j].get()) {
                dataIdx = static_cast<int>(j);
                break;
            }
        }
        IE_ASSERT(dataIdx != -1);
        return dataIdx;
    };

    // compares data, for copied network and in old network
    auto areEqualDatas = [&](DataPtr source, DataPtr target) {
        if (source.get() == target.get()) {
            return true;
        }

        // dims comparison -
        // actual dims value might be incorrect dueto syntetic case
        // , when getbatch() size returns value not reflect in actual data

        if (source->getTensorDesc().getDims().size() != target->getTensorDesc().getDims().size()) {
            return false;
        }

        // name comparison
        if (source->getName() != target->getName()) {
            return false;
        }

        // inputTO layers are identical by design
        return true;
    };
    // internal utility to locate input data idx in layer
    auto findInsDataIdx = [&](DataPtr sourceData, CNNLayerPtr layer) {
        int dataIdx = -1;
        auto sourceLayerMap = getInputTo(sourceData);
        for (auto& layersMapping : sourceLayerMap) {
            if (layersMapping.second.get() != layer.get()) {
                continue;
            }
            for (size_t j = 0; j < layer->insData.size(); j++) {
                if (areEqualDatas(layer->insData[j].lock(), sourceData)) {
                    dataIdx = static_cast<int>(j);
                }
            }
            if (dataIdx != -1) {
                break;
            }
        }
        IE_ASSERT(dataIdx != -1);
        return dataIdx;
    };

    // 2nd pass edges creation
    CNNNetForestDFS(
        starters,
        [&](CNNLayerPtr current) {
            auto newLayer = oldToNewLayers[current.get()];
            // remap output data
            for (size_t i = 0; i != current->outData.size(); i++) {
                getCreatorLayer(newLayer->outData[i]) = CNNLayerWeakPtr(newLayer);

                // transfer data info for getData routine
                net->getData(newLayer->outData[i]->getName()) = newLayer->outData[i];

                for (auto inputTo = std::begin(getInputTo(newLayer->outData[i]));
                     inputTo != std::end(getInputTo(newLayer->outData[i])); inputTo++) {
                    inputTo->second = oldToNewLayers[inputTo->second.get()];
                }
            }
            // remap input data
            for (size_t i = 0; i != current->insData.size(); i++) {
                // found that data IDX
                auto sourceData = current->insData[i].lock();
                auto sourceLayer = getCreatorLayer(sourceData).lock();
                if (!sourceLayer) {
                    IE_THROW() << "Data " << sourceData->getName() << " has no creator layer";
                }
                // find insData Entry in outData of sourceLayer
                newLayer->insData[i] = oldToNewLayers[sourceLayer.get()]->outData[findOutDataIdx(sourceData)];
            }
        },
        true);

    // transfer input info
    InputsDataMap inputsInfo = input.getInputsInfo();
    std::set<DataPtr> insDatas;
    for (auto&& info : inputsInfo) {
        for (auto secondLayer : getInputTo(info.second->getInputData())) {
            auto secondLayerNew = oldToNewLayers[secondLayer.second.get()];
            InputInfo::Ptr infoNew = std::make_shared<InputInfo>();
            infoNew->setInputData(
                secondLayerNew->insData[findInsDataIdx(info.second->getInputData(), secondLayer.second)].lock());
            infoNew->getPreProcess() = info.second->getPreProcess();
            net->setInputInfo(infoNew);
        }
    }

    // transfer output info
    OutputsDataMap outmap = input.getOutputsInfo();
    for (auto&& data : outmap) {
        ResponseDesc dsc;
        if (OK != net->addOutput(getCreatorLayer(data.second).lock()->name, findOutDataIdx(data.second), &dsc)) {
            IE_THROW() << dsc.msg;
        }
    }

    ResponseDesc dsc;
    // transfer batch size
    if (OK != net->setBatchSize(input.getBatchSize(), &dsc)) {
        IE_THROW() << dsc.msg;
    }

    return CNNNetwork(net);
}

/**
 * @brief deep copy of the entire network
 * @param input
 * @return
 */
inline CNNNetwork CNNNetCopy(const CNNNetwork& input) {
    struct EmptyStruct {};
    auto copier = [](CNNLayerPtr lp) {
        return injectData<EmptyStruct>(lp);
    };
    return InferenceEngine::CNNNetCopy(input, copier);
}

namespace details {

/**
 * The structure to wrap network as lists of input and output data objects
 * Each layer of network is achievable by DFS started from inputs.
 *
 * NB! The input collection may contain a "fake" data object which is not a
 *     real input to network, but just a holder to keep "const" and "memory"
 *     layers alive. Fake data object points layers with empty creator field.
 *     The fake data object always has "UNSPECIFIED" precision attribute.
 */
struct CNNSubnet {
    std::vector<DataPtr> inputs;
    std::vector<DataPtr> outputs;
};

/**
 * @brief Detect all input data object, not only provided as entry point.
 * @param heads collection of some input into graph
 * @return all input data objects including "fake" data (layers holder).
 */
inline std::vector<DataPtr> CNNSubnetGetAllInputs(const std::vector<DataPtr>& heads) {
    CNNLayerSet inputLayers;
    std::unordered_set<CNNLayer*> allLayers;

    // Define all start layers
    for (const auto& data : heads) {
        auto& secondLayers = getInputTo(data);

        if (secondLayers.empty()) continue;

        details::UnorderedDFS(
                allLayers, secondLayers.begin()->second,
                [&](CNNLayerPtr layer) {
                    if (layer->insData.empty()) {
                        inputLayers.insert(layer);
                    }
                },
                false);
    }

    std::vector<DataPtr> res = heads;
    // Add fake input data to point on not achievable
    // layers from head (like const placeholders)
    for (auto& starter : inputLayers) {
        DataPtr holder(new Data(starter->name + ":input_holder", starter->precision));
        getInputTo(holder)[starter->name] = starter;
        res.push_back(holder);
    }

    return res;
}

/**
 * @brief Sorts SNNSubnet graph representation in topological order
 * @param subnet input object
 * @return layer collection sorted in topological order
 */
inline std::vector<CNNLayerPtr> CNNSubnetSortTopologically(const CNNSubnet& subnet) {
    std::vector<CNNLayerPtr> stackOfVisited;
    bool res = CNNNetForestDFS(
            CNNSubnetGetAllInputs(subnet.inputs),
            [&](CNNLayerPtr current) {
                stackOfVisited.push_back(current);
            },
            false);
    if (!res) {
        IE_THROW() << "Sorting not possible, due to existed loop.";
    }

    std::reverse(stackOfVisited.begin(), stackOfVisited.end());
    return stackOfVisited;
}

}  // namespace details
}  // namespace InferenceEngine

IE_SUPPRESS_DEPRECATED_END
