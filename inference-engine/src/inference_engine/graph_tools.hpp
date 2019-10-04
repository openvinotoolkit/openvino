// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <string>
#include <queue>
#include <set>
#include <algorithm>
#include <list>
#include <memory>
#include <functional>
#include <utility>
#include <map>
#include "layer_transform.hpp"


#include "ie_icnn_network.hpp"
#include "cnn_network_impl.hpp"
#include "ie_algorithm.hpp"

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

    static OutLayersIterator make_begin(std::vector<DataPtr> &origin) {
        if (origin.empty()) {
            return {};
        }
        OutLayersIterator it;

        it.dataCntIteratorCurrent = origin.begin();
        it.dataCntIteratorEnd = origin.end();
        it.moveToNextNonEmptyData();

        return it;
    }

    bool operator == (const OutLayersIterator &it) const {
        if (pointingToEnd || it.pointingToEnd) {
            return pointingToEnd && it.pointingToEnd;
        }
        return it.dataCntIteratorCurrent == dataCntIteratorCurrent && it.currentIterator == currentIterator;
    }

    bool operator != (const OutLayersIterator &it) const {
        return !this->operator==(it);
    }

    void operator ++() {
        if (dataCntIteratorCurrent == dataCntIteratorEnd) {
            return;
        }

        if (pointingToEnd) {
            return;
        }
        currentIterator++;
        if (currentIterator != dataCntIteratorCurrent->get()->getInputTo().end()) {
            return;
        }

        dataCntIteratorCurrent++;
        moveToNextNonEmptyData();
    }

    CNNLayerPtr operator * () const {
        return currentIterator->second;
    }

 protected:
    void moveToNextNonEmptyData() {
        pointingToEnd = true;
        for (; dataCntIteratorCurrent != dataCntIteratorEnd; dataCntIteratorCurrent++) {
            if (!dataCntIteratorCurrent->get()->getInputTo().empty()) {
                currentIterator = dataCntIteratorCurrent->get()->getInputTo().begin();
                pointingToEnd = false;
                break;
            }
        }
    }
};

class OutInfoWrapper {
    CNNLayer* origin = nullptr;
 public:
    explicit OutInfoWrapper(CNNLayer* origin) : origin(origin) {}
    OutLayersIterator begin() const {
        return OutLayersIterator::make_begin(origin->outData);
    }

    OutLayersIterator end() const  {
        return {};
    }
};

inline OutInfoWrapper default_order(CNNLayer * layer) {
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
template<class T, class Ordering = std::function<OutInfoWrapper(CNNLayer*)>>
inline bool DFS(std::unordered_map<CNNLayer *, bool> &visited,
                const InferenceEngine::CNNLayerPtr &layer,
                const T &visit,
                bool visitBefore,
                const Ordering & order = &default_order) {
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
template<class T>
inline void UnorderedDFS(std::unordered_set<CNNLayer *> &visited,
                         const InferenceEngine::CNNLayerPtr &layer,
                         const T &visit,
                         bool visitBefore) {
    std::queue<InferenceEngine::CNNLayerPtr> layers;
    auto cycleDFS = [&]() {
        if (layers.empty())
            return;
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

        // visit childs
        for (auto &od : cnnLayer->outData) {
            for (auto nl : od->getInputTo()) {
                layers.push(nl.second);
            }
        }

        // visit parents
        for (auto && input  : cnnLayer->insData) {
            if (!input.lock()) {
                THROW_IE_EXCEPTION << "Data inserted into layer " << cnnLayer->name << " is nullptr";
            } else {
                auto creatorLayer = input.lock()->getCreatorLayer().lock();
                if (creatorLayer) {
                    layers.push(creatorLayer);
                }
            }
        }

        if (!visitBefore)
            visit(cnnLayer);
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
template<class T>
inline void BFS(InferenceEngine::CNNLayerPtr layer, const T &visit, int maxDepth) {
    std::set<InferenceEngine::CNNLayer*> visited;
    std::list<InferenceEngine::CNNLayerPtr> nextLayers;
    nextLayers.push_back(layer);

    int layersOnLevel = 1;
    for (; !nextLayers.empty() && maxDepth != 0;) {
        visit(*nextLayers.begin());
        for (auto &od : (*nextLayers.begin())->outData) {
            for (auto nl : od->getInputTo()) {
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
template<class T, class Ordering = std::function<details::OutInfoWrapper(CNNLayer*)>>
inline bool CNNNetDFS(const InferenceEngine::CNNLayerPtr &layer,
                      const T &visit,
                      bool visitBefore = true,
                      const Ordering & order = &details::default_order) {
    if (layer == nullptr) {
        return true;
    }

    std::unordered_map < CNNLayer *, bool> visited;
    return details::DFS(visited, layer, visit, visitBefore, order);
}
/**
 * DFS algorithm with multiple starting data
 * @param layer - starting data
 * @param visit - callback to be called upon visiting
 * @param visitBefore - indicates when callback is happened before all child nodes or after
 */
template<class T>
inline bool CNNNetForestDFS(const std::vector<DataPtr> &heads, const T &visit, bool bVisitBefore) {
    std::unordered_map< CNNLayer *, bool> visited;
    for (const auto &in : heads) {
        for (const auto &to : in->getInputTo()) {
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
template<class Forest, class T>
inline bool CNNNetForestDFS(const Forest &heads, const T &visit, bool bVisitBefore) {
    if (heads.empty()) {
        return true;
    }

    std::unordered_map< CNNLayer *, bool> visited;
    for (auto & layer : heads) {
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
template<class Ordering, class Forest, class T>
inline bool CNNNetForestDFS(const Forest &heads, const T &visit, bool bVisitBefore, const Ordering &order) {
    if (heads.empty()) {
        return true;
    }

    std::unordered_map< CNNLayer *, bool> visited;
    for (auto & layer : heads) {
        if (!details::DFS(visited, layer, visit, bVisitBefore, order)) {
            return false;
        }
    }
    return true;
}

/**
 * Generic BFS algorithm traverser - with limiting depth
 * @param layer - starting layer
 * @param visit - callback to be called upon visiting
 */
template<class T>
inline void CNNNetNBFS(const InferenceEngine::CNNLayerPtr &layer, int maxDept, const T &visit) {
    if (!layer) {
        return;
    }
    details::BFS(layer, visit, maxDept + 1);
}

/**
 * Generic BFS algorithm traverser
 * @param layer - starting layer
 * @param visit - callback to be called upon visiting
 */
template<class T>
inline void CNNNetBFS(const InferenceEngine::CNNLayerPtr &layer, const T &visit) {
    if (!layer) {
        return;
    }

    details::BFS(layer, visit, -1);
}

/**
 * @brief name of the previous layer for given data
 * @param layer
 */
inline std::string  CNNNetPrevLayerName(const InferenceEngine::DataWeakPtr & dataWeak) {
    DataPtr dataStrong;

    IE_ASSERT(dataStrong = dataWeak.lock());

    CNNLayerPtr layerStrong;
    if (!(layerStrong = dataStrong->getCreatorLayer().lock())) {
        return dataStrong->getName();
    }

    return layerStrong->name;
}
/**
 * @brief pointer of previous layers
 * @param idx - index in previous layer collection
 * @param layer
 */
inline bool  CNNNetHasPrevLayer(const InferenceEngine::CNNLayer* layer, int idx = 0) {
    IE_ASSERT(layer != nullptr);
    if (layer->insData.empty() || layer->insData.size() <= idx) {
        return false;
    }
    auto prevData = layer->insData[idx].lock();
    return !!prevData->getCreatorLayer().lock();
}

/**
 * @brief pointer of previous layers
 * @param idx - index in previous layer collection
 * @param layer
 */
inline InferenceEngine::CNNLayerPtr  CNNNetPrevLayer(const InferenceEngine::CNNLayerPtr & layer, int idx = 0) {
    if (CNNNetHasPrevLayer(layer.get(), idx)) {
        auto prevData = layer->insData[idx].lock();
        return prevData->getCreatorLayer().lock();
    } else {
        THROW_IE_EXCEPTION << "Layer " << layer->name << " has no previous layer";
    }
}

/**
 * @brief pointer of previous layers
 * @param idx - index in previous layer collection
 * @param layer
 */
inline InferenceEngine::CNNLayerPtr  CNNNetPrevLayer(const InferenceEngine::CNNLayer* layer, int idx = 0) {
    IE_ASSERT(layer != nullptr);
    if (CNNNetHasPrevLayer(layer, idx)) {
        auto prevData = layer->insData[idx].lock();
        return prevData->getCreatorLayer().lock();
    } else {
        THROW_IE_EXCEPTION << "Layer " << layer->name << " has no previous layer";
    }
}

/**
 * @brief name of the previous layer
 * @param layer
 */
inline std::string  CNNNetPrevLayerName(const InferenceEngine::CNNLayerPtr & layer, int idx = 0) {
    auto prevLayer = CNNNetPrevLayer(layer, idx);
    return prevLayer->name;
}

/**
 * @brief swap two layer in graph - with modifying input/output references
 * also if layers have different dimensions they are preserved, so layers should be dimensions agnostic
 * lhs is a first node in topological order - this is current limitation to avoid passing cnnnetwork object
 */

inline void CNNNetSwapLayers(InferenceEngine::CNNLayerPtr lhs,
                             InferenceEngine::CNNLayerPtr rhs) {
    if (lhs == nullptr || rhs ==nullptr) {
        THROW_IE_EXCEPTION << "CNNNetSwapLayers : nullptr";
    }
    if (lhs.get() == rhs.get())
        return;

    if (lhs->outData.size() > 1) {
        THROW_IE_EXCEPTION << "Unsupported layer for swap operation : " << lhs->name;
    }
    if (rhs->outData.size() > 1) {
        THROW_IE_EXCEPTION << "Unsupported layer for swap operation : " << rhs->name;
    }

    auto &rhs_outputs = rhs->outData.front()->getInputTo();
    auto &lhs_outputs = lhs->outData.front()->getInputTo();

    // fixing input layers edges
    for (int i = 0; true; i++) {
        if (!CNNNetHasPrevLayer(lhs.get(), i)) break;
        auto prev_lhs = CNNNetPrevLayer(lhs, i);
        if (!prev_lhs) break;
        if (prev_lhs == rhs) continue;

        for (auto &prev_next : prev_lhs->outData) {
            auto lhs_ptr = prev_next->getInputTo().find(lhs->name);
            lhs_ptr->second = rhs;
        }
    }

    for (int i = 0; true; i++) {
        if (!CNNNetHasPrevLayer(rhs.get(), i)) break;
        auto prev_rhs = CNNNetPrevLayer(rhs, i);
        if (!prev_rhs) break;
        if (prev_rhs == lhs) continue;

        for (auto &prev_next : prev_rhs->outData) {
            auto lhs_ptr = prev_next->getInputTo().find(rhs->name);
            lhs_ptr->second = lhs;
        }
    }

    // fixing output layers back edges
    for (auto &next_lhs : lhs_outputs) {
        if (next_lhs.second == rhs) continue;

        bool hasHrsConnection = false;
        for (auto &ins_for_lhs_next : next_lhs.second->insData) {
            if (ins_for_lhs_next.lock()->getCreatorLayer().lock() != rhs ) continue;
            hasHrsConnection = true;
            break;
        }
        if (!hasHrsConnection) {
            for (auto &ins_for_lhs_next : next_lhs.second->insData) {
                if (ins_for_lhs_next.lock()->getCreatorLayer().lock() != lhs) continue;
                ins_for_lhs_next = rhs->outData.front();
            }
        }
    }

    for (auto &next_rhs : rhs_outputs) {
        if (next_rhs.second == lhs) continue;

        bool hasLHSConnection = false;
        for (auto &ins_for_rhs_next : next_rhs.second->insData) {
            if (ins_for_rhs_next.lock()->getCreatorLayer().lock() != lhs) continue;
            hasLHSConnection = true;
            break;
        }
        if (!hasLHSConnection) {
            for (auto &ins_for_rhs_next : next_rhs.second->insData) {
                if (ins_for_rhs_next.lock()->getCreatorLayer().lock() != rhs) continue;
                ins_for_rhs_next = lhs->outData.front();
            }
        }
    }

    // fixing layers itself output references
    {
        // c++11 lacks generic lambda
        using inputTo_element = std::remove_reference<decltype(*lhs_outputs.begin())>::type;

        std::remove_reference<decltype(lhs_outputs)>::type tmp;
        bool bHadInterconnectR2L = false;

        // 0. remove interconnect rhs->lhs
        details::erase_if(rhs_outputs, [&bHadInterconnectR2L, &lhs](inputTo_element & element) {
            bHadInterconnectR2L |= element.second == lhs;
            return element.second == lhs;
        });

        // 1. move all output references from rhs to tmp
        tmp.insert(std::begin(rhs_outputs), std::end(rhs_outputs));
        rhs_outputs.clear();


        // 2. removing lhs->rhs interconnect
        bool bHadInterConnect = false;
        details::erase_if(lhs_outputs, [&bHadInterConnect, &rhs](inputTo_element & element) {
            bHadInterConnect |= element.second == rhs;
            return element.second == rhs;
        });

        // 3. move all output references from lhs to rhs
        rhs_outputs.insert(std::begin(lhs_outputs), std::end(lhs_outputs));
        lhs_outputs.clear();

        // 4. move from tmp to lhs
        lhs_outputs.insert(std::begin(tmp), std::end(tmp));

        // 5.restore interconnects
        if (bHadInterConnect) {
            rhs_outputs[lhs->name] = lhs;
        }
        if (bHadInterconnectR2L) {
            lhs_outputs[rhs->name] = rhs;
        }
    }

    // fixing layers itself input references
    {
        // 1. removing interconnects lhs->rhs
        bool interConnectBackL2R = false;
        details::erase_if(lhs->insData, [&interConnectBackL2R, &rhs](DataWeakPtr weakData) {
            interConnectBackL2R |= weakData.lock()->getCreatorLayer().lock() == rhs;
            return weakData.lock()->getCreatorLayer().lock() == rhs;
        });

        // 2. removing interconnects rhs->lhs
        auto interConnectBackR2L = false;
        if (!interConnectBackL2R) {
            details::erase_if(rhs->insData, [&interConnectBackR2L, &lhs](DataWeakPtr weakData) {
                interConnectBackR2L |= weakData.lock()->getCreatorLayer().lock() == lhs;
                return weakData.lock()->getCreatorLayer().lock() == lhs;
            });
        }

        // swap back edges
        std::swap(lhs->insData, rhs->insData);

        // 4. Restoring interconnections
        if (interConnectBackL2R) {
            rhs->insData.push_back(lhs->outData.front());
        }
        if (interConnectBackR2L) {
            lhs->insData.push_back(rhs->outData.front());
        }
    }

    // TODO :
    // 1. step find out what layer is first in topological order
    // 2. integrate shape infer mechanism starting from lhs
    lhs->outData.front()->setDims(rhs->outData.front()->getDims());
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
inline CNNLayerSet CNNNetGetAllInputLayers(const ICNNNetwork &network) {
    InputsDataMap inputs;
    network.getInputsInfo(inputs);

    CNNLayerSet inputLayers;
    std::unordered_set<CNNLayer *> allLayers;

    if (inputs.empty())
        return inputLayers;

    for (const auto & input : inputs) {
        auto &secondLayers = input.second->getInputData()->getInputTo();

        if (secondLayers.empty())
            continue;

        details::UnorderedDFS(allLayers, secondLayers.begin()->second, [&](CNNLayerPtr layer) {
            if (layer->insData.empty()) {
                inputLayers.insert(layer);
            }
        }, false);
    }
    return inputLayers;
}

/**
 * @brief returns all layers that are input or memory , search started from arbitrary location in network
 * @param start layer
 * @return set of input layers
 */
inline CNNLayerSet CNNNetGetAllInputLayers(CNNLayer* layer) {
    CNNLayerSet inputLayers;
    std::unordered_set<CNNLayer *> allLayers;

    CNNLayerPtr layerPtr(layer, [](CNNLayer*){});

    details::UnorderedDFS(allLayers, layerPtr, [&](CNNLayerPtr layer) {
        if (layer->insData.empty()) {
            inputLayers.insert(layer);
        }
    }, false);
    return inputLayers;
}

/**
 * @brief copy Data from original graph, and insert into new graph, using layers remap information
 * @param input
 * @return
 */
inline DataPtr copyData(DataPtr input, std::unordered_map<CNNLayer*, CNNLayerPtr> &layersRemap) {
    auto newData = std::make_shared<Data>(*(input.get()));
    if (!input->getCreatorLayer().lock()) {
        THROW_IE_EXCEPTION << "Data " << input->getName() << " has no creator layer";
    }
    newData->getCreatorLayer() = layersRemap[input->getCreatorLayer().lock().get()];
    for (auto && input : newData->getInputTo()) {
        input.second = layersRemap[input.second.get()];
    }
    return newData;
}

using CNNNetPtr = std::shared_ptr<ICNNNetwork>;
using CNNNetCPtr = std::shared_ptr<const ICNNNetwork>;

// compares data, for copied network and in old network
inline bool areEqualDatas(DataPtr source, DataPtr target) {
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
}

/**
 * @brief deep copy of the entire network, structure using custom copier for layers
 * @param input - source network
 * @param cp - custom copier object, ex: [](CNNLayerPtr lp) { return injectData<EmptyStruct>(lp); }
 * @return copied network
 */
template <class Copier>
inline CNNNetPtr CNNNetCopy(const ICNNNetwork &input, const Copier &cp) {
    auto net = std::make_shared<details::CNNNetworkImpl>();

    // setting base args
    net->setPrecision(input.getPrecision());

    char name[1024];
    input.getName(name, sizeof(name));
    net->setName(name);

    // rest info is layer dependent so have to create graph clone
    std::unordered_map<CNNLayer*, CNNLayerPtr> oldToNewLayers;

    auto starters = CNNNetGetAllInputLayers(input);

    // 1st pass node creation
    bool res = CNNNetForestDFS(starters, [&](CNNLayerPtr  current){
        auto newLayer = cp(current);
        oldToNewLayers[current.get()] = newLayer;
        net->addLayer(newLayer);
    }, true);

    if (!res) {
        THROW_IE_EXCEPTION << "Copying of network not possible, due to existed loop.";
    }

    // internal utility to locate out data idx in layer
    auto findOutDataIdx = [&](DataPtr sourceData) {
        int dataIdx = -1;
        auto sourceLayer = sourceData->getCreatorLayer().lock();
        if (!sourceLayer) {
            THROW_IE_EXCEPTION << "Data " << sourceData->getName() << " has no creator layer";
        }
        for (int j = 0; j < sourceLayer->outData.size(); j++) {
            if (sourceData.get() == sourceLayer->outData[j].get()) {
                dataIdx = j;
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
        auto sourceLayerMap = sourceData->getInputTo();
        for (auto & layersMapping : sourceLayerMap) {
            if (layersMapping.second.get() != layer.get()) {
                continue;
            }
            for (int j = 0; j < layer->insData.size(); j++) {
                if (areEqualDatas(layer->insData[j].lock(), sourceData)) {
                    dataIdx = j;
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
    CNNNetForestDFS(starters, [&](CNNLayerPtr  current){
        auto newLayer = oldToNewLayers[current.get()];
        // remap output data
        for (int i = 0; i != current->outData.size(); i++) {
            newLayer->outData[i]->getCreatorLayer() = CNNLayerWeakPtr(newLayer);

            // transfer data info for getData routine
            net->getData(newLayer->outData[i]->getName()) = newLayer->outData[i];

            for (auto inputTo = std::begin(newLayer->outData[i]->getInputTo());
                 inputTo != std::end(newLayer->outData[i]->getInputTo());
                 inputTo++) {
                inputTo->second = oldToNewLayers[inputTo->second.get()];
            }
        }
        // remap input data
        for (int i = 0; i != current->insData.size(); i++) {
            // found that data IDX
            auto sourceData = current->insData[i].lock();
            auto sourceLayer = sourceData->getCreatorLayer().lock();
            if (!sourceLayer) {
                THROW_IE_EXCEPTION << "Data " << sourceData->getName() << " has no creator layer";
            }
            // find insData Entry in outData of sourceLayer
            newLayer->insData[i] = oldToNewLayers[sourceLayer.get()]->outData[findOutDataIdx(sourceData)];
        }
    }, true);

    // transfer input info
    InputsDataMap inputsInfo;
    input.getInputsInfo(inputsInfo);
    std::set<DataPtr> insDatas;
    for (auto &&info : inputsInfo) {
        for (auto secondLayer : info.second->getInputData()->getInputTo()) {
            auto secondLayerNew = oldToNewLayers[secondLayer.second.get()];
            InputInfo::Ptr infoNew = std::make_shared<InputInfo>();
            infoNew->setInputData(secondLayerNew->insData[findInsDataIdx(info.second->getInputData(), secondLayer.second)].lock());
            infoNew->getPreProcess() = info.second->getPreProcess();
            net->setInputInfo(infoNew);
        }
    }

    // transfer output info
    OutputsDataMap outmap;
    input.getOutputsInfo(outmap);
    for (auto && data : outmap) {
        ResponseDesc dsc;
        if (OK != net->addOutput(data.second->getCreatorLayer().lock()->name, findOutDataIdx(data.second), &dsc)) {
            THROW_IE_EXCEPTION << dsc.msg;
        }
    }

    ResponseDesc dsc;
    // transfer batch size
    if (OK != net->setBatchSize(input.getBatchSize(), &dsc)) {
        THROW_IE_EXCEPTION << dsc.msg;
    }

    return net;
}

/**
 * @brief deep copy of the entire network
 * @param input
 * @return
 */
inline CNNNetPtr CNNNetCopy(const ICNNNetwork &input) {
    struct EmptyStruct {};
    auto copier = [](CNNLayerPtr lp) { return injectData<EmptyStruct>(lp); };
    return  InferenceEngine::CNNNetCopy(input, copier);
}

/**
 * @@brief insertLayer between given layers
 * @param after, insertion happened after this layer, if after is nullptr, insertion happened after all inputLayers for before layer
 * @param before, insertion happened before layer, if before is nullptr, insertion happened before all outputLayers of after layer
 * @param layerToInsert inserted layer
 * @param outDataIndex index data to be used to insert layer after it. Cannot be used to specify allOutputDatas
 */
INFERENCE_ENGINE_API(size_t) invalid_data_idx;

inline void CNNNetworkInsertLayer(CNNLayerPtr after,
                                  CNNLayerPtr before,
                                  CNNLayerPtr layerToInsert,
                                  size_t outDataIndex = invalid_data_idx) {
    if (after == nullptr && before == nullptr) {
        THROW_IE_EXCEPTION << "Cannot Insert Layer: before or after layers should be valid layer pointers";
    }

    // internal utility to locate input data idx in layer
    auto findInsDataIdxes = [&](DataPtr sourceData, CNNLayerPtr layer) {
        std::vector<int> dataIdxes;
        auto outLayers = sourceData->getInputTo();
        for (auto & outLayer : outLayers) {
            if (outLayer.second.get() != layer.get()) {
                continue;
            }
            for (int j = 0; j < layer->insData.size(); j++) {
                if (areEqualDatas(layer->insData[j].lock(), sourceData)) {
                    dataIdxes.push_back(j);
                }
            }
        }
        IE_ASSERT(!dataIdxes.empty());
        return dataIdxes;
    };

    bool bLocated = false;
    bool hasOutputIndex = outDataIndex != invalid_data_idx;
    if (after != nullptr) {
        for (auto && data : after->outData) {
            if (hasOutputIndex && outDataIndex) {
                --outDataIndex;
                continue;
            }
            auto inputTo = data->getInputTo();
            for (auto inputIt = inputTo.begin(); inputIt != inputTo.end(); ++inputIt) {
                auto input = inputIt->second;
                if (before != nullptr && input.get() != before.get())
                    continue;

                // located data


                bool bLocatedBackward = true;

                for (auto x : findInsDataIdxes(data, input)) {
                    input->insData[x] = layerToInsert->outData.front();
                }

                if (!bLocatedBackward) {
                    THROW_IE_EXCEPTION << "Layer before has no back connection to after : " << after->name << " vs "
                                       << input->name;
                }

                layerToInsert->outData.front()->getInputTo()[inputIt->first] = input;

                bLocated = true;

                // erasing only one particular connection
                data->getInputTo().erase(inputIt->first);
                if (before != nullptr) {
                    break;
                }
            }
            if (data->getInputTo().empty()) {
                bLocated = true;
            }
            if (bLocated) {
                // erasing all connection
                if (before == nullptr) {
                    data->getInputTo().clear();
                }

                data->getInputTo()[layerToInsert->outData.front()->getName()]  = layerToInsert;
                layerToInsert->insData.push_back(data);
            }
            break;
        }

        // if given outputDataIndex is not correct, lets find index that matches *before* layer
        if (!bLocated) {
            if (before != nullptr) {
                IE_ASSERT(before->insData.size() == 1);
                auto prevLayer = after;
                for (auto idx = prevLayer->outData.begin(); idx != prevLayer->outData.end(); idx++) {
                    auto &outputports = (*idx)->getInputTo();
                    for (auto ll = outputports.begin(); ll != outputports.end(); ll++) {
                        if (ll->second.get() == before.get()) {
                            // looks we found where need to remove
                            outputports.erase(ll);
                            before->insData.clear();
                            before->insData.push_back(layerToInsert->outData.front());
                            layerToInsert->outData.front()->getInputTo()[before->name] = before;

                            bLocated = true;
                            break;
                        }
                    }
                    if (bLocated) {
                        break;
                    }
                }
                // now we have a before layer without inputs
            }
            if (bLocated) {
                // inserting into node that doesnt have child
                IE_ASSERT(!after->outData.empty());
                for (auto &&next : after->outData) {
                    if (!next->getInputTo().empty()) continue;
                    next->getInputTo()[layerToInsert->name] = layerToInsert;
                    layerToInsert->insData.push_back(after->outData.front());
                }
            }
        }
    }
    if (!bLocated) {
        THROW_IE_EXCEPTION << "Cannot insert layer between: " <<
                           ((after == nullptr) ? std::string("nullptr") : after->name) << " and " <<
                           ((before == nullptr) ? std::string("nullptr") : before->name);
    }
}

/**
 * @brief remove givven layer from topology, currently only layers with one input data and one output data supported
 */
inline void CNNNetworkRemoveLayer(CNNLayerPtr layer) {
    if (!layer) {
        THROW_IE_EXCEPTION << "Cannot remove layer pointed to NULL";
    }
    if (layer->insData.size() != 1) {
        THROW_IE_EXCEPTION << "Cannot remove layer : "<< layer->name <<" that has not 1 input";
    }
    if (layer->outData.size() != 1) {
        THROW_IE_EXCEPTION << "Cannot remove layer : "<< layer->name <<" that has not 1 output";
    }

    auto isp = layer->insData.front().lock();
    if (!isp) {
        THROW_IE_EXCEPTION << "Cannot remove layer : "<< layer->name <<" cannot get it's input";
    }
    // if dimensions of input layer not equal target dimensions - shape infer  or reshape layer required, so skipping those cases
    auto osp = layer->outData.front();
    if (isp->getDims() != osp->getDims()) {
        THROW_IE_EXCEPTION << "Cannot remove layer : "<< layer->name <<" its input layer("
                           << isp->getName() << ") and output(" << osp->getName() << ") have incompatible dimensions";
    }

    // remove isp->layer connection
    for (auto i = isp->getInputTo().begin(); i != isp->getInputTo().end(); i++) {
        if (i->second.get() == layer.get()) {
            isp->getInputTo().erase(i);
            break;
        }
    }

    // remove osp->layer connection
    for (auto  && outData : osp->getInputTo()) {
        for (auto i = outData.second->insData.begin(); i != outData.second->insData.end(); i++) {
            auto insData = i->lock();
            if (!insData) {
                THROW_IE_EXCEPTION << "Cannot remove layer : "<< layer->name <<", its output layer(" <<
                                   outData.first << " has invalid input configuration";
            }
            auto creator = insData->getCreatorLayer().lock();
            if (!creator) {
                THROW_IE_EXCEPTION << "Cannot remove layer : "<< layer->name <<", its output layer(" <<
                                   outData.first << " has invalid input configuration";
            }

            // found layer that need to be removed
            if (creator.get() == layer.get()) {
                outData.second->insData.erase(i);
                break;
            }
        }
    }

    // add isp->osp connections
    for (auto  && outData : osp->getInputTo()) {
        // new syntetic name to avoid duplicates in map
        isp->getInputTo()[layer->name + "_" + outData.first] = outData.second;
    }

    // add osp->isp connections
    for (auto  && outData : osp->getInputTo()) {
        outData.second->insData.push_back(isp);
    }

    // removing layer->osp, and layer->isp connection not necessary - layer will delete it by itself
}

/**
 * @brief Replaces layer with newLayer in network
 * @param network  - graph containing the layer
 * @param layer    - layer which need to replace
 * @param newLayer - new layer instead of layer; it must have same name like a layer for replace
 */
void CNNNetSubstituteLayer(InferenceEngine::ICNNNetwork &network,
                           const InferenceEngine::CNNLayerPtr &layer,
                           const InferenceEngine::CNNLayerPtr &newLayer);

/**
 * @brief Sorts CNNNetork graph in topological order, while uses custom ordering when walking among child nodes
 * @param network - input CNNNetwork
 * @param ordering - callback that returns output layers for given CNNLayer pointer, see default_order function
 * @return sorted CNNNetwork layers
 */
template <class LayerOrdering>
std::vector<CNNLayerPtr> CNNNetSortTopologicallyEx(const ICNNNetwork & network, LayerOrdering ordering) {
    std::vector<CNNLayerPtr> stackOfVisited;
    bool res = CNNNetForestDFS(
        CNNNetGetAllInputLayers(network),
        [&](CNNLayerPtr  current) {
            stackOfVisited.push_back(current);
        },
        false,
        ordering);

    if (!res) {
        THROW_IE_EXCEPTION << "Sorting not possible, due to existed loop.";
    }

    std::reverse(std::begin(stackOfVisited), std::end(stackOfVisited));

    return stackOfVisited;
}


}  // namespace InferenceEngine
