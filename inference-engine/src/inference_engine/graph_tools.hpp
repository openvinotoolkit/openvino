// Copyright (C) 2018 Intel Corporation
//
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
#include "layer_transform.hpp"


#include "ie_icnn_network.hpp"
#include "cnn_network_impl.hpp"
#include "ie_algorithm.hpp"

namespace InferenceEngine {
namespace details {

/**
 * @brief implementation of DFS with visiting checking to avoid multientry
 * @param visited - set to store visited layers
 * @param layer - current layer to start DFS from
 * @param visit - user callback on visited node
 * @param visitBefore - indicates when callback is happened before all child nodes or after
 * @return false if cycle detected
 */
template<class T>
inline bool DFS(std::unordered_map<CNNLayer *, bool> &visited,
                const InferenceEngine::CNNLayerPtr &layer,
                const T &visit,
                bool visitBefore) {
    if (layer == nullptr) {
        return true;
    }

    if (visitBefore) visit(layer);
    visited[layer.get()] = false;
    for (auto &od : layer->outData) {
        for (auto nl : od->getInputTo()) {
            auto i = visited.find(nl.second.get());
            if (i != visited.end()) {
                /**
                 * cycle detected we entered still not completed node
                 */
                if (!i->second) {
                    return false;
                }
                continue;
            }
            if (!DFS(visited, nl.second, visit, visitBefore)) {
                return false;
            }
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
template<class T>
inline bool CNNNetDFS(const InferenceEngine::CNNLayerPtr &layer, const T &visit, bool visitBefore = true) {
    if (layer == nullptr) {
        return true;
    }

    std::unordered_map < CNNLayer *, bool> visited;
    return details::DFS(visited, layer, visit, visitBefore);
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

    auto &rhs_outputs = rhs->outData.front()->inputTo;
    auto &lhs_outputs = lhs->outData.front()->inputTo;

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
            if (ins_for_lhs_next.lock()->creatorLayer.lock() != rhs ) continue;
            hasHrsConnection = true;
            break;
        }
        if (!hasHrsConnection) {
            for (auto &ins_for_lhs_next : next_lhs.second->insData) {
                if (ins_for_lhs_next.lock()->creatorLayer.lock() != lhs) continue;
                ins_for_lhs_next = rhs->outData.front();
            }
        }
    }

    for (auto &next_rhs : rhs_outputs) {
        if (next_rhs.second == lhs) continue;

        bool hasLHSConnection = false;
        for (auto &ins_for_rhs_next : next_rhs.second->insData) {
            if (ins_for_rhs_next.lock()->creatorLayer.lock() != lhs) continue;
            hasLHSConnection = true;
            break;
        }
        if (!hasLHSConnection) {
            for (auto &ins_for_rhs_next : next_rhs.second->insData) {
                if (ins_for_rhs_next.lock()->creatorLayer.lock() != rhs) continue;
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
            interConnectBackL2R |= weakData.lock()->creatorLayer.lock() == rhs;
            return weakData.lock()->creatorLayer.lock() == rhs;
        });

        // 2. removing interconnects rhs->lhs
        auto interConnectBackR2L = false;
        if (!interConnectBackL2R) {
            details::erase_if(rhs->insData, [&interConnectBackR2L, &lhs](DataWeakPtr weakData) {
                interConnectBackR2L |= weakData.lock()->creatorLayer.lock() == lhs;
                return weakData.lock()->creatorLayer.lock() == lhs;
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

    auto & secondLayers = inputs.begin()->second->getInputData()->getInputTo();
    if (secondLayers.empty())
        return inputLayers;

    details::UnorderedDFS(allLayers, secondLayers.begin()->second, [&](CNNLayerPtr layer) {
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
        THROW_IE_EXCEPTION << "Data " << input->name << " has no creator layer";
    }
    newData->getCreatorLayer() = layersRemap[input->getCreatorLayer().lock().get()];
    for (auto && input : newData->getInputTo()) {
        input.second = layersRemap[input.second.get()];
    }
    return newData;
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
inline CNNNetPtr CNNNetCopy(const ICNNNetwork &input, const Copier &cp) {
    auto net = std::make_shared<details::CNNNetworkImpl>();

    // setting base args
    net->setTargetDevice(input.getTargetDevice());
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
            THROW_IE_EXCEPTION << "Data " << sourceData->name << " has no creator layer";
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

        if (source->dims.size() != target->dims.size()) {
            return false;
        }

        // name comparison
        if (source->name != target->name) {
            return false;
        }

        // inputTO layers are identical by design
        return true;
    };
    // internal utility to locate input data idx in layer
    auto findInsDataIdx = [&](DataPtr sourceData, CNNLayerPtr layer) {
        int dataIdx = -1;
        auto sourceLayerMap = sourceData->inputTo;
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
            net->getData(newLayer->outData[i]->name) = newLayer->outData[i];

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
                THROW_IE_EXCEPTION << "Data " << sourceData->name << " has no creator layer";
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
        for (auto secondLayer : info.second->getInputData()->inputTo) {
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
 */
inline void CNNNetworkInsertLayer(CNNLayerPtr after, CNNLayerPtr before, CNNLayerPtr layerToInsert) {
    if (after == nullptr && before == nullptr) {
        THROW_IE_EXCEPTION << "Cannot Insert Layer: before or after layers should be valid layer pointers";
    }

    bool bLocated = false;
    if (after != nullptr) {
        // TODO: only one output data supported
        for (auto && data : after->outData) {
            for (auto && input : data->inputTo) {
                if (before != nullptr && input.second.get() != before.get())
                    continue;

                // located data
                bool bLocatedBackward = false;
                for (auto &&backward : input.second->insData) {
                    if (backward.lock()->getCreatorLayer().lock().get() == after.get()) {
                        backward = layerToInsert->outData.front();
                        bLocatedBackward = true;
                        break;
                    }
                }
                if (!bLocatedBackward) {
                    THROW_IE_EXCEPTION << "Layer before has no back connection to after : " << after->name << " vs "
                                       << input.second->name;
                }

                layerToInsert->outData.front()->inputTo[input.first] = input.second;

                bLocated = true;

                if (before != nullptr) {
                    // erasing only one particular connection
                    data->inputTo.erase(input.first);
                    break;
                }
            }
            if (bLocated) {
                // erasing all connection
                if (before == nullptr) {
                    data->inputTo.clear();
                }

                data->inputTo[layerToInsert->outData.front()->name]  = layerToInsert;
                layerToInsert->insData.push_back(data);

                if (before != nullptr) {
                    break;
                }
            }
        }

        if (!bLocated) {
            if (before != nullptr) {
                THROW_IE_EXCEPTION << "Layers are not adjiacend: " << after->name << " vs " << before->name;
            }
            // inserting into node that doesnt have childs
            IE_ASSERT(!after->outData.empty());
            after->outData.front()->inputTo[layerToInsert->name] = layerToInsert;
            layerToInsert->insData.push_back(after->outData.front());
        }
    }
}

}  // namespace InferenceEngine
