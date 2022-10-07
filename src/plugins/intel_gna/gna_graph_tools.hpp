// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <legacy/graph_tools.hpp>
#include "gna_plugin_log.hpp"
#include "frontend/quantized_layer_params.hpp"
#include <utility>
#include <string>
#include <vector>
#include <limits>
#include <memory>

namespace InferenceEngine {

static constexpr size_t invalid_data_idx = std::numeric_limits<size_t>::max();

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

/// @brief utility to locate input data idx from given outdata and given layer
inline std::vector<int> CNNLayerFindInsDataIdxes(DataPtr sourceData, CNNLayerPtr layer) {
    std::vector<int> dataIdxes;
    auto outLayers = getInputTo(sourceData);
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
}

/**
 * @brief pointer of previous layers
 * @param idx - index in previous layer collection
 * @param layer
 */
inline InferenceEngine::CNNLayerPtr  CNNNetPrevLayer(const InferenceEngine::CNNLayerPtr & layer, int idx = 0) {
    if (CNNNetHasPrevLayer(layer.get(), idx)) {
        auto prevData = layer->insData[idx].lock();
        IE_ASSERT(prevData != nullptr);
        return getCreatorLayer(prevData).lock();
    } else {
        IE_THROW() << "Layer " << layer->name << " has no previous layer";
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
        return getCreatorLayer(prevData).lock();
    } else {
        IE_THROW() << "Layer " << layer->name << " has no previous layer";
    }
}

template <class T>
class ExtractRawPtr {
    T ptr;
 public:
    using raw_ptr_type = T;
    explicit ExtractRawPtr(T ptr) : ptr(ptr) {}
    T raw() {
        return ptr;
    }
};

template <class U>
class ExtractRawPtr<std::shared_ptr<U>> {
    std::shared_ptr<U> ptr;
 public:
    using raw_ptr_type = U*;
    explicit ExtractRawPtr(const std::shared_ptr<U> & ptr) : ptr(ptr) {}
    U * raw() {
        return ptr.get();
    }
};

template <class T>
inline typename ExtractRawPtr<T>::raw_ptr_type raw_ptr(T obj) {
    ExtractRawPtr<T> x(obj);
    return x.raw();
}

/**
 * @brief gets pointer to previous layer
 * @param idx - index in previous layer connection - in other layers only zero idx will be used
 * @param layer - source layer
 * @param shouldSkip - skip kriteria
 */
template <class Layer>
inline InferenceEngine::CNNLayerPtr  CNNNetPrevLayerSkipCertain(Layer layer, int idx,
                                                                const std::function<bool(CNNLayerPtr)> &shouldSkip) {
    IE_ASSERT(layer != nullptr);
    if (!CNNNetHasPrevLayer(raw_ptr(layer), idx)) {
        THROW_GNA_EXCEPTION << "Can't find PrevLayer. All layers are skipped.";
        return nullptr;
    }
    auto prev = CNNNetPrevLayer(layer, idx);

    /// using upper search simplified version
    while (shouldSkip(prev)) {
        if (!CNNNetHasPrevLayer(prev.get(), 0)) {
            THROW_GNA_EXCEPTION << "Can't find PrevLayer. All layers are skipped.";
            return nullptr;
        }
        prev = CNNNetPrevLayer(prev, 0);
    }

    return prev;
}

/**
 * @brief returns next layer, skipping certain layers based on given functor
 * @param layer - given start layer
 * @param oidx - index of output data for start layer, in other layers only zero oidx will be used
 * @param iidx - index of input layers for given output right after start layer, in other layers only zero oidx will be used
 * @param bOnlyCheck - doesn't throw exception if next layer missed
 * @param shouldSkip
 * @return layer pointer and it's insData index that uses to connect to previous layer in chain
 */

template <class Layer>
inline std::pair<InferenceEngine::CNNLayerPtr, std::vector<int>>  CNNNetCheckNextLayerSkipCertain(Layer layer, int oidx, int iidx, bool bOnlyCheck,
                                                                const std::function<bool(CNNLayerPtr)> &shouldSkip) {
    if (oidx >= layer->outData.size()) {
        if (bOnlyCheck) return {nullptr, {}};
        THROW_GNA_LAYER_EXCEPTION(layer) << " no next output layer for outdata: " << oidx;
    }
    if (getInputTo(layer->outData[oidx]).empty() || iidx >= getInputTo(layer->outData[oidx]).size()) {
        if (bOnlyCheck) return {nullptr, {}};
        THROW_GNA_LAYER_EXCEPTION(layer) << " no next output layer for outdata: " << oidx << " and inputTo index: " << iidx;
    }

    auto outLayer = getInputTo(layer->outData[oidx]).begin();
    std::advance(outLayer, iidx);

    int new_oidx = shouldSkip(outLayer->second) ? 0 : oidx;
    int new_iidx = shouldSkip(outLayer->second) ? 0 : iidx;

    while (shouldSkip(outLayer->second)) {
        if (outLayer->second->outData.size() <= new_oidx) {
            if (bOnlyCheck) return { nullptr, {} };
            THROW_GNA_LAYER_EXCEPTION(outLayer->second) << " no next output layer for outdata: " << new_oidx;
        }

        if (getInputTo(outLayer->second->outData[new_oidx]).size() <= new_iidx) {
            if (bOnlyCheck) return { nullptr, {} };
            THROW_GNA_LAYER_EXCEPTION(outLayer->second) << " no next output layer for outdata: " << new_oidx << " and inputTo index: " << new_iidx;
        }

        layer = outLayer->second;
        outLayer = getInputTo(layer->outData[new_oidx]).begin();
    }

    auto insDataIdx = CNNLayerFindInsDataIdxes(layer->outData[new_oidx], outLayer->second);
    return { outLayer->second, insDataIdx };
}

/**
 * @brief return all layers reachable from given one
 * @param layer
 * @param oDataIdx - -1 means iterate over all odata indexes
 * @param shouldSkip
 * @return
 */
    template <class Layer>
    inline std::vector<CNNLayerPtr> CNNNetGetAllNextLayersSkipCertain(Layer layer, int oDataIdx, const std::function<bool(CNNLayerPtr)> &shouldSkip)  {
        std::list<CNNLayerPtr> currentSet;
        std::vector<CNNLayerPtr> resultSet;

        std::vector<std::map<std::string, CNNLayerPtr>> start;
        if (oDataIdx == -1) {
            for (int i = 0; i != layer->outData.size(); i++) {
                start.push_back(getInputTo(layer->outData[i]));
            }
        } else {
            start.push_back(getInputTo(layer->outData[oDataIdx]));
        }

        auto separate_layers = [&currentSet, &resultSet, &shouldSkip](std::map<std::string, CNNLayerPtr>& inputTo) {
            for (auto &&bfsLayer : inputTo) {
                if (shouldSkip(bfsLayer.second)) {
                    currentSet.push_back(bfsLayer.second);
                    continue;
                }
                resultSet.push_back(bfsLayer.second);
            }
        };

        int startIdx, endIdx;
        if (oDataIdx == -1) {
            startIdx = 0;
            endIdx = layer->outData.size();
        } else {
            startIdx = oDataIdx;
            endIdx = oDataIdx + 1;
        }

        for (int i = startIdx; i != endIdx; i++) {
            separate_layers(getInputTo(layer->outData[i]));
        }

        std::set< CNNLayerPtr > visited;
        while (!currentSet.empty()) {
            auto currentLayer = currentSet.front();
            currentSet.pop_front();
            if (visited.count(currentLayer)) {
                continue;
            }
            visited.insert(currentLayer);
            for (auto && oData : currentLayer->outData) {
                separate_layers(getInputTo(oData));
            }
        }
        return resultSet;
    }

/// @brief alias for strict checkNextLayer (false)
template <class Layer>
inline std::pair<InferenceEngine::CNNLayerPtr, std::vector<int>>  CNNNetGetNextLayerSkipCertain(Layer layer, int oidx, int iidx,
                                                                               const std::function<bool(CNNLayerPtr)> &shouldSkip) {
    return CNNNetCheckNextLayerSkipCertain(layer, oidx, iidx, false, shouldSkip);
}

/// @brief alias for non-strict checkNextLayer (false)
template <class Layer>
inline bool CNNNetHasNextLayerSkipCertain(Layer layer, int oidx, int iidx, const std::function<bool(CNNLayerPtr)> &shouldSkip) {
    auto l = CNNNetCheckNextLayerSkipCertain(layer, oidx, iidx, true, shouldSkip);
    return l.first.get() != nullptr;
}


/// @brief utility to locate output data idx from given insData index and given layer
inline int CNNLayerFindOutDataIdx(CNNLayerPtr layer, int insDataIdx) {
    auto prevLayer = CNNNetPrevLayer(layer, insDataIdx);
    auto outDataToSearch = layer->insData[insDataIdx].lock();
    auto outDataIt = std::find(prevLayer->outData.begin(), prevLayer->outData.end(), outDataToSearch);
    return static_cast<int>(std::distance(prevLayer->outData.begin(), outDataIt));
}

/// @brief utility to locate output data from given insData index and given layer
/// also it returns iterator that represent link to this layer in inputToMap
inline std::pair<DataPtr, std::map<std::string, CNNLayerPtr>::iterator> CNNLayerFindOutData(CNNLayerPtr layer, int insDataIdx) {
    auto oDataIdx  = CNNLayerFindOutDataIdx(layer, insDataIdx);
    auto prevLayer = CNNNetPrevLayer(layer, insDataIdx);
    auto oData = prevLayer->outData[oDataIdx];
    for (auto inputTo  = getInputTo(oData).begin();
    inputTo != getInputTo(oData).end();
    inputTo++) {
        if (inputTo->second == layer) {
            return {oData, inputTo};
        }
    }
    THROW_GNA_LAYER_EXCEPTION(layer) << "cannot locate input data for: " << insDataIdx;
}

/**
 * @brief swap two layer in graph - with modifying input/output references
 * also if layers have different dimensions they are preserved, so layers should be dimensions agnostic
 * lhs is a first node in topological order - this is current limitation to avoid passing cnnnetwork object
 */
inline void CNNNetSwapLayers(InferenceEngine::CNNLayerPtr lhs,
                             InferenceEngine::CNNLayerPtr rhs) {
    if (lhs == nullptr || rhs ==nullptr) {
        IE_THROW() << "CNNNetSwapLayers : nullptr";
    }
    if (lhs.get() == rhs.get())
        return;

    if (lhs->outData.size() > 1) {
        IE_THROW() << "Unsupported layer for swap operation : " << lhs->name;
    }
    if (rhs->outData.size() > 1) {
        IE_THROW() << "Unsupported layer for swap operation : " << rhs->name;
    }

    auto &rhs_outputs = getInputTo(rhs->outData.front());
    auto &lhs_outputs = getInputTo(lhs->outData.front());

    // fixing input layers edges
    for (int i = 0; true; i++) {
        if (!CNNNetHasPrevLayer(lhs.get(), i)) break;
        auto prev_lhs = CNNNetPrevLayer(lhs, i);
        if (!prev_lhs) break;
        if (prev_lhs == rhs) continue;

        for (auto &prev_next : prev_lhs->outData) {
            auto lhs_ptr = getInputTo(prev_next).find(lhs->name);
            lhs_ptr->second = rhs;
        }
    }

    for (int i = 0; true; i++) {
        if (!CNNNetHasPrevLayer(rhs.get(), i)) break;
        auto prev_rhs = CNNNetPrevLayer(rhs, i);
        if (!prev_rhs) break;
        if (prev_rhs == lhs) continue;

        for (auto &prev_next : prev_rhs->outData) {
            auto lhs_ptr = getInputTo(prev_next).find(rhs->name);
            lhs_ptr->second = lhs;
        }
    }

    // fixing output layers back edges
    for (auto &next_lhs : lhs_outputs) {
        if (next_lhs.second == rhs) continue;

        bool hasHrsConnection = false;
        for (auto &ins_for_lhs_next : next_lhs.second->insData) {
            if (getCreatorLayer(ins_for_lhs_next.lock()).lock() != rhs ) continue;
            hasHrsConnection = true;
            break;
        }
        if (!hasHrsConnection) {
            for (auto &ins_for_lhs_next : next_lhs.second->insData) {
                if (getCreatorLayer(ins_for_lhs_next.lock()).lock() != lhs) continue;
                ins_for_lhs_next = rhs->outData.front();
            }
        }
    }

    for (auto &next_rhs : rhs_outputs) {
        if (next_rhs.second == lhs) continue;

        bool hasLHSConnection = false;
        for (auto &ins_for_rhs_next : next_rhs.second->insData) {
            if (getCreatorLayer(ins_for_rhs_next.lock()).lock() != lhs) continue;
            hasLHSConnection = true;
            break;
        }
        if (!hasLHSConnection) {
            for (auto &ins_for_rhs_next : next_rhs.second->insData) {
                if (getCreatorLayer(ins_for_rhs_next.lock()).lock() != rhs) continue;
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
            InferenceEngine::CNNLayerPtr creator = nullptr;
            auto data = weakData.lock();
            if (data != nullptr)
                creator = getCreatorLayer(data).lock();
            interConnectBackL2R |= creator == rhs;
            return creator == rhs;
        });

        // 2. removing interconnects rhs->lhs
        auto interConnectBackR2L = false;
        if (!interConnectBackL2R) {
            details::erase_if(rhs->insData, [&interConnectBackR2L, &lhs](DataWeakPtr weakData) {
                auto data = weakData.lock();
                IE_ASSERT(data != nullptr);
                interConnectBackR2L |= getCreatorLayer(data).lock() == lhs;
                return getCreatorLayer(data).lock() == lhs;
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
* @brief changes the Tensor Desctiption if data by created a new one with correct description and replacing original one
*/
inline DataPtr CNNReplaceDataWithChangedTensorDescription(DataPtr old_data, TensorDesc& new_td) {
    auto new_dataPtr = std::make_shared<Data>(old_data->getName() + "_reshaped", new_td);
    getInputTo(new_dataPtr) = getInputTo(old_data);
    auto creatorLayer = getCreatorLayer(old_data).lock();
    getCreatorLayer(new_dataPtr) = creatorLayer;
    size_t idx = -1;
    for (size_t i=0; i < creatorLayer->outData.size(); i++) {
        if (areEqualDatas(old_data, creatorLayer->outData[i])) {
            idx = i;
            break;
        }
    }
    if (idx == -1) THROW_GNA_EXCEPTION << "No idx for data was found";

    creatorLayer->outData[idx] = new_dataPtr;
    auto input_to = getInputTo(new_dataPtr);
    for (auto& input : input_to) {
        for (auto& input_idx : CNNLayerFindInsDataIdxes(old_data, input.second)) {
            input.second->insData[input_idx] = new_dataPtr;
        }
    }
    return new_dataPtr;
}

/**
* @brief Creates a Reshape with given name and tensor description
*/
inline CNNLayerPtr CNNNetworkCreateReshape(const TensorDesc& td, const std::string& name, bool quantized) {
    auto reshape = std::make_shared<ReshapeLayer>(LayerParams({name, "reshape", Precision::FP32}));
    auto reshapeLayerWithQuant = quantized ? InferenceEngine::injectData<GNAPluginNS::QuantizedLayerParams>(reshape) : reshape;
    auto dataPtr = std::make_shared<Data>(name + "_data", td);
    getCreatorLayer(dataPtr) = reshapeLayerWithQuant;
    reshapeLayerWithQuant->outData.push_back(dataPtr);

    return reshapeLayerWithQuant;
}

/**
 * @@brief insertLayer between given layers
 * @param after, insertion happened after this layer, if after is nullptr, insertion happened after all inputLayers for before layer
 * @param before, insertion happened before layer, if before is nullptr, insertion happened before all outputLayers of after layer
 * @param layerToInsert inserted layer
 * @param outDataIndex index data to be used to insert layer after it. Cannot be used to specify allOutputDatas
 */
inline void CNNNetworkInsertLayer(CNNLayerPtr after,
                                  CNNLayerPtr before,
                                  CNNLayerPtr layerToInsert,
                                  size_t outDataIndex = invalid_data_idx,
                                  size_t inDataIndex = invalid_data_idx) {
    if (after == nullptr && before == nullptr) {
        IE_THROW() << "Cannot Insert Layer: before or after layers should be valid layer pointers";
    }

    bool bLocated = false;
    bool hasOutputIndex = outDataIndex != invalid_data_idx;
    int  number_of_connections_between_after_n_before = 0;

    if (after != nullptr) {
        int nUnconnectedOData = 0;
        for (auto && data : after->outData) {
            if (hasOutputIndex && outDataIndex) {
                --outDataIndex;
                continue;
            }
            auto inputTo = getInputTo(data);
            for (auto inputIt = inputTo.begin(); inputIt != inputTo.end(); ++inputIt) {
                auto input = inputIt->second;
                if (before != nullptr && input.get() != before.get())
                    continue;

                // located data
                for (auto input_port_idx : CNNLayerFindInsDataIdxes(data, input)) {
                    if (((size_t)inDataIndex != invalid_data_idx && (size_t)inDataIndex == input_port_idx) || (size_t)inDataIndex == invalid_data_idx)
                        input->insData[input_port_idx] = layerToInsert->outData.front();
                    number_of_connections_between_after_n_before++;
                }

                getInputTo(layerToInsert->outData.front())[inputIt->first] = input;

                bLocated = true;

                // erasing only one particular connection
                // we must check if there is only one connection between after <=> before
                if (number_of_connections_between_after_n_before == 1)
                    getInputTo(data).erase(inputIt->first);

                if (before != nullptr) {
                    break;
                }
            }
            if (inputTo.empty()) {
                nUnconnectedOData++;
            }
            if (bLocated) {
                // erasing all connection
                if (before == nullptr) {
                    getInputTo(data).clear();
                }

                getInputTo(data)[layerToInsert->outData.front()->getName()]  = layerToInsert;
                layerToInsert->insData.push_back(data);
                break;
            }
            if (hasOutputIndex) {
                break;
            }
        }

        // separately checking case of possible single unconnected output of given layer
        if (!bLocated && !before && !hasOutputIndex) {
            if (nUnconnectedOData != 1 && number_of_connections_between_after_n_before <= 1) {
                THROW_GNA_EXCEPTION << "Cannot insert layer: " << LAYER_NAME(layerToInsert) <<" after: " << LAYER_NAME(after);
            }

            for (auto && data : after->outData) {
                if (!getInputTo(data).empty()) continue;

                bLocated = true;
                getInputTo(data)[layerToInsert->outData.front()->getName()]  = layerToInsert;
                layerToInsert->insData.push_back(data);

                break;
            }
        }

        // if given outputDataIndex is not correct, lets find index that matches *before* layer
        if (!bLocated) {
            if (before != nullptr) {
                IE_ASSERT(before->insData.size() == 1 || inDataIndex != invalid_data_idx && inDataIndex < before->insData.size());
                auto prevLayer = after;
                for (auto idx = prevLayer->outData.begin(); idx != prevLayer->outData.end(); idx++) {
                    auto &outputports = getInputTo(*idx);
                    for (auto ll = outputports.begin(); ll != outputports.end(); ll++) {
                        if (ll->second.get() == before.get()) {
                            // looks we found where need to remove
                            outputports.erase(ll);
                            before->insData[inDataIndex != invalid_data_idx ? inDataIndex : 0] = layerToInsert->outData.front();
                            getInputTo(layerToInsert->outData.front())[before->name] = before;

                            bLocated = true;
                            break;
                        }
                    }
                    if (bLocated) {
                        break;
                    }
                }
                // now we have a before layer with one input less
            }
            if (bLocated) {
                // inserting into node that doesnt have child
                IE_ASSERT(!after->outData.empty());
                for (auto &&next : after->outData) {
                    if (!getInputTo(next).empty()) continue;
                    getInputTo(next)[layerToInsert->name] = layerToInsert;
                    layerToInsert->insData.push_back(next);
                }
            }
        }
    }
    if (!bLocated) {
        IE_THROW() << "Cannot insert layer between: " <<
                           ((after == nullptr) ? std::string("nullptr") : after->name) << " and " <<
                           ((before == nullptr) ? std::string("nullptr") : before->name);
    }
}

/**
 * @brief returns previous layers and outData index for it
 * @tparam T
 * @param origin
 * @param acceptanceCriteria
 * @param idx
 */
template <class T>
std::vector<std::pair<CNNLayerPtr, int> > CNNNetGetPrevLayersSkip(CNNLayerPtr origin, const T &acceptanceCriteria, int idx = -1) {
    std::vector<std::pair<CNNLayerPtr, int> > prevLayers;
    for (int i = idx == -1 ? 0 : idx; CNNNetHasPrevLayer(origin.get(), i) && (idx == -1 || i == idx); i++) {
        auto prevLayer = CNNNetPrevLayer(origin, i);
        if (acceptanceCriteria(prevLayer)) {
            prevLayers.push_back({prevLayer, CNNLayerFindOutDataIdx(origin, i)});
        } else {
            // if for some input we need to look in upper layers - original index not used here intentionally
            auto prevPrevLayers = CNNNetGetPrevLayersSkip(prevLayer, acceptanceCriteria);
            prevLayers.insert(prevLayers.end(), prevPrevLayers.begin(), prevPrevLayers.end());
        }
    }

    return prevLayers;
}

/**
 * @brief remove given layer from topology, currently only layers with one input data and one output data supported
 */
inline void CNNNetworkRemoveLayer(CNNLayerPtr layer, bool checkDims = true) {
    if (!layer) {
        IE_THROW() << "Cannot remove layer pointed to NULL";
    }
    gnalog() << "Removing " << layer->name << " layer\n";
    if (layer->insData.size() != 1) {
        IE_THROW() << "Cannot remove layer : "<< layer->name <<" that has different number of inputs than 1";
    }
    if (layer->outData.size() != 1) {
        IE_THROW() << "Cannot remove layer : "<< layer->name <<" that has different number of outputs than 1";
    }

    auto isp = layer->insData.front().lock();
    if (!isp) {
        IE_THROW() << "Cannot remove layer : "<< layer->name <<" cannot get it's input";
    }
    // if dimensions of input layer not equal target dimensions - shape infer  or reshape layer required, so skipping those cases
    auto osp = layer->outData.front();
    if (checkDims && isp->getDims() != osp->getDims()) {
        IE_THROW() << "Cannot remove layer : "<< layer->name <<" its input layer("
                           << isp->getName() << ") and output(" << osp->getName() << ") have incompatible dimensions";
    }

    // remove isp->layer connection
    for (auto i = getInputTo(isp).begin(); i != getInputTo(isp).end(); i++) {
        if (i->second.get() == layer.get()) {
            getInputTo(isp).erase(i);
            break;
        }
    }

    // remove osp->layer connection
    for (auto  && outData : getInputTo(osp)) {
        for (int i = 0; i < outData.second->insData.size(); i++) {
            auto insData = outData.second->insData[i].lock();
            if (!insData) {
                IE_THROW() << "Cannot remove layer : "<< layer->name <<", its output layer(" <<
                                   outData.first << " has invalid input configuration";
            }
            auto creator = getCreatorLayer(insData).lock();
            if (!creator) {
                IE_THROW() << "Cannot remove layer : "<< layer->name <<", its output layer(" <<
                                   outData.first << " has invalid input configuration";
            }

            // found layer that need to be removed
            if (creator.get() == layer.get()) {
                outData.second->insData[i] = isp;
                break;
            }
        }
    }

    // add isp->osp connections
    for (auto  && outData : getInputTo(osp)) {
        // new syntetic name to avoid duplicates in map
        getInputTo(isp)[layer->name + "_" + outData.first] = outData.second;
    }

    // removing layer->osp, and layer->isp connection not necessary - layer will delete it by itself
}

/**
 * @brief reconnects given layer to different parent
 * before:
 *   old_prev_layer --> layer
 * after:
 *   new_prev_layer --> layer
 * limitations:
 *    - new & old prev layer must have exactly one outgoing port
 */
inline void CNNNetworkReconnectLayer(CNNLayerPtr old_prev_layer, CNNLayerPtr new_prev_layer, CNNLayerPtr layer, bool checkDims = true) {
    if (!layer) {
        IE_THROW() << "Cannot reconnect layer pointed to NULL";
    }
    if (!old_prev_layer) {
        IE_THROW() << "Cannot reconnect layer old parent is NULL";
    }
    if (!new_prev_layer) {
        IE_THROW() << "Cannot reconnect layer new parent is NULL";
    }

    gnalog() << "Reconnecting " << old_prev_layer->name << " --> " << layer->name << " layer to "
        << new_prev_layer->name << " -- > " << layer->name << "layer\n";

    if (layer->insData.size() < 1) {
        IE_THROW() << "Cannot reconnect layer : " << layer->name
            << " operation supports only layers with at least 1 incomming port";
    }

    if (old_prev_layer->outData.size() != 1) {
        IE_THROW() << "Cannot reconnect layer : " << old_prev_layer->name << " must have exactly 1 outgoing port";
    }

    if (new_prev_layer->outData.size() != 1) {
        IE_THROW() << "Cannot reconnect layer : " << new_prev_layer->name << " must have exactly 1 outgoing port";
    }
    // layer has ports
    // each port has several layers connected to port
    // we are assuming that old & new prev has only one outgoing port
    auto old_prev_layer_out_port_0 = old_prev_layer->outData.front();
    auto new_prev_layer_out_port_0 = new_prev_layer->outData.front();

    if (checkDims && old_prev_layer_out_port_0->getDims() != new_prev_layer_out_port_0->getDims()) {
        IE_THROW() << "Cannot reconnect layer : " << old_prev_layer->name << " as its output have different dims than"
            << new_prev_layer->name;
    }

    // find connection between old_prev & layer connection in layer input collection
    for (auto layer_input_port : CNNLayerFindInsDataIdxes(old_prev_layer_out_port_0, layer)) {
        layer->insData[layer_input_port] = new_prev_layer_out_port_0;
    }

    // remove old_prev->layer connection
    for (auto i = getInputTo(old_prev_layer_out_port_0).begin(); i != getInputTo(old_prev_layer_out_port_0).end(); i++) {
        if (i->second.get() == layer.get()) {
            getInputTo(new_prev_layer_out_port_0).insert({ layer->name, layer });
            getInputTo(old_prev_layer_out_port_0).erase(i);
            break;
        }
    }
}

/**
 * @brief returns a size of a specified data dimension depending on its back offset
 * @param data a pointer to the data
 * @param backOffset back dimension offset
 */
inline uint32_t GetDataDimSize(InferenceEngine::DataPtr data, uint32_t backOffset) {
    auto dims = data->getDims();
    return (dims.size() > backOffset - 1) ? static_cast<uint32_t>(dims[dims.size() - backOffset]) : uint32_t(1);
}

enum class DataDimName {
    N, C, H, W
};

/**
 * @brief returns a size of a specified data dimension depending on the layout
 * @param data a pointer to the data
 * @param dimName dimension name
 */
inline uint32_t GetDataDimSize(InferenceEngine::DataPtr data, DataDimName dimName) {
    uint32_t dimIxInNCHW = static_cast<uint32_t>(dimName);
    IE_ASSERT(dimIxInNCHW <= 3);

    std::vector<uint32_t> backOffsets;
    switch (data->getLayout()) {
        case Layout::C:
        case Layout::NC:
            // 1 will be returned for offsets > 2
            backOffsets = std::vector<uint32_t>{2, 1, 3, 4};
            break;
        case Layout::HWC:
            // 1 will be returned for offset 4
        case Layout::NHWC:
            backOffsets = std::vector<uint32_t>{4, 1, 3, 2};
            break;
        case Layout::CHW:
            // 1 will be returned for offset 4
        case Layout::NCHW:
            backOffsets = std::vector<uint32_t>{4, 3, 2, 1};
            break;
        default:
            THROW_GNA_EXCEPTION << data->getName() << " Unexpected layout " << data->getLayout();
    }
    return GetDataDimSize(data, backOffsets[dimIxInNCHW]);
}

}  // namespace InferenceEngine
