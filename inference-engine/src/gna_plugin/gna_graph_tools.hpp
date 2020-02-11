// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "graph_tools.hpp"

#include <utility>
#include <string>
#include <vector>
#include <limits>

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
            InferenceEngine::CNNLayerPtr creator = nullptr;
            auto data = weakData.lock();
            if (data != nullptr)
                creator = data->getCreatorLayer().lock();
            interConnectBackL2R |= creator == rhs;
            return creator == rhs;
        });

        // 2. removing interconnects rhs->lhs
        auto interConnectBackR2L = false;
        if (!interConnectBackL2R) {
            details::erase_if(rhs->insData, [&interConnectBackR2L, &lhs](DataWeakPtr weakData) {
                auto data = weakData.lock();
                IE_ASSERT(data != nullptr);
                interConnectBackR2L |= data->getCreatorLayer().lock() == lhs;
                return data->getCreatorLayer().lock() == lhs;
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
 * @@brief insertLayer between given layers
 * @param after, insertion happened after this layer, if after is nullptr, insertion happened after all inputLayers for before layer
 * @param before, insertion happened before layer, if before is nullptr, insertion happened before all outputLayers of after layer
 * @param layerToInsert inserted layer
 * @param outDataIndex index data to be used to insert layer after it. Cannot be used to specify allOutputDatas
 */
inline void CNNNetworkInsertLayer(CNNLayerPtr after,
                                  CNNLayerPtr before,
                                  CNNLayerPtr layerToInsert,
                                  size_t outDataIndex = invalid_data_idx) {
    if (after == nullptr && before == nullptr) {
        THROW_IE_EXCEPTION << "Cannot Insert Layer: before or after layers should be valid layer pointers";
    }

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
                for (auto x : CNNLayerFindInsDataIdxes(data, input)) {
                    input->insData[x] = layerToInsert->outData.front();
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
                break;
            }
            if (hasOutputIndex) {
                break;
            }
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
                    layerToInsert->insData.push_back(next);
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

}  // namespace InferenceEngine
