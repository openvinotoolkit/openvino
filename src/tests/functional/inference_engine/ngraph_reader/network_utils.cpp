// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>
#include <cmath>
#include <utility>
#include <map>
#include <algorithm>

#include "network_utils.hpp"
#include "cpp/ie_cnn_network.h"
#include "functional_test_utils/blob_utils.hpp"
#include <legacy/net_pass.h>
#include <legacy/details/ie_cnn_network_iterator.hpp>

namespace FuncTestUtils {

    bool compareParamVal(const std::string &val1, const std::string &val2) {
        std::vector<std::string> vals1, vals2;
        std::stringstream ss1(val1);
        std::string field;
        while (getline(ss1, field, ',')) {
            std::stringstream fs(field);
            std::string value;
            fs >> value;
            vals1.emplace_back(value);
        }

        std::stringstream ss2(val2);
        while (getline(ss2, field, ',')) {
            std::stringstream fs(field);
            std::string value;
            fs >> value;
            vals2.emplace_back(value);
        }

        if (vals1.size() != vals2.size())
            return false;

        for (size_t i = 0; i < vals1.size(); i++) {
            try {
                float v1 = std::stof(vals1[i]);
                float v2 = std::stof(vals2[i]);
                if (std::fabs(v2 - v1) > 0.00001f)
                    return false;
            } catch (...) {
                if (vals1[i] != vals2[i])
                    return false;
            }
        }
        return true;
    }

    void
    compareTensorIterators(const InferenceEngine::CNNLayerPtr &new_layer, const InferenceEngine::CNNLayerPtr &old_layer,
                           bool sameNetVersions);

    IE_SUPPRESS_DEPRECATED_START

    void compareCNNNLayers(const InferenceEngine::CNNLayerPtr &layer, const InferenceEngine::CNNLayerPtr &refLayer,
                           bool sameNetVersions) {
        std::vector<std::string> err_log;

        if (layer->type != refLayer->type) {
            err_log.push_back("Layer " + layer->name + " and ref layer " + refLayer->name + " have different type: " +
                              layer->type + " and " + refLayer->type);
        } else if (layer->type == "TensorIterator") {
            compareTensorIterators(layer, refLayer, sameNetVersions);
        }
        if (layer->type == "Activation") {
            err_log.pop_back();
            layer->type = "not";
            refLayer->params["type"] = "not";
        }

        if (layer->precision != refLayer->precision) {
            err_log.push_back(
                    "Layer " + layer->name + " and ref layer " + refLayer->name + " have different precisions: "
                    + layer->precision.name() + " and " + refLayer->precision.name());
        }

        if (layer->insData.size() != refLayer->insData.size()) {
            err_log.push_back(
                    "Layer " + layer->name + " and ref layer " + refLayer->name + " have different number of inputs: " +
                    std::to_string(layer->insData.size()) + " and " + std::to_string(refLayer->insData.size()));
        }

        if (layer->outData.size() != refLayer->outData.size()) {
            err_log.push_back(
                    "Layer " + layer->name + " and ref layer " + refLayer->name +
                    " have different number of outputs: " +
                    std::to_string(layer->outData.size()) + " and " + std::to_string(refLayer->outData.size()));
        }


        if (layer->blobs.size() != refLayer->blobs.size()) {
            err_log.push_back(
                    "Layer " + layer->type + " with name " + layer->name +
                    " and ref layer " + layer->type + " with name " + refLayer->name +
                    " have different number of blobs: " +
                    std::to_string(layer->blobs.size()) + " and " + std::to_string(refLayer->blobs.size()));
        }

        bool success = layer->type == refLayer->type &&
                       layer->precision == refLayer->precision &&
                       layer->insData.size() == refLayer->insData.size() &&
                       layer->outData.size() == refLayer->outData.size() &&
                       layer->blobs.size() == refLayer->blobs.size() &&
                       layer->affinity == refLayer->affinity;


        for (size_t i = 0; i < layer->insData.size() && success; i++) {
            auto lockedRefData = refLayer->insData[i].lock();
            auto lockedData = layer->insData[i].lock();
            success = success && lockedRefData->getTensorDesc() == lockedData->getTensorDesc();
            if (lockedRefData->getTensorDesc() != lockedData->getTensorDesc()) {
                err_log.push_back("Layer " + layer->name + " and ref layer " + refLayer->name +
                                  " have different tensor desc for locked input data");
            }
        }

        for (size_t i = 0; i < layer->outData.size() && success; i++) {
            if (refLayer->outData[i]->getTensorDesc() != layer->outData[i]->getTensorDesc()) {
                err_log.push_back("Layer " + layer->name + " and ref layer " + refLayer->name +
                                  " have different tensor desc for out Data");
            }
            success = success && refLayer->outData[i]->getTensorDesc() == layer->outData[i]->getTensorDesc();
        }

        // Different IR versions may have different layers specification which leads to parameters and blobs data mismatch
        // E.g.:
        // 1. V10 MatMul const weights input stores transposed weights relative to FullyConnected V5-V7 weights
        // which makes them not element-wise comparable.
        // 2. Interpolate layer has different parameters set in V5-V7 vs V10 specifications.
        if (sameNetVersions) {
            bool sameParamsCount = layer->params.size() == refLayer->params.size();
            if (!sameParamsCount) {
                err_log.push_back(
                        "Layer " + layer->name + " and ref layer " + refLayer->name +
                        " have different number of parameters: " +
                        std::to_string(layer->params.size()) + " and " + std::to_string(refLayer->params.size()));
            }
            success = success && sameParamsCount;
            for (const auto &item : layer->blobs) {
                if (!success) {
                    break;
                }
                const InferenceEngine::Blob::Ptr layerBlob = item.second;
                const InferenceEngine::Blob::Ptr refLayerBlob = refLayer->blobs[item.first];
                FuncTestUtils::compareBlobs(layerBlob, refLayerBlob);
            }
        }

        for (const auto &item : layer->params) {
            if (!success)
                break;
            if (refLayer->params.find(item.first) != refLayer->params.end()) {
                if (!compareParamVal(refLayer->params[item.first], item.second)) {
                    success = false;
                    err_log.push_back(
                            "Layer " + layer->name + " in new network differs from reference parameter " + item.first +
                            " (new, old): " + item.second + ", " + refLayer->params[item.first]);
                }
            } else {
                if (item.first == "originalLayersNames") continue;
                // ROIPooling specification says that there should be two parameters- pooled_h and pooled_w
                // our implementation of this op has a single parameter - output_size.
                if (item.first == "output_size" && layer->type == "ROIPooling") continue;
                // autob is a WA for nGraph ops
                if ((item.first != "auto_broadcast" && item.first != "autob") || item.second != "numpy") {
                    success = false;
                    err_log.push_back("Layer " + refLayer->name + " in ref net has no " + item.first + " attribute.");
                }
            }
        }

        if (!success) {
            for (auto &it : err_log) {
                std::cout << "ERROR: " << it << std::endl;
            }
            IE_THROW() << "CNNNetworks have different layers!";
        }
    }

    IE_SUPPRESS_DEPRECATED_END

    template<class T>
    void compareInfo(T &new_info, T &old_info, const std::string &err_msg) {
        bool success = new_info.size() == old_info.size();
        for (const auto &it : new_info) {
            if (!success)
                break;
            success = success && old_info.find(it.first) != old_info.end();
        }
        if (!success)
            IE_THROW() << err_msg;
    }

    void
    compareTensorIterators(const InferenceEngine::CNNLayerPtr &new_layer, const InferenceEngine::CNNLayerPtr &old_layer,
                           bool sameNetVersions) {
        IE_SUPPRESS_DEPRECATED_START
        auto ti_new = std::dynamic_pointer_cast<InferenceEngine::TensorIterator>(new_layer);
        auto ti_old = std::dynamic_pointer_cast<InferenceEngine::TensorIterator>(old_layer);

        if (!ti_new || !ti_old) {
            IE_THROW() << "Cannot cast the layer to TensorIterator.";
        }

        auto get_port_map = [](
                const std::vector<InferenceEngine::TensorIterator::PortMap> &port_map_list,
                const std::vector<InferenceEngine::DataPtr> &data_from,
                const std::vector<InferenceEngine::DataPtr> &data_to) {
            std::map<std::pair<std::string, std::string>, InferenceEngine::TensorIterator::PortMap> ordered_port_maps;
            for (auto &port_map : port_map_list) {
                ordered_port_maps[{data_from[port_map.from]->getName(), data_to[port_map.to]->getName()}] = port_map;
            }

            return std::move(ordered_port_maps);
        };

        auto get_data_ptrs = [](std::vector<InferenceEngine::DataWeakPtr> &wk_data_ptrs) {
            std::vector<InferenceEngine::DataPtr> data_ptrs;
            for (auto &wk_data : wk_data_ptrs) {
                auto data_ptr = wk_data.lock();
                IE_ASSERT(data_ptr != nullptr);
                data_ptrs.push_back(data_ptr);
            }
            return std::move(data_ptrs);
        };

        auto compare_port_maps = [](
                std::map<std::pair<std::string, std::string>,
                        InferenceEngine::TensorIterator::PortMap> &new_ordered_port_maps,
                std::map<std::pair<std::string, std::string>,
                        InferenceEngine::TensorIterator::PortMap> &old_ordered_port_maps) {
            if (new_ordered_port_maps.size() != old_ordered_port_maps.size()) {
                IE_THROW() << "PortMaps have different numbers of layers: " << new_ordered_port_maps.size() <<
                                   " and " << old_ordered_port_maps.size();
            }

            auto iterator_new = new_ordered_port_maps.begin();
            auto iterator_old = old_ordered_port_maps.begin();

            for (; iterator_new != new_ordered_port_maps.end() && iterator_old != old_ordered_port_maps.end();
                   iterator_new++, iterator_old++) {
                if (iterator_new->first != iterator_old->first) {
                    IE_THROW() << R"(Names of "from" and "to" layers in the port maps do not match!)";
                }

                InferenceEngine::TensorIterator::PortMap &pm_new = iterator_new->second;
                InferenceEngine::TensorIterator::PortMap &pm_old = iterator_old->second;

                if (pm_new.part_size != pm_old.part_size || pm_new.axis != pm_old.axis ||
                    pm_new.stride != pm_old.stride ||
                    pm_new.end != pm_old.end || pm_new.start != pm_old.start) {
                    IE_THROW() << "Parameters in the port maps do not match!";
                }
            }
        };

        auto output_port_mp_new = get_port_map(ti_new->output_port_map, ti_new->outData, ti_new->body.outputs);
        auto output_port_mp_old = get_port_map(ti_old->output_port_map, ti_old->outData, ti_old->body.outputs);
        compare_port_maps(output_port_mp_new, output_port_mp_old);

        auto input_port_mp_new = get_port_map(ti_new->input_port_map, get_data_ptrs(ti_new->insData),
                                              ti_new->body.inputs);
        auto input_port_mp_old = get_port_map(ti_old->input_port_map, get_data_ptrs(ti_old->insData),
                                              ti_old->body.inputs);
        compare_port_maps(input_port_mp_new, input_port_mp_old);

        auto back_edges_mp_new = get_port_map(ti_new->back_edges, ti_new->body.outputs, ti_new->body.inputs);
        auto back_edges_mp_old = get_port_map(ti_old->back_edges, ti_old->body.outputs, ti_old->body.inputs);
        compare_port_maps(back_edges_mp_new, back_edges_mp_old);

        // TI body comparison
        auto nodes_new = InferenceEngine::NetPass::TIBodySortTopologically(ti_new->body);
        auto nodes_old = InferenceEngine::NetPass::TIBodySortTopologically(ti_old->body);

        std::sort(nodes_new.begin(), nodes_new.end(), [](
                const InferenceEngine::CNNLayerPtr &l,
                const InferenceEngine::CNNLayerPtr &r) {
            return l->name < r->name;
        });

        std::sort(nodes_old.begin(), nodes_old.end(), [](
                const InferenceEngine::CNNLayerPtr &l,
                const InferenceEngine::CNNLayerPtr &r) {
            return l->name < r->name;
        });

        compareLayerByLayer(nodes_new, nodes_old, sameNetVersions);

        auto get_map = [](
                const std::vector<InferenceEngine::DataPtr> &data) -> std::map<std::string, InferenceEngine::DataPtr> {
            std::map<std::string, InferenceEngine::DataPtr> name_to_data_map;
            for (auto &it : data) {
                name_to_data_map[it->getName()] = it;
            }
            return std::move(name_to_data_map);
        };

        auto new_inputs = get_map(ti_new->body.inputs);
        auto old_inputs = get_map(ti_old->body.inputs);
        compareInfo<std::map<std::string, InferenceEngine::DataPtr>>(new_inputs, old_inputs,
                                                                     "Bodies of TensorIterator have different inputs!");

        auto new_outputs = get_map(ti_new->body.outputs);
        auto old_outputs = get_map(ti_old->body.outputs);
        compareInfo<std::map<std::string, InferenceEngine::DataPtr>>(new_outputs, old_outputs,
                                                                     "Bodies of TensorIterator have different outputs!");
        IE_SUPPRESS_DEPRECATED_END
    }

    void compareCNNNetworks(const InferenceEngine::CNNNetwork &network, const InferenceEngine::CNNNetwork &refNetwork,
                            bool sameNetVersions) {
        if (network.getName() != refNetwork.getName())
            IE_THROW() << "CNNNetworks have different names! " << network.getName()
                               << " and " << refNetwork.getName();

        if (network.getBatchSize() != refNetwork.getBatchSize())
            IE_THROW() << "CNNNetworks have different batch size! " << std::to_string(network.getBatchSize())
                               << " and " << std::to_string(refNetwork.getBatchSize());

        compareLayerByLayer(network, refNetwork, sameNetVersions);
        InferenceEngine::InputsDataMap newInput = network.getInputsInfo();
        InferenceEngine::InputsDataMap oldInput = refNetwork.getInputsInfo();
        InferenceEngine::OutputsDataMap newOutput = network.getOutputsInfo();
        InferenceEngine::OutputsDataMap oldOutput = refNetwork.getOutputsInfo();
        compareInfo<InferenceEngine::InputsDataMap>(newInput, oldInput, "CNNNetworks have different inputs!");
        compareInfo<InferenceEngine::OutputsDataMap>(newOutput, oldOutput, "CNNNetworks have different outputs!");
    }

IE_SUPPRESS_DEPRECATED_START

void compareLayerByLayer(const std::vector<InferenceEngine::CNNLayerPtr>& network,
                         const std::vector<InferenceEngine::CNNLayerPtr>& refNetwork,
                         bool sameNetVersions) {
    auto iterator = network.begin();
    auto refIterator = refNetwork.begin();
    if (network.size() != refNetwork.size())
        IE_THROW() << "CNNNetworks have different number of layers: " <<
            network.size() << " vs " << refNetwork.size();
    for (; iterator != network.end() && refIterator != refNetwork.end(); iterator++, refIterator++) {
        InferenceEngine::CNNLayerPtr layer = *iterator;
        InferenceEngine::CNNLayerPtr refLayer = *refIterator;
        compareCNNNLayers(layer, refLayer, sameNetVersions);
    }
}

void compareLayerByLayer(const InferenceEngine::CNNNetwork& network,
                         const InferenceEngine::CNNNetwork& refNetwork,
                         bool sameNetVersions) {
    InferenceEngine::details::CNNNetworkIterator iterator, refIterator, end;
    std::shared_ptr<InferenceEngine::details::CNNNetworkImpl> convertedNetwork, convertedRefNetwork;

    auto convertNetwork = [] (const InferenceEngine::CNNNetwork & net,
        std::shared_ptr<InferenceEngine::details::CNNNetworkImpl> & convertedNet,
        InferenceEngine::details::CNNNetworkIterator & it) {
        if (net.getFunction()) {
            convertedNet.reset(new InferenceEngine::details::CNNNetworkImpl(net));
            it = InferenceEngine::details::CNNNetworkIterator(convertedNet.get());
        } else {
            it = InferenceEngine::details::CNNNetworkIterator(net);
        }
    };

    convertNetwork(network, convertedNetwork, iterator);
    convertNetwork(refNetwork, convertedRefNetwork, refIterator);

    size_t layerCount = convertedNetwork ? convertedNetwork->layerCount() : network.layerCount();
    size_t layerRefCount = convertedRefNetwork ? convertedRefNetwork->layerCount() : refNetwork.layerCount();

    if (layerCount != layerRefCount)
        IE_THROW() << "CNNNetworks have different number of layers: " << layerCount << " vs " << layerRefCount;
    for (; iterator != end && refIterator != end; iterator++, refIterator++) {
        InferenceEngine::CNNLayerPtr layer = *iterator;
        InferenceEngine::CNNLayerPtr refLayer = *refIterator;
        compareCNNNLayers(layer, refLayer, sameNetVersions);
    }
    std::cout << std::endl;
}

IE_SUPPRESS_DEPRECATED_END

}  // namespace FuncTestUtils
