// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>
#include <map>
#include <queue>
#include <deque>
#include <string>
#include <vector>
#include <unordered_set>
#include <sstream>

#include "xml_parse_utils.h"
#include "network_serializer_v7.hpp"

namespace InferenceEngine {
namespace Serialization {

std::vector<CNNLayerPtr> TopologicalSort(const ICNNNetwork& network) {
    std::vector<CNNLayerPtr> ordered;
    std::unordered_set<std::string> used;

    OutputsDataMap outputs;
    network.getOutputsInfo(outputs);

    InputsDataMap inputs;
    network.getInputsInfo(inputs);

    auto get_consumers = [](const CNNLayerPtr& node) -> std::vector<CNNLayerPtr> {
        std::vector<CNNLayerPtr> consumers;
        for (const auto & output : node->outData) {
            for (const auto &consumer : getInputTo(output)) {
                consumers.push_back(consumer.second);
            }
        }
        return consumers;
    };
    auto bfs = [&used, &ordered, &get_consumers](const CNNLayerPtr& start_node, bool traverse_via_outputs = false) {
        if (!start_node) return;
        std::deque<CNNLayerPtr> q;
        q.push_front(start_node);
        while (!q.empty()) {
            auto node = q.front();
            q.pop_front();
            if (used.insert(node->name).second) {
                ordered.push_back(node);
            }

            // Traverse via inputs
            for (const auto & input : node->insData) {
                auto locked_input = input.lock();
                if (!locked_input) {
                    THROW_IE_EXCEPTION << "insData for " << node->name << " is not valid.";
                }
                if (auto next_node = getCreatorLayer(locked_input).lock()) {
                    if (!used.count(next_node->name)) {
                        // Check that all consumers were used
                        bool all_consumers_used(true);
                        for (const auto & consumer : get_consumers(next_node)) {
                            if (!used.count(consumer->name)) all_consumers_used = false;
                        }
                        if (all_consumers_used) {
                            q.push_front(next_node);
                        }
                    }
                }
            }

            // Traverse via outputs
            if (traverse_via_outputs) {
                for (const auto &consumer : get_consumers(node)) {
                    if (!used.count(consumer->name)) {
                        q.push_front(consumer);
                    }
                }
            }
        }
    };

    // First we run bfs starting from outputs that provides deterministic graph traverse
    for (const auto & output : outputs) {
        if (!used.count(output.first)) {
            bfs(getCreatorLayer(output.second).lock());
        }
    }

    // For cases when graph has no outputs we start bfs from inputs to ensure topological sort
    for (const auto & input : inputs) {
        const auto data_ptr = input.second->getInputData();
        for (const auto & consumer : getInputTo(data_ptr))
        if (!used.count(consumer.first)) {
            bfs(consumer.second, true);
        }
    }

    std::reverse(ordered.begin(), ordered.end());
    return ordered;
}

std::size_t FillXmlDoc(const InferenceEngine::ICNNNetwork& network, pugi::xml_document& doc) {
    const std::vector<CNNLayerPtr> ordered = TopologicalSort(network);
    pugi::xml_node netXml = doc.append_child("net");
    netXml.append_attribute("name").set_value(network.getName().c_str());

    pugi::xml_node layers = netXml.append_child("layers");

    std::map<CNNLayer::Ptr, size_t> matching;
    for (size_t i = 0; i < ordered.size(); i++) {
        matching[ordered[i]] = i;
    }

    const std::string dataName = "data";
    size_t dataOffset = 0;
    for (size_t i = 0; i < ordered.size(); ++i) {
        const CNNLayerPtr node = ordered[i];

        pugi::xml_node layer = layers.append_child("layer");
        const Precision precision = node->precision;
        layer.append_attribute("name").set_value(node->name.c_str());
        layer.append_attribute("type").set_value(node->type.c_str());
        layer.append_attribute("precision").set_value(precision.name());
        layer.append_attribute("id").set_value(i);

        const auto& params = node->params;
        if (!params.empty()) {
            pugi::xml_node data = layer.append_child(dataName.c_str());

            for (const auto& it : params) {
                data.append_attribute(it.first.c_str()).set_value(it.second.c_str());
            }
        }

        if (!node->insData.empty()) {
            pugi::xml_node input = layer.append_child("input");

            for (size_t iport = 0; iport < node->insData.size(); iport++) {
                const DataPtr d = node->insData[iport].lock();
                pugi::xml_node port = input.append_child("port");

                port.append_attribute("id").set_value(iport);

                for (auto dim : d->getDims()) {
                    port.append_child("dim").text().set(dim);
                }
            }
        }
        if (!node->outData.empty()) {
            pugi::xml_node output = layer.append_child("output");
            for (size_t oport = 0; oport < node->outData.size(); oport++) {
                pugi::xml_node port = output.append_child("port");

                port.append_attribute("id").set_value(node->insData.size() + oport);
                port.append_attribute("precision").set_value(node->outData[oport]->getPrecision().name());

                for (const auto dim : node->outData[oport]->getDims()) {
                    port.append_child("dim").text().set(dim);
                }
            }
        }
    }

    pugi::xml_node edges = netXml.append_child("edges");

    for (const auto& ord : ordered) {
        const CNNLayer::Ptr node = ord;

        if (!node->outData.empty()) {
            auto itFrom = matching.find(node);
            if (itFrom == matching.end()) {
                THROW_IE_EXCEPTION << "Internal error, cannot find " << node->name
                                   << " in matching container during serialization of IR";
            }
            for (size_t oport = 0; oport < node->outData.size(); oport++) {
                const DataPtr outData = node->outData[oport];
                for (const auto& inputTo : getInputTo(outData)) {
                    for (int iport = 0; iport < inputTo.second->insData.size(); iport++) {
                        if (inputTo.second->insData[iport].lock() == outData) {
                            auto itTo = matching.find(inputTo.second);
                            if (itTo == matching.end()) {
                                THROW_IE_EXCEPTION << "Broken edge form layer " << node->name << " to layer "
                                                   << inputTo.first << "during serialization of IR";
                            }
                            pugi::xml_node edge = edges.append_child("edge");
                            edge.append_attribute("from-layer").set_value(itFrom->second);
                            edge.append_attribute("from-port").set_value(oport + node->insData.size());

                            edge.append_attribute("to-layer").set_value(itTo->second);
                            edge.append_attribute("to-port").set_value(iport);
                        }
                    }
                }
            }
        }
    }

    return dataOffset;
}

void Serialize(const std::string& xmlPath, const InferenceEngine::ICNNNetwork& network) {
    pugi::xml_document doc;
    FillXmlDoc(network, doc);

    if (!doc.save_file(xmlPath.c_str())) {
        THROW_IE_EXCEPTION << "file '" << xmlPath << "' was not serialized";
    }
}
}  //  namespace Serialization
}  //  namespace InferenceEngine
