// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <deque>
#include <unordered_set>

#include <legacy/ie_layers.h>
#include <ie_icnn_network.hpp>

using namespace InferenceEngine;

IE_SUPPRESS_DEPRECATED_START

std::vector<InferenceEngine::CNNLayerPtr> TopologicalSort(const InferenceEngine::ICNNNetwork& network) {
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