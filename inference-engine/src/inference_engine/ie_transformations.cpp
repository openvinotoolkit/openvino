// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_transformations.hpp"
#include <ngraph/pass/low_latency.hpp>
#include <ngraph/pass/freeze_nodes.hpp>
#include <ngraph/pass/manager.hpp>

using namespace InferenceEngine;

void InferenceEngine::LowLatency(InferenceEngine::CNNNetwork &network) {
    auto function = network.getFunction();
    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::LowLatency>();
    manager.run_passes(function);
}

void InferenceEngine::FreezeNodes(InferenceEngine::CNNNetwork &network,
                                  const std::map<std::string, std::vector<std::vector<char>>>& nodes_to_replace) {
    auto function = network.getFunction();
    using values_for_node = std::vector<std::vector<char>>;
    ngraph::NodeVector nodes_to_freeze;
    std::vector<values_for_node> replacing_values;
    for (const auto& node : function->get_ordered_ops()) {
        if (nodes_to_replace.count(node->get_friendly_name())) {
            nodes_to_freeze.push_back(node);
            replacing_values.push_back(nodes_to_replace.at(node->get_friendly_name()));
        }
    }
    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::FreezeNodes>(nodes_to_freeze, replacing_values);
    manager.run_passes(function);
}
