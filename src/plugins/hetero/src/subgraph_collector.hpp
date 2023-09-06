// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include "openvino/runtime/common.hpp"
#include "openvino/runtime/core.hpp"

namespace ov {
namespace hetero {

class SubgraphCollector {
public:
    struct Subgraph {
        ov::ResultVector _results;
        ov::ParameterVector _parameters;
        ov::SinkVector _sinks;
        std::string _affinity;
    };
    template <typename T>
    using NodeMap = std::unordered_map<std::shared_ptr<ov::Node>, T>;
    using NodeSet = std::unordered_set<std::shared_ptr<ov::Node>>;
    using AffinitiesMap = NodeMap<std::string>;
    using SubgraphId = int;
    using SubgraphIdsMap = NodeMap<SubgraphId>;
    using ParameterResultMap = NodeMap<std::shared_ptr<ov::Node>>;
    using Input = ov::Input<ov::Node>;
    using InputSet = std::set<Input>;

    SubgraphCollector(const std::shared_ptr<ov::Model>& model, const AffinitiesMap& affinities);
    SubgraphIdsMap get_subgraph_ids() {
        return _subgraph_ids;
    }
    ParameterResultMap get_subgraph_parameter_to_prev_result() {
        return _subgraph_parameter_to_prev_result;
    }
    std::vector<Subgraph> get_ordered_subgraphs();

private:
    void init();
    void split_cyclic_dependencies();
    void split_subgraphs_by_parameter_results();
    SubgraphIdsMap collect_subgraphs_ids();
    std::unordered_map<SubgraphId, Subgraph> collect_subgraphs();
    std::shared_ptr<ov::Node> input_node(const Input& input) const;

    ov::NodeVector _ordered_ops;
    AffinitiesMap _affinities;
    NodeSet _graph_input_nodes;
    NodeMap<InputSet> _node_input_dependencies;
    InputSet _subgraph_inputs;
    SubgraphIdsMap _subgraph_ids;
    ParameterResultMap _subgraph_parameter_to_prev_result;
};

}  // namespace hetero
}  // namespace ov
