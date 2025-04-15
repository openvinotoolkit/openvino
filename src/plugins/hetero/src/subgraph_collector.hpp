// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include "openvino/runtime/common.hpp"
#include "openvino/runtime/core.hpp"

namespace ov {
namespace hetero {

using NodeInfo = std::pair<size_t /*submodel_idx*/, size_t /*node_idx*/>;

struct SubgraphsMappingInfo {
    std::vector<NodeInfo> _inputs_to_submodels_inputs;
    std::vector<NodeInfo> _outputs_to_submodels_outputs;
    std::map<NodeInfo, NodeInfo> _submodels_input_to_prev_output;
    bool empty() {
        return _inputs_to_submodels_inputs.empty() && _outputs_to_submodels_outputs.empty() &&
               _submodels_input_to_prev_output.empty();
    }
};
struct Subgraph {
    ov::ResultVector _results;
    ov::ParameterVector _parameters;
    ov::SinkVector _sinks;
    std::string _affinity;
};
using SubgraphsVector = std::vector<Subgraph>;
using SubmodelsVector = std::vector<std::shared_ptr<ov::Model>>;
class SubgraphCollector {
public:
    template <typename T>
    using NodeMap = std::unordered_map<std::shared_ptr<ov::Node>, T>;
    using AffinitiesMap = NodeMap<std::string>;
    using SubgraphId = int;
    using SubgraphIdsMap = NodeMap<SubgraphId>;
    using ParameterResultMap = NodeMap<std::shared_ptr<ov::Node>>;
    using Input = ov::Input<ov::Node>;
    using Output = ov::Output<ov::Node>;
    using InputSet = std::set<Input>;
    using OutputSet = std::set<Output>;
    using InputVector = std::vector<Input>;
    using OutputVector = std::vector<Output>;
    SubgraphCollector(const std::shared_ptr<ov::Model>& model, const AffinitiesMap& affinities);
    SubgraphIdsMap get_subgraph_ids() {
        return _subgraph_ids;
    }
    std::pair<SubgraphsVector, SubgraphsMappingInfo> run();

private:
    void init();
    bool is_graph_input_node(const ov::Node* node) const;
    void split_cyclic_dependencies();
    void split_subgraphs_by_parameter_results();
    SubgraphIdsMap collect_subgraphs_ids();
    std::unordered_map<SubgraphId, Subgraph> collect_subgraphs();
    SubgraphsMappingInfo get_subgraphs_mapping_info(const std::vector<Subgraph>& ordered_subgraphs);
    std::shared_ptr<ov::Node> input_node(const Input& input) const;
    std::shared_ptr<ov::Node> output_node(const Output& output) const;

    ov::NodeVector _ordered_ops;
    ov::ParameterVector _origin_parameters;
    ov::ResultVector _origin_results;
    ov::SinkVector _origin_sinks;
    ov::ParameterVector _intermediate_parameters;
    ov::ResultVector _intermediate_results;
    AffinitiesMap _affinities;
    NodeMap<InputSet> _node_input_dependencies;
    InputSet _subgraph_inputs;
    SubgraphIdsMap _subgraph_ids;
    ParameterResultMap _subgraph_parameter_to_prev_result;
};

void merge_submodels(SubmodelsVector& submodels, const std::map<NodeInfo, NodeInfo>& submodels_input_to_prev_output);

std::pair<SubgraphsVector, SubgraphsMappingInfo> get_model_subgraphs(const std::shared_ptr<ov::Model>& model,
                                                                     ov::SupportedOpsMap& supported_ops,
                                                                     const bool user_set_affinities = false,
                                                                     const bool dump_dot_files = false,
                                                                     const std::string default_device = "");

SubgraphsMappingInfo mask_model_subgraphs_by_ops(std::shared_ptr<ov::Model>& model,
                                                 ov::SupportedOpsMap& supported_ops,
                                                 const bool dump_dot_files = false,
                                                 const std::string default_device = "");

}  // namespace hetero
}  // namespace ov
