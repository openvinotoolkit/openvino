// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_collector.hpp"

#include <deque>

#include "openvino/core/except.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/util/op_types.hpp"

namespace {

template <typename Set>
static Set intersection(const Set& lhs, const Set& rhs) {
    Set result;
    const auto& min_size_set = (lhs.size() < rhs.size()) ? lhs : rhs;
    const auto& max_size_set = (lhs.size() >= rhs.size()) ? lhs : rhs;
    for (auto&& val : min_size_set)
        if (max_size_set.find(val) != max_size_set.end())
            result.insert(val);
    return result;
}

template <typename Set>
static bool intersects(const Set& lhs, const Set& rhs) {
    const auto& min_size_set = (lhs.size() < rhs.size()) ? lhs : rhs;
    const auto& max_size_set = (lhs.size() >= rhs.size()) ? lhs : rhs;
    for (auto&& val : min_size_set)
        if (max_size_set.find(val) != max_size_set.end())
            return true;
    return false;
}
}  // namespace

std::shared_ptr<ov::Node> ov::hetero::SubgraphCollector::input_node(
    const ov::hetero::SubgraphCollector::Input& input) const {
    return input.get_source_output().get_node_shared_ptr();
}

ov::hetero::SubgraphCollector::SubgraphCollector(const std::shared_ptr<ov::Model>& model,
                                                 const AffinitiesMap& affinities)
    : _ordered_ops{model->get_ordered_ops()},
      _affinities{affinities},
      _graph_input_nodes{},
      _node_input_dependencies{},
      _subgraph_inputs{},
      _subgraph_parameter_to_prev_result{} {
    init();
    split_cyclic_dependencies();
    _subgraph_ids = collect_subgraphs_ids();
}

void ov::hetero::SubgraphCollector::init() {
    // Get all subgraph inputs using just node affinities. Also collect transitive closure
    for (const auto& node : _ordered_ops) {
        if (ov::op::util::is_parameter(node) || ov::op::util::is_constant(node)) {
            _graph_input_nodes.insert(node);
            _subgraph_inputs.insert(Input{node.get(), 0});
            _node_input_dependencies[node].insert(Input{node.get(), 0});
        } else {
            auto inputs = node->inputs();
            auto& node_input_dependency = _node_input_dependencies[node];
            for (const auto& input : inputs) {
                node_input_dependency.insert(input);
                auto& inputDependency = _node_input_dependencies[input_node(input)];
                node_input_dependency.insert(inputDependency.begin(), inputDependency.end());
                if (_affinities.at(node) != _affinities.at(input_node(input))) {
                    _subgraph_inputs.insert(input);
                }
            }
        }
    }
}
void ov::hetero::SubgraphCollector::split_cyclic_dependencies() {
    // Split cyclic dependencies.
    for (size_t prev_subgraphs = 0, cyclic_split_step = 0; prev_subgraphs != _subgraph_inputs.size();
         ++cyclic_split_step) {
        OPENVINO_ASSERT(cyclic_split_step < _ordered_ops.size(), "Cannot resolve cycles during submodels split!");
        prev_subgraphs = _subgraph_inputs.size();
        auto subgraph_ids = collect_subgraphs_ids();
        // All inputs that belong to the same subgraph as node
        std::unordered_map<std::shared_ptr<ov::Node>, InputSet> node_subgraph_input_dependencies;
        // All inputs that depends on the same subgraph as node
        std::unordered_map<std::shared_ptr<ov::Node>, InputSet> node_subgraph_cyclic_iput_dependencies;
        for (const auto& node : _ordered_ops) {
            auto& node_subgraph_input_dependency = node_subgraph_input_dependencies[node];
            auto all_node_subgraph_inputs = intersection(_node_input_dependencies[node], _subgraph_inputs);
            for (const auto& subgraphInput : all_node_subgraph_inputs) {
                if (subgraph_ids[node] == subgraph_ids[subgraphInput.get_node()->shared_from_this()]) {
                    node_subgraph_input_dependency.emplace(subgraphInput);
                }
            }
            auto& node_subgraph_cyclic_input_dependency = node_subgraph_cyclic_iput_dependencies[node];
            for (const auto& subgraphInput : all_node_subgraph_inputs) {
                if (!ov::op::util::is_parameter(subgraphInput.get_node()) &&
                    !ov::op::util::is_constant(subgraphInput.get_node()) &&
                    subgraph_ids[node] == subgraph_ids[input_node(subgraphInput)]) {
                    node_subgraph_cyclic_input_dependency.emplace(subgraphInput);
                }
            }
        }

        for (const auto& node : _ordered_ops) {
            auto& node_subgraph_cyclic_input_dependency = node_subgraph_cyclic_iput_dependencies[node];
            if (!node_subgraph_cyclic_input_dependency.empty()) {
                // Collect all subgraph inputs that cyclic subgraph output depends on
                InputSet cyclicInputsDependencies;
                for (const auto& cyclicInput : node_subgraph_cyclic_input_dependency) {
                    for (const auto& input : node_subgraph_input_dependencies[input_node(cyclicInput)]) {
                        cyclicInputsDependencies.emplace(input);
                    }
                }
                for (const auto& input : node->inputs()) {
                    auto& input_node_subgraph_cyclic_input_dependency =
                        node_subgraph_cyclic_iput_dependencies[input_node(input)];
                    auto& input_node_subgraph_input_dependency = node_subgraph_input_dependencies[input_node(input)];
                    if (!intersects(node_subgraph_cyclic_input_dependency,
                                    input_node_subgraph_cyclic_input_dependency) &&
                        intersects(cyclicInputsDependencies, input_node_subgraph_input_dependency)) {
                        _subgraph_inputs.insert(input);
                    }
                }
            }
        }
    }
}

ov::hetero::SubgraphCollector::SubgraphIdsMap ov::hetero::SubgraphCollector::collect_subgraphs_ids() {
    std::deque<SubgraphId> subgraph_ids;
    NodeMap<SubgraphId*> subgraph_id_ptrs;
    for (const auto& node : _ordered_ops) {
        auto all_node_inputs = node->inputs();
        std::vector<Input> inputs;
        for (const auto& input : all_node_inputs) {
            if (_subgraph_inputs.find(input) == _subgraph_inputs.end()) {
                inputs.emplace_back(std::move(input));
            }
        }
        if (inputs.empty()) {
            subgraph_ids.push_back(static_cast<SubgraphId>(subgraph_ids.size()));
            subgraph_id_ptrs.emplace(node, &(subgraph_ids.back()));
        } else {
            auto first_input_subgraph_id_ptr = subgraph_id_ptrs[input_node(inputs.front())];
            for (const auto& input : inputs) {
                auto input_id = *subgraph_id_ptrs[input_node(input)];
                for (auto& subgraph_id : subgraph_ids) {
                    if (subgraph_id == input_id) {
                        subgraph_id = *first_input_subgraph_id_ptr;
                    }
                }
            }
            subgraph_id_ptrs.emplace(node, first_input_subgraph_id_ptr);
        }
    }
    SubgraphIdsMap result;
    for (const auto& subgraph_id_ptr : subgraph_id_ptrs) {
        result.emplace(subgraph_id_ptr.first, *(subgraph_id_ptr.second));
    }
    return result;
}

void ov::hetero::SubgraphCollector::split_subgraphs_by_parameter_results() {
    // Break graph using insertion of result parameter split
    std::vector<std::shared_ptr<ov::op::v0::Result>> results;
    {
        std::set<ov::Output<ov::Node>> subgraph_outputs;
        for (const auto& input : _subgraph_inputs) {
            if (!ov::op::util::is_parameter(input.get_node()) && !ov::op::util::is_constant(input.get_node())) {
                auto input_source_output = input.get_source_output();
                if (!ov::op::util::is_parameter(input_source_output.get_node()) &&
                    !ov::op::util::is_constant(input_source_output.get_node())) {
                    subgraph_outputs.insert(input_source_output);
                }
            }
        }
        for (const auto& output : subgraph_outputs) {
            auto output_subgraph_id = _subgraph_ids.at(output.get_node_shared_ptr());
            auto inputs = output.get_target_inputs();
            // Collect input subsets from other subgraphs. Each subset of inputs belongs to the same subgraph
            std::map<int, std::set<ov::Input<ov::Node>>> input_subsets;
            for (const auto& input : inputs) {
                auto input_subgraph_id = _subgraph_ids.at(input.get_node()->shared_from_this());
                if (output_subgraph_id != input_subgraph_id) {
                    input_subsets[input_subgraph_id].emplace(input);
                }
            }
            // Avoid duplicate results on the same output port
            auto result = std::make_shared<ov::op::v0::Result>(output);
            const auto name = output.get_node()->get_friendly_name() + "_" + std::to_string(output.get_index());
            result->set_friendly_name(name + "_result");
            ov::copy_runtime_info(output.get_node_shared_ptr(), result);
            _subgraph_ids.emplace(result, output_subgraph_id);
            results.push_back(result);
            for (const auto& input_subset : input_subsets) {
                // Avoid duplicate parameters in the same subgraph
                auto parameter =
                    std::make_shared<ov::op::v0::Parameter>(output.get_element_type(), output.get_partial_shape());
                parameter->set_friendly_name(name + "_parameter");
                for (const auto& input : input_subset.second) {
                    output.remove_target_input(input);
                    ov::copy_runtime_info(input.get_node()->shared_from_this(), parameter);
                    input.replace_source_output(parameter->output(0));
                    _subgraph_ids.emplace(parameter, input_subset.first);
                    _subgraph_parameter_to_prev_result.emplace(parameter, result);
                }
            }
        }
    }
}

std::vector<ov::hetero::SubgraphCollector::Subgraph> ov::hetero::SubgraphCollector::get_ordered_subgraphs() {
    // Break graph using insertion of result parameter split
    split_subgraphs_by_parameter_results();

    // Extracts subgraph parameters, results and affinities
    auto subgraphs = collect_subgraphs();

    // Subgraph topological sort
    std::vector<Subgraph> all_subgraphs;
    for (const auto& subgraph : subgraphs) {
        all_subgraphs.emplace_back(std::move(subgraph.second));
    }

    std::vector<Subgraph> ordered_subgraphs;
    NodeSet prev_results;
    size_t subgraph_topo_sorts_step = 0;
    do {
        OPENVINO_ASSERT(subgraph_topo_sorts_step < subgraphs.size(), "Cannot sort subgraphs!");
        ++subgraph_topo_sorts_step;
        std::vector<Subgraph> new_ordered_subgraphs;
        auto is_ordered_subgraph = [&](const Subgraph& subgraph) {
            auto& parameters = subgraph._parameters;
            return std::all_of(
                parameters.begin(),
                parameters.end(),
                [&](const ov::ParameterVector::value_type& parameter) {
                    return (_graph_input_nodes.find(parameter) != _graph_input_nodes.end()) ||
                           (prev_results.find(_subgraph_parameter_to_prev_result[parameter]) != prev_results.end());
                });
        };
        std::remove_copy_if(std::begin(all_subgraphs),
                            std::end(all_subgraphs),
                            std::back_inserter(new_ordered_subgraphs),
                            [&](const Subgraph& subgraph) {
                                return !is_ordered_subgraph(subgraph);
                            });
        all_subgraphs.erase(std::remove_if(std::begin(all_subgraphs), std::end(all_subgraphs), is_ordered_subgraph),
                            std::end(all_subgraphs));
        for (const auto& subgraph : new_ordered_subgraphs) {
            for (const auto& result : subgraph._results) {
                prev_results.insert(result);
            }
        }
        std::move(std::begin(new_ordered_subgraphs),
                  std::end(new_ordered_subgraphs),
                  std::back_inserter(ordered_subgraphs));
    } while (!all_subgraphs.empty());
    return ordered_subgraphs;
}

std::unordered_map<ov::hetero::SubgraphCollector::SubgraphId, ov::hetero::SubgraphCollector::Subgraph>
ov::hetero::SubgraphCollector::collect_subgraphs() {
    std::unordered_map<SubgraphId, Subgraph> subgraphs;
    // Extracts subgraph parameters, results and affinities
    for (const auto& subgraph_id_ptr_value : _subgraph_ids) {
        auto node = subgraph_id_ptr_value.first;
        auto& subgraph = subgraphs[subgraph_id_ptr_value.second];
        if (ov::op::util::is_output(node)) {
            subgraph._results.emplace_back(ov::as_type_ptr<ov::op::v0::Result>(node->shared_from_this()));
        } else if (ov::op::util::is_parameter(node)) {
            subgraph._parameters.emplace_back(ov::as_type_ptr<ov::op::v0::Parameter>(node->shared_from_this()));
        } else if (ov::op::util::is_sink(node)) {
            subgraph._sinks.emplace_back(ov::as_type_ptr<ov::op::Sink>(node->shared_from_this()));
        }
        auto it_affinity = _affinities.find(node);
        if (it_affinity != _affinities.end()) {
            subgraph._affinity = it_affinity->second;
        }
    }
    return subgraphs;
}
