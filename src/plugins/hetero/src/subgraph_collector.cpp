// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_collector.hpp"

#include <deque>

#include "graph_debug_dump.hpp"
#include "op/device_subgraph.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/util/common_util.hpp"
#include "transformations/utils/utils.hpp"

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

template <typename T>
static std::vector<T> addition(const std::vector<T>& vector1, const std::vector<T>& vector2) {
    std::vector<T> addition;
    std::copy_if(vector1.begin(), vector1.end(), std::back_inserter(addition), [&vector2](const T& arg) {
        return (std::find(vector2.begin(), vector2.end(), arg) == vector2.end());
    });
    return addition;
}

template <typename T>
static size_t get_index(const std::vector<T>& vector, const T& item) {
    const auto& it = std::find(vector.begin(), vector.end(), item);
    OPENVINO_ASSERT(it != vector.end());
    return static_cast<size_t>(std::distance(vector.begin(), it));
}

}  // namespace

std::shared_ptr<ov::Node> ov::hetero::SubgraphCollector::output_node(
    const ov::hetero::SubgraphCollector::Output& output) const {
    return output.get_node_shared_ptr();
}

std::shared_ptr<ov::Node> ov::hetero::SubgraphCollector::input_node(
    const ov::hetero::SubgraphCollector::Input& input) const {
    return output_node(input.get_source_output());
}

ov::hetero::SubgraphCollector::SubgraphCollector(const std::shared_ptr<ov::Model>& model,
                                                 const AffinitiesMap& affinities)
    : _ordered_ops{model->get_ordered_ops()},
      _origin_parameters(model->get_parameters()),
      _origin_results(model->get_results()),
      _origin_sinks(model->get_sinks()),
      _intermediate_parameters{},
      _intermediate_results(),
      _affinities{affinities},
      _node_input_dependencies{},
      _subgraph_inputs{},
      _subgraph_parameter_to_prev_result{} {
    init();
    split_cyclic_dependencies();
    _subgraph_ids = collect_subgraphs_ids();
}

bool ov::hetero::SubgraphCollector::is_graph_input_node(const ov::Node* node) const {
    return ov::op::util::is_parameter(node) || ov::op::util::is_constant(node);
}

void ov::hetero::SubgraphCollector::init() {
    // Get all subgraph inputs using just node affinities. Also collect transitive closure
    for (const auto& node : _ordered_ops) {
        if (is_graph_input_node(node.get())) {
            _subgraph_inputs.insert(Input{node.get(), 0});
            _node_input_dependencies[node].insert(Input{node.get(), 0});
        } else {
            auto inputs = node->inputs();
            auto& node_input_dependency = _node_input_dependencies[node];
            for (const auto& input : inputs) {
                node_input_dependency.insert(input);
                auto& input_dependency = _node_input_dependencies[input_node(input)];
                node_input_dependency.insert(input_dependency.begin(), input_dependency.end());
                if (_affinities.at(node) != _affinities.at(input_node(input))) {
                    if (ov::op::util::is_output(node)) {
                        _affinities[node] = _affinities.at(input_node(input));
                    } else {
                        _subgraph_inputs.insert(input);
                    }
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
        NodeMap<InputSet> node_subgraph_input_dependencies;
        // All inputs that depends on the same subgraph as node
        NodeMap<InputSet> node_subgraph_cyclic_iput_dependencies;
        for (const auto& node : _ordered_ops) {
            auto& node_subgraph_input_dependency = node_subgraph_input_dependencies[node];
            auto all_node_subgraph_inputs = intersection(_node_input_dependencies[node], _subgraph_inputs);
            for (const auto& subgraph_input : all_node_subgraph_inputs) {
                if (subgraph_ids[node] == subgraph_ids[subgraph_input.get_node()->shared_from_this()]) {
                    node_subgraph_input_dependency.emplace(subgraph_input);
                }
            }
            auto& node_subgraph_cyclic_input_dependency = node_subgraph_cyclic_iput_dependencies[node];
            for (const auto& subgraph_input : all_node_subgraph_inputs) {
                if (!is_graph_input_node(subgraph_input.get_node()) &&
                    subgraph_ids[node] == subgraph_ids[input_node(subgraph_input)]) {
                    node_subgraph_cyclic_input_dependency.emplace(subgraph_input);
                }
            }
        }

        for (const auto& node : _ordered_ops) {
            auto& node_subgraph_cyclic_input_dependency = node_subgraph_cyclic_iput_dependencies[node];
            if (!node_subgraph_cyclic_input_dependency.empty()) {
                // Collect all subgraph inputs that cyclic subgraph output depends on
                InputSet cyclic_inputs_dependencies;
                for (const auto& cyclicInput : node_subgraph_cyclic_input_dependency) {
                    for (const auto& input : node_subgraph_input_dependencies[input_node(cyclicInput)]) {
                        cyclic_inputs_dependencies.emplace(input);
                    }
                }
                for (const auto& input : node->inputs()) {
                    auto& input_node_subgraph_cyclic_input_dependency =
                        node_subgraph_cyclic_iput_dependencies[input_node(input)];
                    auto& input_node_subgraph_input_dependency = node_subgraph_input_dependencies[input_node(input)];
                    if (!intersects(node_subgraph_cyclic_input_dependency,
                                    input_node_subgraph_cyclic_input_dependency) &&
                        intersects(cyclic_inputs_dependencies, input_node_subgraph_input_dependency)) {
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
        InputVector inputs;
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
    // Sort subgraph inputs by order
    InputVector ordered_subgraph_inputs;
    for (auto& op : _ordered_ops) {
        for (auto& input : op->inputs()) {
            if (_subgraph_inputs.count(input)) {
                ordered_subgraph_inputs.push_back(input);
            }
        }
    }
    // Collect subgraph ordered subgraph outputs
    OutputVector ordered_subgraph_outputs;
    for (const auto& input : ordered_subgraph_inputs) {
        if (!is_graph_input_node(input.get_node())) {
            auto input_source_output = input.get_source_output();
            if (!ov::op::util::is_constant(input_source_output.get_node())) {
                ordered_subgraph_outputs.push_back(input_source_output);
            }
        }
    }
    // Break graph using insertion of result parameter split
    for (const auto& output : ordered_subgraph_outputs) {
        auto output_subgraph_id = _subgraph_ids.at(output.get_node_shared_ptr());
        auto inputs = output.get_target_inputs();
        // Collect input subsets from other subgraphs. Each subset of inputs belongs to the same subgraph
        std::map<SubgraphId, InputSet> input_subsets;
        for (const auto& input : inputs) {
            auto input_subgraph_id = _subgraph_ids.at(input.get_node()->shared_from_this());
            if (output_subgraph_id != input_subgraph_id) {
                input_subsets[input_subgraph_id].emplace(input);
            }
        }
        if (input_subsets.size()) {
            // Avoid duplicate results on the same output port
            auto result = std::make_shared<ov::op::v0::Result>(output);
            ov::copy_runtime_info(output.get_node_shared_ptr(), result);
            _subgraph_ids.emplace(result, output_subgraph_id);
            _intermediate_results.push_back(result);
            for (const auto& input_subset : input_subsets) {
                const auto& input_subgraph_id = input_subset.first;
                const auto& inputs = input_subset.second;
                // Avoid duplicate parameters in the same subgraph
                auto parameter =
                    std::make_shared<ov::op::v0::Parameter>(output.get_element_type(), output.get_partial_shape());
                _intermediate_parameters.push_back(parameter);
                for (const auto& input : inputs) {
                    output.remove_target_input(input);
                    ov::copy_runtime_info(input.get_node()->shared_from_this(), parameter);
                    input.replace_source_output(parameter->output(0));
                    _subgraph_ids.emplace(parameter, input_subgraph_id);
                    _subgraph_parameter_to_prev_result.emplace(parameter, result);
                }
            }
        }
    }
}

std::pair<ov::hetero::SubgraphsVector, ov::hetero::SubgraphsMappingInfo> ov::hetero::SubgraphCollector::run() {
    // Break graph using insertion of result parameter split
    split_subgraphs_by_parameter_results();

    // Extracts subgraph parameters, results and affinities
    auto subgraphs = collect_subgraphs();

    // Subgraph topological sort
    SubgraphsVector all_subgraphs;
    for (const auto& subgraph : subgraphs) {
        all_subgraphs.emplace_back(std::move(subgraph.second));
    }

    SubgraphsVector ordered_subgraphs;
    using NodeSet = std::unordered_set<std::shared_ptr<ov::Node>>;
    NodeSet prev_results;
    size_t subgraph_topo_sorts_step = 0;
    do {
        OPENVINO_ASSERT(subgraph_topo_sorts_step < subgraphs.size(), "Cannot sort subgraphs!");
        ++subgraph_topo_sorts_step;
        SubgraphsVector new_ordered_subgraphs;
        auto is_ordered_subgraph = [&](const Subgraph& subgraph) {
            auto& parameters = subgraph._parameters;
            return std::all_of(parameters.begin(),
                               parameters.end(),
                               [&](const ov::ParameterVector::value_type& parameter) {
                                   return (ov::util::contains(_origin_parameters, parameter) ||
                                           prev_results.count(_subgraph_parameter_to_prev_result[parameter]));
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

    // Get submodels mapping information
    auto mapping_info = get_subgraphs_mapping_info(ordered_subgraphs);

    return {ordered_subgraphs, mapping_info};
}

std::unordered_map<ov::hetero::SubgraphCollector::SubgraphId, ov::hetero::Subgraph>
ov::hetero::SubgraphCollector::collect_subgraphs() {
    std::unordered_map<SubgraphId, Subgraph> subgraphs;
    auto update_subgraph = [&](SubgraphId subgraph_id, const std::shared_ptr<ov::Node>& node) {
        auto& subgraph = subgraphs[subgraph_id];
        auto update_affinity = [&](const std::shared_ptr<ov::Node>& node) {
            auto it_affinity = _affinities.find(node);
            if (it_affinity != _affinities.end()) {
                subgraph._affinity = it_affinity->second;
            }
        };
        if (ov::op::util::is_output(node)) {
            subgraph._results.emplace_back(ov::as_type_ptr<ov::op::v0::Result>(node));
            update_affinity(input_node(node->input(0)));
        } else if (ov::op::util::is_parameter(node)) {
            subgraph._parameters.emplace_back(ov::as_type_ptr<ov::op::v0::Parameter>(node));
            update_affinity(output_node(node->output(0)));
        } else if (ov::op::util::is_sink(node)) {
            subgraph._sinks.emplace_back(ov::as_type_ptr<ov::op::Sink>(node));
            update_affinity(input_node(node->input(0)));
        }
    };
    // Update subgraph parameters
    for (auto& op_vec : {_origin_parameters, _intermediate_parameters}) {
        for (const auto& op : op_vec) {
            const auto node = std::dynamic_pointer_cast<ov::Node>(op);
            auto subgraph_id = _subgraph_ids.at(node);
            update_subgraph(subgraph_id, node);
        }
    }
    // Update subgraph results
    for (auto& op_vec : {_origin_results, _intermediate_results}) {
        for (const auto& op : op_vec) {
            const auto node = std::dynamic_pointer_cast<ov::Node>(op);
            auto subgraph_id = _subgraph_ids.at(node);
            update_subgraph(subgraph_id, node);
        }
    }
    // Update subgraph sinks
    for (const auto& op : _origin_sinks) {
        const auto node = std::dynamic_pointer_cast<ov::Node>(op);
        auto subgraph_id = _subgraph_ids.at(node);
        update_subgraph(subgraph_id, node);
    }
    return subgraphs;
}

ov::hetero::SubgraphsMappingInfo ov::hetero::SubgraphCollector::get_subgraphs_mapping_info(
    const SubgraphsVector& ordered_subgraphs) {
    SubgraphsMappingInfo info;
    // Prepare mapping between original inputs/outputs and compiled
    // submodels inputs/outputs. Example:
    // original input 0 -> submodel 0 input 0,
    // original input 1 -> submodel 1 input 0,
    // original output 0 -> submodel 1 output 0.
    //
    // Mapping is required only because before compilation
    // submodel may be preprocessed (if legacy API used),
    // so original inputs/outputs != submodels inputs/outputs
    info._inputs_to_submodels_inputs.resize(_origin_parameters.size());
    info._outputs_to_submodels_outputs.resize(_origin_results.size());
    for (size_t id = 0; id < ordered_subgraphs.size(); id++) {
        for (size_t i = 0; i < ordered_subgraphs[id]._parameters.size(); i++) {
            for (size_t j = 0; j < _origin_parameters.size(); j++)
                if (ordered_subgraphs[id]._parameters[i] == _origin_parameters[j])
                    info._inputs_to_submodels_inputs[j] = {id, i};
        }
        for (size_t i = 0; i < ordered_subgraphs[id]._results.size(); i++) {
            for (size_t j = 0; j < _origin_results.size(); j++)
                if (ordered_subgraphs[id]._results[i] == _origin_results[j])
                    info._outputs_to_submodels_outputs[j] = {id, i};
        }
    }
    // Prepare mapping between manually splitted inputs/outputs
    // to connect tensors between compiled submodels
    for (const auto& kvp : _subgraph_parameter_to_prev_result) {
        const auto& intermed_output = ov::as_type_ptr<ov::op::v0::Result>(kvp.second);
        const auto& intermed_input = ov::as_type_ptr<ov::op::v0::Parameter>(kvp.first);
        for (size_t out_subgraph_index = 0; out_subgraph_index < ordered_subgraphs.size(); out_subgraph_index++) {
            if (ov::util::contains(ordered_subgraphs[out_subgraph_index]._results, intermed_output)) {
                for (size_t in_subgraph_index = 0; in_subgraph_index < ordered_subgraphs.size(); in_subgraph_index++) {
                    if (in_subgraph_index == out_subgraph_index)
                        continue;
                    if (ov::util::contains(ordered_subgraphs[in_subgraph_index]._parameters, intermed_input)) {
                        auto out_idx = get_index(ordered_subgraphs[out_subgraph_index]._results, intermed_output);
                        auto in_idx = get_index(ordered_subgraphs[in_subgraph_index]._parameters, intermed_input);
                        info._submodels_input_to_prev_output[{in_subgraph_index, in_idx}] = {out_subgraph_index,
                                                                                             out_idx};
                    }
                }
            }
        }
    }
    return info;
}

void ov::hetero::merge_submodels(std::vector<std::shared_ptr<ov::Model>>& submodels,
                                 const std::map<NodeInfo, NodeInfo>& submodels_input_to_prev_output) {
    // Results which should not be present in final graph
    std::set<std::string> result_names_to_be_removed;
    // Remap port indexes to names, because order of them will be modified during merge
    std::map<std::pair<size_t, std::string>, std::pair<size_t, std::string>> input_to_prev_output;
    for (const auto& kvp : submodels_input_to_prev_output) {
        const auto& input_node = submodels[kvp.first.first]->inputs()[kvp.first.second].get_node();
        const auto& output_node = submodels[kvp.second.first]->outputs()[kvp.second.second].get_node();
        input_to_prev_output[{kvp.first.first, input_node->get_friendly_name()}] = {kvp.second.first,
                                                                                    output_node->get_friendly_name()};
        result_names_to_be_removed.insert(output_node->get_friendly_name());
    }
    int submodel_in_index = static_cast<int>(submodels.size()) - 1;
    while (submodel_in_index >= 0 && input_to_prev_output.size() > 0) {
        auto& submodel_in = submodels[submodel_in_index];
        size_t port_in_index = 0;
        while (port_in_index < submodel_in->get_parameters().size()) {
            auto parameter_to_replace = submodel_in->get_parameters()[port_in_index];
            auto item = input_to_prev_output.find({submodel_in_index, parameter_to_replace->get_friendly_name()});
            if (item == input_to_prev_output.end()) {
                port_in_index++;
                continue;
            }
            const auto& submodel_out_index = item->second.first;
            const auto& submodel_out_result_name = item->second.second;
            const auto& submodel_out = submodels.at(submodel_out_index);

            std::shared_ptr<ov::op::v0::Result> result_to_replace = nullptr;
            for (auto& result : submodel_out->get_results()) {
                if (result->get_friendly_name() == submodel_out_result_name) {
                    result_to_replace = result;
                }
            }
            OPENVINO_ASSERT(result_to_replace != nullptr);
            // Get all results from previous subgraph except already existed in next subgraph
            auto add_results = addition(submodel_out->get_results(), submodel_in->get_results());

            // Get all sinks from previous subgraph except already existed in next subgraph
            auto add_sinks = addition(submodel_out->get_sinks(), submodel_in->get_sinks());

            // Get all parameters from previous subgraph except already existed in next subgraph
            auto add_parameters = addition(submodel_out->get_parameters(), submodel_in->get_parameters());

            // Reconnect appropariate target inputs to the new source output
            auto result_source = result_to_replace->get_input_source_output(0);
            auto parameter_targets = parameter_to_replace->get_output_target_inputs(0);
            for (auto parameter_target : parameter_targets) {
                parameter_target.replace_source_output(result_source);
            }

            // Update parameter and results
            submodel_in->remove_parameter(parameter_to_replace);
            submodel_in->add_parameters(add_parameters);
            submodel_in->add_results(add_results);
            submodel_in->add_sinks(add_sinks);

            // Remove processed connection
            input_to_prev_output.erase(item);

            // Update incoming model since it is merged
            for (size_t i = 0; i < submodels.size(); i++) {
                if (submodels[i] == submodel_out) {
                    submodels[i] = submodel_in;
                }
            }

            // Start check ports from the beginning because number of ports are modified
            port_in_index = 0;
        }
        --submodel_in_index;
    }
    // Finally all subgraphs should be merged into single one
    OPENVINO_ASSERT(input_to_prev_output.size() == 0);
    std::set<size_t> distinct_submodels_index;
    for (size_t i = 0; i < submodels.size(); i++) {
        bool has_same_model = false;
        for (auto& index : distinct_submodels_index) {
            if (submodels[i] == submodels[index]) {
                has_same_model = true;
                break;
            }
        }
        if (!has_same_model) {
            distinct_submodels_index.insert(i);
        }
    }
    auto& result_model = submodels[0];
    for (size_t i = 1; i < submodels.size(); i++) {
        if (submodels[i] != result_model && distinct_submodels_index.count(i)) {
            result_model->add_parameters(submodels[i]->get_parameters());
            result_model->add_results(submodels[i]->get_results());
            result_model->add_sinks(submodels[i]->get_sinks());
        }
        submodels[i] = result_model;
    }
    OPENVINO_ASSERT(all_of(submodels.begin(), submodels.end(), [&](const std::shared_ptr<ov::Model>& submodel) {
        return submodel == result_model;
    }));
    // Cleanup intermidiate results
    for (size_t i = 0; i < result_model->get_results().size();) {
        auto& result = result_model->get_results()[i];
        if (result_names_to_be_removed.count(result->get_friendly_name())) {
            result_model->remove_result(result);
        } else {
            i++;
        }
    }
    submodels.resize(1);
}

std::pair<ov::hetero::SubgraphsVector, ov::hetero::SubgraphsMappingInfo> ov::hetero::get_model_subgraphs(
    const std::shared_ptr<ov::Model>& model,
    ov::SupportedOpsMap& supported_ops,
    const bool user_set_affinities,
    const bool dump_dot_files,
    const std::string default_device) {
    std::unordered_set<std::string> devices;
    ov::hetero::SubgraphCollector::AffinitiesMap affinities;
    ov::SupportedOpsMap debug_supported_ops{supported_ops};
    // Check that all nodes has user or plugin defined affinitie
    std::function<void(const std::shared_ptr<ov::Model>&, const std::string&)> collect_affinities =
        [&](const std::shared_ptr<ov::Model>& model, const std::string& default_device) {
            for (const auto& node : model->get_ordered_ops()) {
                auto it_affinity = supported_ops.find(node->get_friendly_name());
                if (it_affinity != supported_ops.end()) {
                    affinities[node] = it_affinity->second;
                    devices.emplace(it_affinity->second);
                } else if (!default_device.empty()) {
                    affinities[node] = default_device;
                    devices.emplace(default_device);
                    debug_supported_ops.insert({node->get_friendly_name(), default_device});
                } else if (!user_set_affinities) {
                    OPENVINO_THROW("Hetero device used default fallback policy, but some layers eg: \n(Name:",
                                   node->get_friendly_name(),
                                   ", Type: ",
                                   node->get_type_name(),
                                   ") were not able to be assigned on any pointed device.\n",
                                   "It happened because these layers are not supported in plugins by default.\n",
                                   "You need to implement custom layers to support them.");
                } else {
                    OPENVINO_THROW(
                        "Model passed to CompiledModel has affinity assigned, but some layers eg: \n(Name:",
                        node->get_friendly_name(),
                        ", Type: ",
                        node->get_type_name(),
                        ") were not assigned to any device.\n",
                        "It might happen if you assigned layers manually and missed some layers or\n",
                        "if you used some automatic assigning mode which decided that these layers are not\n",
                        "supported by any plugin");
                }
                if (dump_dot_files) {
                    if (auto multi_subgraph_op = ov::as_type_ptr<ov::op::util::MultiSubGraphOp>(node)) {
                        for (size_t i = 0; i < multi_subgraph_op->get_internal_subgraphs_size(); ++i) {
                            if (const auto& sub_graph = multi_subgraph_op->get_function(i)) {
                                collect_affinities(sub_graph, debug_supported_ops.at(node->get_friendly_name()));
                            }
                        }
                    }
                }
            }
        };
    collect_affinities(model, default_device);
    if (dump_dot_files) {
        ov::hetero::debug::dump_affinities(model, debug_supported_ops, devices);
    }

    // Init subgraph collector
    ov::hetero::SubgraphCollector subgraph_collector(model, affinities);

    if (dump_dot_files) {
        auto subgraph_ids = subgraph_collector.get_subgraph_ids();
        std::map<std::string, ov::hetero::SubgraphCollector::SubgraphId> map_id;
        std::function<void(const std::shared_ptr<ov::Model>&, const ov::hetero::SubgraphCollector::SubgraphId&)>
            collect_map_id = [&](const std::shared_ptr<ov::Model>& model,
                                 const ov::hetero::SubgraphCollector::SubgraphId& default_id) {
                for (const auto& node : model->get_ordered_ops()) {
                    ov::hetero::SubgraphCollector::SubgraphId subgraph_id;
                    if (subgraph_ids.count(node)) {
                        subgraph_id = subgraph_ids.at(node);
                    } else {
                        OPENVINO_ASSERT(default_id >= 0, "Invalid default id for node " + node->get_friendly_name());
                        subgraph_id = default_id;
                    }
                    map_id.emplace(node->get_friendly_name(), subgraph_id);
                    if (auto multi_subgraph_op = ov::as_type_ptr<ov::op::util::MultiSubGraphOp>(node)) {
                        for (size_t i = 0; i < multi_subgraph_op->get_internal_subgraphs_size(); ++i) {
                            if (const auto& sub_graph = multi_subgraph_op->get_function(i)) {
                                collect_map_id(sub_graph, subgraph_id);
                            }
                        }
                    }
                }
            };
        collect_map_id(model, -1);
        ov::hetero::debug::dump_subgraphs(model, debug_supported_ops, map_id);
    }

    // Get subgraphs sorted topologically and appropriate mapping information
    return subgraph_collector.run();
}

ov::hetero::SubgraphsMappingInfo ov::hetero::mask_model_subgraphs_by_ops(std::shared_ptr<ov::Model>& model,
                                                                         ov::SupportedOpsMap& supported_ops,
                                                                         const bool dump_dot_files,
                                                                         const std::string default_device) {
    const std::string subgraph_id_rt_info_name = "HETERO_SUBGRAPH_ID";
    const std::string input_id_rt_info_name = "HETERO_INPUT_ID";
    const std::string output_id_rt_info_name = "HETERO_OUTPUT_ID";
    const auto& name = model->get_friendly_name();

    SubgraphsVector ordered_subgraphs;
    SubgraphsMappingInfo mapping_info;
    std::tie(ordered_subgraphs, mapping_info) =
        get_model_subgraphs(model, supported_ops, false, dump_dot_files, default_device);

    SubmodelsVector submodels{ordered_subgraphs.size()};
    for (size_t i = 0; i < ordered_subgraphs.size(); i++) {
        const auto& subgraph = ordered_subgraphs.at(i);
        auto submodel_name = name + '_' + std::to_string(i);
        submodels[i] =
            std::make_shared<ov::Model>(subgraph._results, subgraph._sinks, subgraph._parameters, submodel_name);
        const auto& submodel = submodels[i];
        // Check whether model is subgraph already
        bool is_subgraph = ov::op::util::has_op_with_type<ov::hetero::op::DeviceSubgraph>(submodel);
        if (is_subgraph) {
            for (auto& op : submodel->get_ordered_ops()) {
                if (!ov::as_type_ptr<ov::hetero::op::DeviceSubgraph>(op) && !ov::op::util::is_parameter(op) &&
                    !ov::op::util::is_output(op) && !ov::op::util::is_sink(op)) {
                    is_subgraph = false;
                    break;
                }
            }
        }
        if (subgraph._affinity != default_device && !is_subgraph) {
            // Replace submodel by subgraph operation
            ParameterVector subgraph_parameters{submodel->inputs().size()};
            OutputVector args{submodel->inputs().size()};
            for (size_t j = 0; j < submodel->inputs().size(); j++) {
                auto const& input = submodel->input(j);
                subgraph_parameters[j] =
                    std::make_shared<ov::op::v0::Parameter>(input.get_element_type(), input.get_partial_shape());
                supported_ops[subgraph_parameters[j]->get_friendly_name()] = subgraph._affinity;
                args[j] = subgraph_parameters[j]->output(0);
            }
            auto subgraph_op = std::make_shared<ov::hetero::op::DeviceSubgraph>(args, submodel, subgraph._affinity);
            supported_ops[subgraph_op->get_friendly_name()] = subgraph._affinity;
            ResultVector subgraph_results{subgraph_op->outputs().size()};
            for (size_t j = 0; j < subgraph_op->outputs().size(); j++) {
                const auto& output = subgraph_op->output(j);
                subgraph_results[j] = std::make_shared<ov::op::v0::Result>(output);
                supported_ops[subgraph_results[j]->get_friendly_name()] = subgraph._affinity;
            }
            submodels[i] = std::make_shared<ov::Model>(subgraph_results, subgraph_parameters);
        }
        // Store original subgraph id
        for (auto& op : submodels[i]->get_ordered_ops()) {
            if (auto subgraph_op = ov::as_type_ptr<ov::hetero::op::DeviceSubgraph>(op)) {
                auto& rt_info = op->get_rt_info();
                rt_info[subgraph_id_rt_info_name] = i;

                const auto& parameters = submodels[i]->get_parameters();
                OPENVINO_ASSERT(parameters.size() == op->inputs().size());
                for (auto& input : op->inputs()) {
                    const auto& source_output = input.get_source_output().get_node()->shared_from_this();
                    if (auto parameter = ov::as_type_ptr<ov::op::v0::Parameter>(source_output)) {
                        input.get_rt_info()[input_id_rt_info_name] = get_index(parameters, parameter);
                    }
                }

                const auto& results = submodels[i]->get_results();
                OPENVINO_ASSERT(results.size() == op->outputs().size());
                for (auto& output : op->outputs()) {
                    for (auto& input : output.get_target_inputs()) {
                        auto target_input = input.get_node()->shared_from_this();
                        if (auto result = ov::as_type_ptr<ov::op::v0::Result>(target_input)) {
                            output.get_rt_info()[output_id_rt_info_name] = get_index(results, result);
                        }
                    }
                }
            }
        }
    }

    merge_submodels(submodels, mapping_info._submodels_input_to_prev_output);

    model = submodels[0];

    // Finally update mapping information according to the new operation order
    std::map<size_t, size_t> subgraph_id_map;
    std::map<size_t, std::map<size_t, size_t>> input_id_map;
    std::map<size_t, std::map<size_t, size_t>> output_id_map;
    size_t subgraph_op_id = 0;
    for (auto& op : model->get_ordered_ops()) {
        if (ov::as_type_ptr<ov::hetero::op::DeviceSubgraph>(op)) {
            auto& rt_info = op->get_rt_info();
            subgraph_id_map[rt_info[subgraph_id_rt_info_name].as<size_t>()] = subgraph_op_id;
            rt_info.erase(subgraph_id_rt_info_name);
            for (size_t j = 0; j < op->inputs().size(); j++) {
                auto& input_rt_info = op->input(j).get_rt_info();
                input_id_map[subgraph_op_id][input_rt_info[input_id_rt_info_name].as<size_t>()] = j;
                input_rt_info.erase(input_id_rt_info_name);
            }
            for (size_t j = 0; j < op->outputs().size(); j++) {
                auto& output_rt_info = op->output(j).get_rt_info();
                output_id_map[subgraph_op_id][output_rt_info[output_id_rt_info_name].as<size_t>()] = j;
                output_rt_info.erase(output_id_rt_info_name);
            }
            subgraph_op_id++;
        }
    }
    SubgraphsMappingInfo new_mapping_info;
    if (ordered_subgraphs.size() == subgraph_op_id) {
        // Only if all subgraphs were replaced by subgraph operations
        // we can provide appropriate mapping information
        // otherwise this information is unavailable
        auto get_new_subgraph_index = [&subgraph_id_map](const size_t old_subgraph_index) {
            OPENVINO_ASSERT(subgraph_id_map.count(old_subgraph_index));
            return subgraph_id_map.at(old_subgraph_index);
        };
        auto get_new_input_index = [&input_id_map](const size_t subgraph_index, const size_t old_input_index) {
            OPENVINO_ASSERT(input_id_map.at(subgraph_index).count(old_input_index));
            return input_id_map.at(subgraph_index).at(old_input_index);
        };
        auto get_new_output_index = [&output_id_map](const size_t subgraph_index, const size_t old_output_index) {
            OPENVINO_ASSERT(output_id_map.at(subgraph_index).count(old_output_index));
            return output_id_map.at(subgraph_index).at(old_output_index);
        };
        for (auto& item : mapping_info._inputs_to_submodels_inputs) {
            const auto& subgraph_index = get_new_subgraph_index(item.first);
            const auto& input_index = get_new_input_index(subgraph_index, item.second);
            new_mapping_info._inputs_to_submodels_inputs.push_back({subgraph_index, input_index});
        }
        for (auto& item : mapping_info._outputs_to_submodels_outputs) {
            const auto& subgraph_index = get_new_subgraph_index(item.first);
            const auto& output_index = get_new_output_index(subgraph_index, item.second);
            new_mapping_info._outputs_to_submodels_outputs.push_back({subgraph_index, output_index});
        }
        for (auto& item : mapping_info._submodels_input_to_prev_output) {
            const auto& subgraph_in_index = get_new_subgraph_index(item.first.first);
            const auto& input_index = get_new_input_index(subgraph_in_index, item.first.second);

            const auto& subgraph_out_index = get_new_subgraph_index(item.second.first);
            const auto& output_index = get_new_output_index(subgraph_out_index, item.second.second);

            new_mapping_info._submodels_input_to_prev_output[{subgraph_in_index, input_index}] = {subgraph_out_index,
                                                                                                  output_index};
        }
    }
    return new_mapping_info;
}
