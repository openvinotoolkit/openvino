// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/utils/extract_subgraph.hpp"

#include <map>
#include <set>
#include <stack>
#include <unordered_set>
#include <utility>

#include "openvino/core/graph_util.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/sink.hpp"

namespace ov::util {

namespace {
ov::SinkVector collect_sinks(const ov::OutputVector& subgraph_outputs,
                             const ov::ParameterVector& subgraph_parameters) {
    std::unordered_set<ov::Node*> param_set;
    for (const auto& p : subgraph_parameters) {
        param_set.insert(p.get());
    }

    // First pass: collect all nodes in the subgraph by walking backward from outputs
    std::unordered_set<ov::Node*> subgraph_nodes;
    std::stack<ov::Node*> stack;
    for (const auto& output : subgraph_outputs) {
        stack.push(output.get_node());
    }
    while (!stack.empty()) {
        auto* node = stack.top();
        stack.pop();
        if (!subgraph_nodes.insert(node).second || param_set.count(node)) {
            continue;
        }
        for (size_t i = 0; i < node->get_input_size(); ++i) {
            stack.push(node->get_input_node_ptr(i));
        }
    }

    // Second pass: find Sink nodes that consume outputs of subgraph nodes
    ov::SinkVector sinks;
    for (auto* node : subgraph_nodes) {
        for (const auto& output : node->shared_from_this()->outputs()) {
            for (const auto& target_input : output.get_target_inputs()) {
                if (auto sink = std::dynamic_pointer_cast<ov::op::Sink>(target_input.get_node()->shared_from_this())) {
                    sinks.push_back(sink);
                }
            }
        }
    }
    return sinks;
}
}  // namespace

std::shared_ptr<ov::Model> extract_subgraph(const std::vector<ov::Input<ov::Node>>& subgraph_inputs,
                                            const ov::OutputVector& subgraph_outputs) {
    ov::OutputVector replaced_outputs;
    ov::ParameterVector subgraph_parameters;
    subgraph_parameters.reserve(subgraph_inputs.size());
    replaced_outputs.reserve(subgraph_inputs.size());

    for (const auto& input : subgraph_inputs) {
        const auto new_parameter =
            std::make_shared<ov::op::v0::Parameter>(input.get_element_type(), input.get_partial_shape());
        subgraph_parameters.push_back(new_parameter);

        const auto& source_output = input.get_source_output();
        replaced_outputs.push_back(source_output);
        OPENVINO_ASSERT(ov::replace_output_update_name(source_output, new_parameter->output(0)),
                        "extract_subgraph: failed to replace boundary source output '",
                        source_output.get_node()->get_friendly_name(),
                        "' at port ",
                        source_output.get_index(),
                        ". The requested subgraph input cannot be replaced safely.");
    }
    OPENVINO_ASSERT(subgraph_parameters.size() == replaced_outputs.size(),
                    "extract_subgraph: number of subgraph_parameters is not equal to the number of replaced_outputs");

    auto sinks = collect_sinks(subgraph_outputs, subgraph_parameters);
    auto subgraph =
        std::make_shared<ov::Model>(subgraph_outputs, sinks, subgraph_parameters)->clone();

    for (size_t i = 0; i < subgraph_parameters.size(); ++i) {
        subgraph_parameters[i]->output(0).replace(replaced_outputs[i]);
    }
    return subgraph;
}

namespace {
std::pair<std::vector<ov::Input<ov::Node>>, std::vector<ov::Output<ov::Node>>> resolve_subgraph_boundaries(
    const std::shared_ptr<ov::Model>& model,
    const std::multimap<std::string, size_t>& input_map,
    const std::multimap<std::string, size_t>& output_map) {
    std::vector<ov::Input<ov::Node>> inputs;
    std::vector<ov::Output<ov::Node>> outputs;

    std::set<std::string> resolved_input_names;
    std::set<std::string> resolved_output_names;

    for (const auto& node : model->get_ordered_ops()) {
        const auto& name = node->get_friendly_name();

        const auto input_range = input_map.equal_range(name);
        for (auto it = input_range.first; it != input_range.second; ++it) {
            inputs.push_back(node->input(it->second));
            resolved_input_names.insert(name);
        }

        const auto output_range = output_map.equal_range(name);
        for (auto it = output_range.first; it != output_range.second; ++it) {
            outputs.push_back(node->output(it->second));
            resolved_output_names.insert(name);
        }
    }

    for (const auto& kv : input_map) {
        OPENVINO_ASSERT(resolved_input_names.count(kv.first),
                        "extract_subgraph: node '",
                        kv.first,
                        "' specified in subgraph_inputs was not found in the model");
    }
    for (const auto& kv : output_map) {
        OPENVINO_ASSERT(resolved_output_names.count(kv.first),
                        "extract_subgraph: node '",
                        kv.first,
                        "' specified in subgraph_outputs was not found in the model");
    }

    return {inputs, outputs};
}
}  // namespace

std::shared_ptr<ov::Model> extract_subgraph(const std::shared_ptr<ov::Model>& model,
                                            const std::multimap<std::string, size_t>& subgraph_inputs,
                                            const std::multimap<std::string, size_t>& subgraph_outputs) {
    auto [inputs, outputs] = resolve_subgraph_boundaries(model, subgraph_inputs, subgraph_outputs);
    return extract_subgraph(inputs, outputs);
}

}  // namespace ov::util
