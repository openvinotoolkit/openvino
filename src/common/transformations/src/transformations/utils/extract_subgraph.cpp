// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/utils/extract_subgraph.hpp"

#include <map>
#include <set>
#include <utility>

#include "openvino/core/graph_util.hpp"
#include "openvino/op/parameter.hpp"

namespace ov {
namespace op {
namespace util {

std::shared_ptr<ov::Model> extract_subgraph(const std::shared_ptr<ov::Model>& model,
                                            const std::vector<ov::Input<ov::Node>>& subgraph_inputs,
                                            const std::vector<ov::Output<ov::Node>>& subgraph_outputs) {
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
        ov::replace_output_update_name(source_output, new_parameter->output(0));
    }
    OPENVINO_ASSERT(subgraph_parameters.size() == replaced_outputs.size(),
                    "extract_subgraph: number of subgraph_parameters is not equal to the number of replaced_outputs");

    auto subgraph = std::make_shared<ov::Model>(subgraph_outputs, subgraph_parameters)->clone();

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
    return extract_subgraph(model, inputs, outputs);
}

}  // namespace util
}  // namespace op
}  // namespace ov
