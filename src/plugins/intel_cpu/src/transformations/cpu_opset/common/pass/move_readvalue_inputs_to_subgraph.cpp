// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "move_readvalue_inputs_to_subgraph.hpp"

#include <unordered_set>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"
#include "transformations/cpu_opset/common/op/read_value_with_subgraph.hpp"
#include "transformations/cpu_opset/common/op/sdpa.hpp"
#include "transformations/cpu_opset/common/op/submodel.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"
#include "transformations/utils/gen_pattern.hpp"
#include "transformations/utils/utils.hpp"

ov::intel_cpu::MoveReadValueInputsToSubgraph::MoveReadValueInputsToSubgraph() {
    MATCHER_SCOPE(MoveReadValueInputsToSubgraph);
    using namespace ov::pass::pattern;

    auto readvalue_pattern = pass::pattern::wrap_type<ov::op::v6::ReadValue>();

    ov::matcher_pass_callback callback = [=](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto readvalue = as_type_ptr<ov::opset6::ReadValue>(pattern_map.at(readvalue_pattern).get_node_shared_ptr());
        if (!readvalue || readvalue->get_input_size() != 1u) {
            return false;
        }

        if (readvalue->get_rt_info().count("DisableInitSubgraphFusing") &&
            readvalue->get_rt_info()["DisableInitSubgraphFusing"].as<bool>()) {
            return false;
        }

        NodeVector subgraph_nodes;
        std::unordered_set<std::shared_ptr<ov::Node>> visited_path_to_output;  // Cache nodes which connect to Output.
        std::unordered_set<std::shared_ptr<ov::Node>> visited_path_to_rv;  // Cache nodes which connect to ReadValue.
        NodeVector inputs = {};
        OutputVector outputs = {};

        // DFS, Check if current node's final successor is only ReadValue.
        std::function<void(std::shared_ptr<ov::Node>, bool&)> dfs = [&](const std::shared_ptr<ov::Node>& node,
                                                                        bool& found_output) {
            if (found_output) {
                return;
            }

            if (visited_path_to_output.find(node) != visited_path_to_output.end()) {
                found_output = true;
                return;
            }

            if (visited_path_to_rv.find(node) != visited_path_to_rv.end()) {
                return;
            }

            // node is Output
            if (node->get_output_target_inputs(0).size() == 0u) {
                found_output = true;
                return;
            }

            bool any_child_on_output_path = false;
            for (const auto& child : node->get_output_target_inputs(0)) {
                auto son = child.get_node()->shared_from_this();
                if (son == readvalue) {
                    continue;
                }

                bool new_found_output = false;
                dfs(son, new_found_output);
                if (new_found_output) {
                    any_child_on_output_path = true;
                }
            }

            if (any_child_on_output_path) {
                visited_path_to_output.insert(node);
                found_output = any_child_on_output_path;
            }
        };

        std::function<void(std::shared_ptr<ov::Node>)> reverse_dfs = [&](const std::shared_ptr<ov::Node>& node) {
            if (visited_path_to_output.find(node) != visited_path_to_output.end()) {
                inputs.emplace_back(node);
                return;
            }

            if (visited_path_to_rv.find(node) != visited_path_to_rv.end()) {
                return;
            }

            if (ov::op::util::is_parameter(node)) {
                inputs.emplace_back(node);
                return;
            }

            // Check if the current node has path(bypassing the ReadValue node) to the Output node via dfs algorithm.
            bool found_output = false;  // Flag: find Output node
            dfs(node, found_output);

            if (found_output) {
                inputs.emplace_back(node);
                visited_path_to_output.insert(node);
                return;
            }

            visited_path_to_rv.insert(node);

            // Cache to subgraph_nodes
            subgraph_nodes.emplace_back(node);

            for (size_t i = 0; i < node->get_input_size(); i++) {
                reverse_dfs(node->get_input_node_shared_ptr(i));
            }
        };

        // Reverse DFS ReadValue, find all suitable nodes and move them to subgraph_nodes.
        reverse_dfs(readvalue->get_input_node_shared_ptr(0));

        if (inputs.size() == 0 || subgraph_nodes.size() == 0) {
            return false;
        }

        // Subgraph's input
        auto params = ParameterVector{};
        for (const auto& inp : inputs) {
            auto param =
                std::make_shared<ov::op::v0::Parameter>(inp->get_element_type(), inp->get_output_partial_shape(0));
            params.push_back(param);
            for (const auto& child : inp->get_output_target_inputs(0)) {
                auto it = std::find(subgraph_nodes.begin(), subgraph_nodes.end(), child.get_node()->shared_from_this());
                if (it != subgraph_nodes.end()) {
                    child.replace_source_output(param);
                }
            }
        }

        // Subgraph's output
        auto last_node = readvalue->get_input_node_shared_ptr(0);
        auto output = std::make_shared<ov::op::v0::Result>(last_node);
        auto func = std::make_shared<Model>(ov::ResultVector({output}), params, "state_init_submodel");

        auto new_rv = std::make_shared<ov::intel_cpu::ReadValueWithSubgraph>(readvalue->get_variable(), func);

        for (size_t i = 0; i < inputs.size(); i++) {
            new_rv->set_input(inputs[i]->output(0), params[i]);
        }
        new_rv->set_output(output);

        // Replace ReadValue with ov::intel_cpu::ReadValueWithSubgraph
        ov::replace_node(readvalue, new_rv);
        ov::copy_runtime_info(subgraph_nodes, new_rv);
        new_rv->validate_and_infer_types();
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(readvalue_pattern, matcher_name);
    this->register_matcher(m, callback);
}
