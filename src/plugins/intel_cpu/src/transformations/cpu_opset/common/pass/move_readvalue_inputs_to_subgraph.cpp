// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/cpu_opset/common/op/read_value_with_subgraph.hpp"
#include "transformations/cpu_opset/common/op/sdpa.hpp"
#include "move_readvalue_inputs_to_subgraph.hpp"

#include <unordered_set>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"
#include "transformations/utils/gen_pattern.hpp"
#include "transformations/utils/utils.hpp"
#include "transformations/cpu_opset/common/op/submodel.hpp"

// Check if skip pattern:
//         ReadValue
//             |
//       Convert(Optional)
//             |
// ScaledDotProductAttentionWithKVCache
//             |
//       Convert(Optional)
//             |
//          Assign
inline bool skip_pattern_sdpa(std::shared_ptr<ov::op::v6::ReadValue> readvalue) {
    for (const auto& node : readvalue->get_output_target_inputs(0)) {
        auto sdpa =
            ov::as_type_ptr<ov::intel_cpu::ScaledDotProductAttentionWithKVCache>(node.get_node()->shared_from_this());
        if (!sdpa) {
            // Check whether ReadValue's grandson is SDPA.
            for (const auto& child : node.get_node()->get_output_target_inputs(0)) {
                sdpa = ov::as_type_ptr<ov::intel_cpu::ScaledDotProductAttentionWithKVCache>(
                    child.get_node()->shared_from_this());
                if (sdpa) {
                    break;
                }
            }
        }

        if (sdpa) {
            for (size_t port = 0; port < sdpa->get_output_size(); port++) {
                for (const auto& child : sdpa->get_output_target_inputs(port)) {
                    auto assign = ov::as_type_ptr<ov::op::v6::Assign>(child.get_node()->shared_from_this());
                    if (!assign) {
                        // Check whether SDPA's grandson is Assign.
                        for (const auto& grandson : child.get_node()->get_output_target_inputs(0)) {
                            assign = ov::as_type_ptr<ov::op::v6::Assign>(grandson.get_node()->shared_from_this());
                            if (assign) {
                                break;
                            }
                        }
                    }
                    if (assign) {
                        if (assign->get_variable_id() == readvalue->get_variable_id()) {
                            return true;
                        }
                    }
                }
            }
        }
    }
    return false;
}

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

        // TODO: Temporarily skip this pattern. If MemoryInputSDPA supports Subgraph in the future, it may be deleted.
        if (skip_pattern_sdpa(readvalue)) {
            return false;
        }

        bool found_output = false;  // Flag: find Output node or out of max depth.
        std::string root_name = readvalue->get_friendly_name();

        NodeVector subgraph_nodes;
        std::unordered_set<std::string> subgraph_node_names;
        std::unordered_set<std::shared_ptr<ov::Node>> visited;  // Visited flag.
        NodeVector inputs = {};
        OutputVector outputs = {};

        // DFS, Check if current node's final successor is only ReadValue.
        std::function<void(std::shared_ptr<ov::Node>)> dfs = [&](std::shared_ptr<ov::Node> node) {
            if (visited.find(node) != visited.end()) {
                return;
            }

            // Output node.
            if (node->get_output_target_inputs(0).size() == 0u) {
                found_output = true;
                return;
            }

            for (const auto& child : node->get_output_target_inputs(0)) {
                auto son = child.get_node()->shared_from_this();
                if (son->get_friendly_name() == root_name) {
                    continue;
                }
                dfs(son);
            }
        };

        std::function<void(std::shared_ptr<ov::Node>)> reverse_dfs = [&](std::shared_ptr<ov::Node> node) {
            if (ov::op::util::is_parameter(node)) {
                inputs.emplace_back(node);
                return;
            }

            // Check whether current node have same successor[root_node_name].
            found_output = false;
            dfs(node);
            visited.insert(node);
            if (found_output) {
                inputs.emplace_back(node);
                return;
            }

            for (size_t i = 0; i < node->get_input_size(); i++) {
                reverse_dfs(node->get_input_node_shared_ptr(i));
            }

            // Cache to subgraph_nodes
            subgraph_nodes.emplace_back(node);
            subgraph_node_names.insert(node->get_friendly_name());
        };

        // Reverse DFS ReadValue, find all suitable nodes and move them to subgraph_nodes.
        reverse_dfs(readvalue->get_input_node_shared_ptr(0));

        if (inputs.size() == 0 || subgraph_nodes.size() == 0) {
            return false;
        }

        auto new_rv = std::make_shared<ov::intel_cpu::ReadValueWithSubgraph>(readvalue->get_variable());

        // Subgraph's input
        auto params = ParameterVector{};
        for (auto inp : inputs) {
            auto param = std::make_shared<ov::op::v0::Parameter>(inp->get_element_type(), inp->get_output_partial_shape(0));
            params.push_back(param);
            for (const auto& child : inp->get_output_target_inputs(0)) {
                if (subgraph_node_names.find(child.get_node()->shared_from_this()->get_friendly_name()) !=
                    subgraph_node_names.end()) {
                    child.replace_source_output(param);
                }
            }
        }

        // Subgraph's output
        auto last_node = readvalue->get_input_node_shared_ptr(0);
        auto output = std::make_shared<ov::op::v0::Result>(last_node);
        auto func = std::make_shared<Model>(ov::ResultVector({output}), params, "state_init_submodel");
        new_rv->set_function(func);

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