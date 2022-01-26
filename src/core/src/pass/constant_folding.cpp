// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/pass/constant_folding.hpp"

#include <ngraph/op/constant.hpp>

#include "ngraph/op/util/sub_graph_base.hpp"
#include "ngraph/rt_info.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;

bool ov::pass::ConstantFolding::run_on_model(const std::shared_ptr<ov::Model>& f) {
    bool rewritten = pre_calculated_values_folding(f);

    for (const auto& node : f->get_ordered_ops()) {
        if (rewritten) {
            node->validate_and_infer_types();
        }

        OutputVector replacements(node->get_output_size());
        if (node->constant_fold(replacements, node->input_values())) {
            NGRAPH_CHECK(replacements.size() == node->get_output_size(),
                         "constant_fold_default returned incorrect number of replacements for ",
                         node);

            for (size_t i = 0; i < replacements.size(); ++i) {
                auto node_output = node->output(i);
                auto replacement = replacements.at(i);
                if (replacement.get_node_shared_ptr() && (node_output != replacement)) {
                    if (replacements.size() == 1) {
                        replacement.get_node_shared_ptr()->set_friendly_name(node->get_friendly_name());
                    } else {
                        replacement.get_node_shared_ptr()->set_friendly_name(node->get_friendly_name() + "." +
                                                                             std::to_string(i));
                    }
                    node_output.replace(replacement);
                    // Propagate runtime info attributes to replacement consumer nodes
                    copy_runtime_info_to_target_inputs(node, replacement);

                    rewritten = true;
                }
            }
        } else {
            // recursively constant fold operators containing subgraphs (ie: TensorIterator, Loop)
            if (auto sub_graph_node = std::dynamic_pointer_cast<ngraph::op::util::MultiSubGraphOp>(node)) {
                size_t sub_graphs_num = sub_graph_node->get_internal_subgraphs_size();
                for (size_t sub_graph_ind = 0; sub_graph_ind < sub_graphs_num; ++sub_graph_ind) {
                    rewritten |= run_on_model(sub_graph_node->get_function(sub_graph_ind));
                }
            }
        }
    }

    return rewritten;
}

void ngraph::pass::ConstantFolding::copy_runtime_info_to_target_inputs(const std::shared_ptr<Node>& node,
                                                                       const Output<Node>& replacement) {
    for (auto& input : replacement.get_target_inputs()) {
        auto consumer = input.get_node()->shared_from_this();
        copy_runtime_info({node, consumer}, consumer);
    }
}

bool ngraph::pass::ConstantFolding::pre_calculated_values_folding(const std::shared_ptr<ngraph::Function>& f) {
    deque<shared_ptr<Node>> nodes;
    set<shared_ptr<Node>> visited;
    for (auto& r : f->get_results())
        nodes.push_back(r);
    for (auto& r : f->get_sinks())
        nodes.emplace_back(r);

    bool rewritten = false;
    while (!nodes.empty()) {
        auto curr_node = nodes.front();
        nodes.pop_front();
        if (visited.count(curr_node) || ov::is_type<ngraph::op::Constant>(curr_node))
            continue;
        visited.insert(curr_node);

        for (auto& input_value : curr_node->input_values()) {
            // Check that ConstantFolding is not disabled on this path
            std::vector<Node*> order;
            auto status = ngraph::could_propagate(input_value, order);
            if (status) {
                for (const auto& node : order) {
                    const auto& rt_info = node->get_rt_info();
                    if (rt_info.count(DisableConstantFolding::get_type_info_static())) {
                        status = false;
                        break;
                    }
                }
            }

            if (status && input_value.get_tensor().has_and_set_bound()) {
                auto input_node = input_value.get_node_shared_ptr();
                auto replacement = std::make_shared<ngraph::op::Constant>(input_value.get_tensor().get_lower_value());
                if (replacement && !ov::is_type<ngraph::op::Constant>(input_node)) {
                    if (input_node->get_output_size() == 1) {
                        replacement->set_friendly_name(input_node->get_friendly_name());
                    } else {
                        replacement->set_friendly_name(input_node->get_friendly_name() + "." +
                                                       std::to_string(input_value.get_index()));
                    }
                    input_value.replace(replacement);
                    // Propagate runtime info attributes to replacement consumer nodes
                    copy_runtime_info_to_target_inputs(input_node, replacement);

                    rewritten = true;
                }
            } else {
                // continue searching
                const auto& input_node = input_value.get_node_shared_ptr();
                nodes.push_front(input_node);
            }
        }
    }
    return rewritten;
}

void ov::pass::disable_constant_folding(const std::shared_ptr<Node>& node) {
    node->get_rt_info().emplace(DisableConstantFolding::get_type_info_static(), DisableConstantFolding{});
}

void ov::pass::enable_constant_folding(const std::shared_ptr<Node>& node) {
    node->get_rt_info().erase(DisableConstantFolding::get_type_info_static());
}

bool ov::pass::constant_folding_is_disabled(const std::shared_ptr<Node>& node) {
    return node->get_rt_info().count(DisableConstantFolding::get_type_info_static());
}
