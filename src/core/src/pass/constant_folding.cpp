// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/pass/constant_folding.hpp"

#include <ngraph/op/constant.hpp>

#include "ngraph/op/util/sub_graph_base.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph/opsets/opset3.hpp"
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

        // We have to check node for DisableConstantFolding because operations can override constant_folding
        // method, so we can't always rely on attribute check inside default node->constant_fold method
        if (node->get_rt_info().count(DisableConstantFolding::get_type_info_static()) == 0 &&
            node->constant_fold(replacements, node->input_values())) {
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
    // To avoid excess graph traversals we have to manually propagate DisableConstantFolding with some
    // temporary attribute which indicates that the node which is marked with this attribute can't be folded because
    // it is included into not foldable sub-graph.
    for (auto&& node : f->get_ordered_ops()) {
        const auto& inputs = node->input_values();
        auto& rt_info = node->get_rt_info();
        bool can_be_folded = true;
        if (rt_info.count(DisableConstantFolding::get_type_info_static())) {
            can_be_folded = false;
        } else if (is_type<ngraph::opset1::ShapeOf>(node) || is_type<ngraph::opset3::ShapeOf>(node)) {
            // In case if node is ShapeOf operation we stop propagation of can_be_folded attribute. We have to limit
            // propagation because we can't detect borders of shape_of sub-graphs, so we propagate can_be_folded
            // attribute through all nodes including nodes on data path. So to limit the spread of attribute to other
            // shape-of sub-graphs we do not propagate it through ShapeOf nodes.
            can_be_folded = true;
        } else if (std::any_of(inputs.cbegin(), inputs.cend(), [](const Output<Node>& output) {
                       const auto& rt_info = output.get_node()->get_rt_info();
                       return rt_info.count("can_be_folded") && !rt_info.at("can_be_folded").as<bool>();
                   })) {
            can_be_folded = false;
        }
        rt_info["can_be_folded"] = can_be_folded;
    }

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

        for (auto& output : curr_node->input_values()) {
            const auto& rt_info = output.get_node()->get_rt_info();
            auto can_be_folded = !rt_info.count("can_be_folded") || rt_info.at("can_be_folded").as<bool>();
            if (can_be_folded && output.get_tensor().has_and_set_bound()) {
                auto input_node = output.get_node_shared_ptr();
                auto replacement = std::make_shared<ngraph::op::Constant>(output.get_tensor().get_lower_value());
                if (replacement && !ov::is_type<ngraph::op::Constant>(input_node)) {
                    if (input_node->get_output_size() == 1) {
                        replacement->set_friendly_name(input_node->get_friendly_name());
                    } else {
                        replacement->set_friendly_name(input_node->get_friendly_name() + "." +
                                                       std::to_string(output.get_index()));
                    }
                    output.replace(replacement);
                    // Propagate runtime info attributes to replacement consumer nodes
                    copy_runtime_info_to_target_inputs(input_node, replacement);

                    rewritten = true;
                }
            } else {
                // continue searching
                const auto& input_node = output.get_node_shared_ptr();
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
