// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <vector>

#include "itt.hpp"
#include <transformations/common_optimizations/simplify_shape_of_sub_graph.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/rt_info.hpp>
#include <transformations/common_optimizations/eliminate_unsqueeze_gather.hpp>
#include <transformations/utils/utils.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::SharedShapeOf, "SharedShapeOf", 0);

bool ngraph::pass::SharedShapeOf::run_on_function(std::shared_ptr<ngraph::Function> f) {
    RUN_ON_FUNCTION_SCOPE(SharedShapeOf);
    bool graph_rewritten = false;

    std::map<ngraph::Output<Node>, std::vector<std::shared_ptr<ngraph::opset1::ShapeOf>>> source_to_shape_of;
    std::map<ngraph::Output<Node>, std::vector<std::shared_ptr<ngraph::opset3::ShapeOf>>> source_to_shape_of_3;
    for (const auto & node : f->get_ordered_ops()) {
        // Recursively apply transformation for sub-graph based operations
        if (auto sub_graph_node = std::dynamic_pointer_cast<op::util::SubGraphOp>(node)) {
            if (auto sub_graph = sub_graph_node->get_function()) {
                graph_rewritten |= run_on_function(sub_graph);
            }
        }
        if (auto shape_of = std::dynamic_pointer_cast<ngraph::opset1::ShapeOf>(node)) {
            source_to_shape_of[shape_of->input_value(0)].push_back(shape_of);
        } else if (auto shape_of_3 = std::dynamic_pointer_cast<ngraph::opset3::ShapeOf>(node)) {
            source_to_shape_of_3[shape_of_3->input_value(0)].push_back(shape_of_3);
        }
    }

    for (const auto& pair : source_to_shape_of) {
        if (pair.second.size() < 2)
            continue;
        auto root_ss = pair.second[0];
        for (auto& child_ss : pair.second) {
            if (root_ss->get_instance_id() != child_ss->get_instance_id()) {
                graph_rewritten |= replace_output_update_name(child_ss->output(0), root_ss->output(0));
            }
        }
    }
    for (const auto& pair : source_to_shape_of_3) {
        if (pair.second.size() < 2)
            continue;
        auto root_ss = pair.second[0];
        for (auto& child_ss : pair.second) {
            if (root_ss->get_instance_id() != child_ss->get_instance_id() && root_ss->get_output_type() == child_ss->get_output_type()) {
                graph_rewritten |= replace_output_update_name(child_ss->output(0), root_ss->output(0));
            }
        }
    }
    return graph_rewritten;
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::GroupedGatherElimination, "GroupedGatherElimination", 0);

bool ngraph::pass::GroupedGatherElimination::run_on_function(std::shared_ptr<ngraph::Function> f) {
    RUN_ON_FUNCTION_SCOPE(GroupedGatherElimination);
    bool graph_rewritten = false;

    for (const auto & node : f->get_ordered_ops()) {
        // Recursively apply transformation for sub-graph based operations
        if (auto sub_graph_node = std::dynamic_pointer_cast<op::util::SubGraphOp>(node)) {
            if (auto sub_graph = sub_graph_node->get_function()) {
                graph_rewritten |= run_on_function(sub_graph);
            }
        }
        if (auto concat = std::dynamic_pointer_cast<ngraph::opset1::Concat>(node)) {
            if (concat->get_output_partial_shape(0).rank() != 1)
                continue;
            OutputVector inputs = concat->input_values();
            NodeVector new_ops;
            size_t i = 0, original_inputs_size = inputs.size();
            while (inputs.size() > i + 1) {
                auto curr = inputs[i].get_node_shared_ptr(), next = inputs[i + 1].get_node_shared_ptr();
                if (curr->get_type_info() != next->get_type_info() ||
                        (!is_type<opset1::Gather>(curr) && !is_type<opset7::Gather>(curr)) ||
                        (curr->input_value(0) != next->input_value(0))) {
                    ++i;
                    continue;
                } // curr and next are the same type of gather which takes data from the same source
                auto joint_indices = ngraph::op::util::make_try_fold<opset1::Concat>(OutputVector{curr->input_value(1), next->input_value(1)}, 0);
                auto new_gather = curr->clone_with_new_inputs(
                        {curr->input_value(0), joint_indices, ngraph::opset1::Constant::create(element::i64, {}, {0})});
                new_ops.push_back(joint_indices);
                new_ops.push_back(new_gather);
                inputs.erase(inputs.begin() + i);
                inputs[i] = new_gather->output(0);
            }
            if (original_inputs_size > inputs.size()) {
                auto new_concat = std::make_shared<opset1::Concat>(inputs, 0);
                new_ops.push_back(new_concat);
                new_concat->set_friendly_name(concat->get_friendly_name());
                ngraph::copy_runtime_info(concat, new_ops);
                ngraph::replace_node(concat, new_concat);
                graph_rewritten = true;
            }
        }
    }
    return graph_rewritten;
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::SimplifyShapeOfSubGraph, "SimplifyShapeOfSubGraph", 0);

bool ngraph::pass::SimplifyShapeOfSubGraph::run_on_function(std::shared_ptr<ngraph::Function> f) {
    RUN_ON_FUNCTION_SCOPE(GroupedGatherElimination);
    ngraph::pass::Manager manager;
    manager.set_per_pass_validation(false);
    manager.register_pass<ngraph::pass::EliminateGatherUnsqueeze>();
    manager.register_pass<ngraph::pass::SharedShapeOf>();
    manager.register_pass<ngraph::pass::GroupedGatherElimination>();
    manager.register_pass<ngraph::pass::Validate>();
    manager.run_passes(f);
    return false;
}
