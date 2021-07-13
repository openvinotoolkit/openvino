// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <vector>

#include "itt.hpp"
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <transformations/common_optimizations/simplify_shape_of_sub_graph.hpp>
#include <transformations/common_optimizations/eliminate_unsqueeze_gather.hpp>
#include <transformations/utils/utils.hpp>
#include <numeric>

NGRAPH_RTTI_DEFINITION(ngraph::pass::SharedShapeOf, "SharedShapeOf", 0);

bool ngraph::pass::SharedShapeOf::run_on_function(std::shared_ptr<ngraph::Function> f) {
    RUN_ON_FUNCTION_SCOPE(SharedShapeOf);
    bool graph_rewritten = false;

    std::map<ngraph::Output<Node>, std::vector<std::shared_ptr<ngraph::Node>>> source_to_shape_of;
    for (const auto & node : f->get_ordered_ops()) {
        // Recursively apply transformation for sub-graph based operations
        if (auto sub_graph_node = std::dynamic_pointer_cast<op::util::SubGraphOp>(node))
            if (auto sub_graph = sub_graph_node->get_function())
                graph_rewritten |= run_on_function(sub_graph);

        if (is_type<ngraph::opset1::ShapeOf>(node) || is_type<ngraph::opset3::ShapeOf>(node))
            source_to_shape_of[node->input_value(0)].push_back(node);
    }

    for (const auto& pair : source_to_shape_of) {
        if (pair.second.size() < 2)
            continue;
        const auto& root_ss = pair.second[0];
        for (const auto& child_ss : pair.second)
            if (root_ss->get_instance_id() != child_ss->get_instance_id() && root_ss->get_output_element_type(0) == root_ss->get_output_element_type(0))
                graph_rewritten |= replace_output_update_name(child_ss->output(0), root_ss->output(0));
    }
    return graph_rewritten;
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::GroupedGatherElimination, "GroupedGatherElimination", 0);

ngraph::pass::GroupedGatherElimination::GroupedGatherElimination() {
    MATCHER_SCOPE(GroupedGatherElimination);
    auto concat_label = ngraph::pattern::wrap_type<ngraph::opset1::Concat>(pattern::rank_equals(1));

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto concat = m.get_match_root();
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
            bool is_opset1 = is_type<opset1::Gather>(curr);
            auto joint_indices = ngraph::op::util::make_try_fold<opset1::Concat>(OutputVector{curr->input_value(1), next->input_value(1)}, 0);
            std::shared_ptr<Node> new_gather;
            if (is_opset1)
                new_gather = register_new_node<ngraph::opset1::Gather>(
                    curr->input_value(0), joint_indices->output(0), ngraph::opset1::Constant::create(element::i64, {}, {0})->output(0));
            else
                new_gather = register_new_node<ngraph::opset7::Gather>(
                        curr->input_value(0), joint_indices->output(0), ngraph::opset1::Constant::create(element::i64, {}, {0})->output(0));
            new_ops.push_back(joint_indices);
            new_ops.push_back(new_gather);
            inputs.erase(inputs.begin() + i);
            inputs[i] = new_gather->output(0);
        }
        ngraph::copy_runtime_info(concat, new_ops);
        if (inputs.size() == 1) // we can optimize out concat
            return replace_output_update_name(concat->output(0), inputs[0]);
        if (original_inputs_size > inputs.size()) {
            auto new_concat = std::make_shared<opset1::Concat>(inputs, 0);
            new_concat->set_friendly_name(concat->get_friendly_name());
            ngraph::copy_runtime_info(concat, new_concat);
            ngraph::replace_node(concat, new_concat);
            return true;
        }
        return false;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(concat_label, matcher_name);
    this->register_matcher(m, callback);
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::GatherNopElimination, "GatherNopElimination", 0);

ngraph::pass::GatherNopElimination::GatherNopElimination() {
    MATCHER_SCOPE(GatherNopElimination);
    const auto gather_label = ngraph::pattern::wrap_type<ngraph::op::util::GatherBase>(
            {ngraph::pattern::any_input(pattern::has_static_shape()),
             ngraph::pattern::wrap_type<ngraph::op::Constant>(),
             ngraph::pattern::wrap_type<ngraph::op::Constant>()});

    ngraph::matcher_pass_callback callback = [](pattern::Matcher& m) {
        auto gather = m.get_match_root();
        const auto& number_of_indices = shape_size(gather->get_input_shape(1));
        if (gather->get_input_shape(0) != gather->get_output_shape(0) || shape_size(gather->get_input_shape(2)) != 1 || number_of_indices > 10)
            return false;
        std::vector<int64_t> expected_vector(number_of_indices);
        std::iota(expected_vector.begin(), expected_vector.end(), 0);
        if (const auto& indices = get_constant_from_source(gather->input_value(1))) {
            const auto& indices_values = indices->cast_vector<int64_t>();
            if (indices_values != expected_vector)
                return false;
        }
        return replace_output_update_name(gather->output(0), gather->input_value(0));
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(gather_label, matcher_name);
    this->register_matcher(m, callback);
}


NGRAPH_RTTI_DEFINITION(ngraph::pass::SimplifyShapeOfSubGraph, "SimplifyShapeOfSubGraph", 0);

bool ngraph::pass::SimplifyShapeOfSubGraph::run_on_function(std::shared_ptr<ngraph::Function> f) {
    RUN_ON_FUNCTION_SCOPE(GroupedGatherElimination);
    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::EliminateGatherUnsqueeze>();
    manager.register_pass<ngraph::pass::SharedShapeOf>();
    manager.register_pass<ngraph::pass::GroupedGatherElimination>();
    manager.register_pass<ngraph::pass::GatherNopElimination>();
    manager.run_passes(f);
    return false;
}
