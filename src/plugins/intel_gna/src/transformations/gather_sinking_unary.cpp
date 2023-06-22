// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/gather_sinking_unary.hpp"

#include <transformations/utils/utils.hpp>
#include <utility>

#include "openvino/cc/ngraph/itt.hpp"
#include "openvino/opsets/opset12.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/gather_sinking_attr.hpp"
#include "transformations/utils/gather_sinking_utils.hpp"

using namespace ov;
using namespace ov::opset12;
using namespace ov::pass::pattern;
using namespace ov::op::util;
using namespace gather_sinking;
using namespace ov::intel_gna::pass;
using namespace ov::intel_gna::rt_info;

namespace {

using NodePtr = std::shared_ptr<ov::Node>;
using NodePair = std::pair<NodePtr, NodePtr>;

/**
 * @brief swap_nodes allows to perform swapping nodes even if there are more than one consumers but has less performance
 *
 * @param first_node first node pointer
 * @param second_node first node pointer
 * @return NodePair pair of nodes in new order that allows to register them in MatcherPass
 */
NodePair swap_nodes(NodePtr first_node, NodePtr second_node) {
    auto second_node_inputs = second_node->input_values();
    second_node_inputs[0] = first_node->input_value(0);

    auto new_first_node = second_node->clone_with_new_inputs(second_node_inputs);

    auto first_node_inputs = first_node->input_values();
    first_node_inputs[0] = new_first_node;
    auto new_second_node = first_node->clone_with_new_inputs(first_node_inputs);

    new_second_node->set_friendly_name(second_node->get_friendly_name());
    ov::copy_runtime_info({first_node, second_node}, {new_first_node, new_second_node});

    ov::replace_node(second_node, new_second_node);

    return std::make_pair(new_first_node, new_second_node);
}

/**
 * @brief swap_outputs has much better performance than swap_nodes and covers the most of the real situations
 *        but cannot work when the consumers count greater than one
 * @param first_node first node pointer
 * @param second_node second node pointer
 * @return NodePair pair of nodes in new order that allows to register them in MatcherPass
 */
NodePair swap_outputs(NodePtr first_node, NodePtr second_node) {
    const auto first_node_output_names = first_node->output(0).get_names();
    const auto second_node_output_names = second_node->output(0).get_names();

    auto swap_names = [&]() {
        const std::string first_name = first_node->get_friendly_name();
        first_node->set_friendly_name(second_node->get_friendly_name());
        second_node->set_friendly_name(first_name);

        first_node->output(0).set_names(second_node_output_names);
        second_node->output(0).set_names(first_node_output_names);
    };

    auto out_1 = first_node->input_value(0);
    second_node->input(0).replace_source_output(out_1);

    auto out_2 = second_node->output(0);
    second_node->output(0).replace(first_node->output(0));

    first_node->input(0).replace_source_output(out_2);

    swap_names();

    return std::make_pair(second_node, first_node);
}

/**
 * swapping inputs/outputs has better perfomance that swapping nodes with clone but it cannot be used
 * in multiple consumers case
 */
NodePair swap_items(NodePtr first_node, NodePtr second_node) {
    NodePair new_nodes;

    if (first_node->output(0).get_target_inputs().size() > 1 || second_node->output(0).get_target_inputs().size() > 1)
        new_nodes = swap_nodes(first_node, second_node);
    else
        new_nodes = swap_outputs(first_node, second_node);

    return new_nodes;
}

}  // namespace

GatherSinkingUnaryForward::GatherSinkingUnaryForward() {
    MATCHER_SCOPE(GatherSinkingUnaryForward);
    auto gather_label = wrap_type<Gather>({any_input(), any_input(), any_input()});
    auto unary_label = wrap_type<UnaryElementwiseArithmetic, Clamp, Elu, SoftPlus, LogicalNot, Convert>({gather_label});

    ov::matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto gather = pattern_to_output.at(gather_label).get_node_shared_ptr();
        auto unary = pattern_to_output.at(unary_label).get_node_shared_ptr();

        const NodePair new_nodes = swap_items(gather, unary);

        register_new_node(new_nodes.first);
        register_new_node(new_nodes.second);

        update_forward_gather_sinking_ability(new_nodes.second);

        return true;
    };

    auto m = std::make_shared<Matcher>(unary_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

GatherSinkingUnaryBackwardSingleConsumer::GatherSinkingUnaryBackwardSingleConsumer() {
    MATCHER_SCOPE(GatherSinkingUnaryBackwardSingleConsumer);
    auto unary_label =
        wrap_type<UnaryElementwiseArithmetic, Clamp, Elu, SoftPlus, LogicalNot, Convert>({any_input()},
                                                                                         consumers_count(1));
    // FakeQuantize pattern
    auto fq_label =
        wrap_type<FakeQuantize>({any_input(), any_input(), any_input(), any_input(), any_input()}, consumers_count(1));

    auto pattern_node = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{unary_label, fq_label});

    auto gather_label = wrap_type<Gather>({pattern_node, any_input(), any_input()}, is_gather_sinking_enabled);

    ov::matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto gather = pattern_to_output.at(gather_label).get_node_shared_ptr();
        auto unary = gather->get_input_node_shared_ptr(0);

        const NodePair new_nodes = swap_items(unary, gather);

        register_new_node(new_nodes.first);
        register_new_node(new_nodes.second);

        return true;
    };

    auto m = std::make_shared<Matcher>(gather_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

GatherSinkingUnaryBackwardMultiConsumers::GatherSinkingUnaryBackwardMultiConsumers() {
    MATCHER_SCOPE(GatherSinkingUnaryBackwardMultiConsumers);
    auto unary_restrictions = [](const Output<Node>& output) -> bool {
        return ov::pass::pattern::consumers_more_than(1)(output) && has_same_output_gather_nodes(output);
    };

    auto unary_label =
        wrap_type<UnaryElementwiseArithmetic, Clamp, Elu, SoftPlus, LogicalNot, Convert>({any_input()},
                                                                                         unary_restrictions);
    // FakeQuantize pattern
    auto fq_label =
        wrap_type<FakeQuantize>({any_input(), any_input(), any_input(), any_input(), any_input()}, unary_restrictions);

    auto pattern_node = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{unary_label, fq_label});

    auto indices_const_label = wrap_type<Constant>();
    auto axes_const_label = wrap_type<Constant>();
    auto gather_label =
        wrap_type<Gather>({pattern_node, indices_const_label, axes_const_label}, is_gather_sinking_enabled);

    ov::matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto indices_const = as_type_ptr<Constant>(pattern_to_output.at(indices_const_label).get_node_shared_ptr());
        auto axes_const = as_type_ptr<Constant>(pattern_to_output.at(axes_const_label).get_node_shared_ptr());
        auto gather = as_type_ptr<Gather>(pattern_to_output.at(gather_label).get_node_shared_ptr());
        auto unary = gather->get_input_node_shared_ptr(0);

        for (auto& new_node : sink_backward::insert_gather_before_node(unary, indices_const, axes_const, gather)) {
            register_new_node(new_node);
        }

        // remove output transposes
        remove_single_output_consumers(unary);

        return true;
    };

    auto m = std::make_shared<Matcher>(gather_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

GatherSinkingUnaryBackward::GatherSinkingUnaryBackward() {
    MATCHER_SCOPE(GatherSinkingUnaryBackward);
    add_matcher<GatherSinkingUnaryBackwardSingleConsumer>();
    add_matcher<GatherSinkingUnaryBackwardMultiConsumers>();
}
