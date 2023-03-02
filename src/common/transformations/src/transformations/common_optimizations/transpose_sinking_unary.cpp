// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/transpose_sinking_unary.hpp"

#include <transformations/utils/utils.hpp>
#include <utility>

#include "itt.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/common_optimizations/transpose_sinking_utils.hpp"
#include "transformations/rt_info/transpose_sinking_attr.hpp"

using namespace ov;
using namespace ov::opset10;
using namespace ov::pass::pattern;
using namespace ov::op::util;
using namespace transpose_sinking;

namespace {

using NodePtr = std::shared_ptr<ov::Node>;
using NodePair = std::pair<NodePtr, NodePtr>;

/**
 * @brief SwapNodes allows to perform swapping nodes even if there are more than one consumers but has less performance
 *
 * @param first_node first node pointer
 * @param second_node first node pointer
 * @return NodePair pair of nodes in new order that allows to register them in MatcherPass
 */
NodePair SwapNodes(const NodePtr& first_node, const NodePtr& second_node) {
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

}  // namespace

ov::pass::TransposeSinkingUnaryForward::TransposeSinkingUnaryForward() {
    MATCHER_SCOPE(TransposeSinkingUnaryForward);

    auto transpose_label = wrap_type<Transpose>({any_input(), any_input()});
    auto unary_label = wrap_type<UnaryElementwiseArithmetic, Clamp, Elu, SoftPlus, LogicalNot, Convert, IsInf, IsNaN, IsFinite>({transpose_label});

    ov::matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto transpose = pattern_to_output.at(transpose_label).get_node_shared_ptr();
        auto unary = pattern_to_output.at(unary_label).get_node_shared_ptr();

        const NodePair new_nodes = SwapNodes(transpose, unary);

        register_new_node(new_nodes.first);
        register_new_node(new_nodes.second);

        UpdateForwardSinkingAbility(new_nodes.second);
        return true;
    };

    auto m = std::make_shared<Matcher>(unary_label, "ov::pass::TransposeSinkingUnaryForward");
    register_matcher(m, matcher_pass_callback);
}

namespace {
bool IfSinkingEnabled(const Output<Node>& output) {
    return is_sinking_node(output.get_node_shared_ptr());
}
}  // namespace

ov::pass::TransposeSinkingUnaryBackward::TransposeSinkingUnaryBackward() {
    MATCHER_SCOPE(TransposeSinkingUnaryBackwardMultiConsumers);

    auto unary_restrictions = [](const Output<Node>& output) -> bool {
        return HasSameOutputTransposeNodes(output);
    };

    auto unary_label =
        wrap_type<UnaryElementwiseArithmetic, Clamp, Elu, SoftPlus, LogicalNot, Convert, IsInf, IsNaN, IsFinite>(
                {any_input()}, unary_restrictions);

    auto transpose_const_label = wrap_type<Constant>();

    auto transpose_label = wrap_type<Transpose>({unary_label, transpose_const_label}, IfSinkingEnabled);

    ov::matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto transpose_const = as_type_ptr<Constant>(pattern_to_output.at(transpose_const_label).get_node_shared_ptr());
        auto transpose = pattern_to_output.at(transpose_label).get_node_shared_ptr();
        auto unary = pattern_to_output.at(unary_label).get_node_shared_ptr();

        for (auto& new_node : sink_backward::InsertTransposeBeforeNode(unary, transpose_const)) {
            register_new_node(new_node);
        }
        unary->validate_and_infer_types();
        // remove output transposes
        RemoveSingleOutputConsumers(unary);
        return true;
    };

    auto m = std::make_shared<Matcher>(transpose_label, "ov::pass::TransposeSinkingUnaryBackward");
    register_matcher(m, matcher_pass_callback);
}
