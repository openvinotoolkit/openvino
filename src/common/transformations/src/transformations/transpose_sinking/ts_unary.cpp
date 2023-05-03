// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/transpose_sinking/ts_unary.hpp"

#include <utility>

#include "itt.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/elu.hpp"
#include "openvino/op/is_finite.hpp"
#include "openvino/op/is_inf.hpp"
#include "openvino/op/logical_not.hpp"
#include "openvino/op/softplus.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/transpose_sinking_attr.hpp"
#include "transformations/transpose_sinking/ts_utils.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;
using namespace ov::pass::pattern;
using namespace ov::op::util;
using namespace ov::pass::transpose_sinking;
using namespace ov::pass::transpose_sinking::utils;

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

TSUnaryForward::TSUnaryForward() {
    MATCHER_SCOPE(TSUnaryForward);

    auto transpose_label = wrap_type<ov::op::v1::Transpose>({any_input(), any_input()});
    auto unary_label = wrap_type<UnaryElementwiseArithmetic,
                                 ov::op::v0::Clamp,
                                 ov::op::v0::Elu,
                                 ov::op::v4::SoftPlus,
                                 ov::op::v1::LogicalNot,
                                 ov::op::v0::Convert,
                                 ov::op::v10::IsInf,
                                 ov::op::v10::IsNaN,
                                 ov::op::v10::IsFinite>({transpose_label});

    ov::matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto transpose = pattern_to_output.at(transpose_label).get_node_shared_ptr();
        auto unary = pattern_to_output.at(unary_label).get_node_shared_ptr();
        if (transformation_callback(unary)) {
            return false;
        }

        const NodePair new_nodes = SwapNodes(transpose, unary);

        register_new_node(new_nodes.first);
        register_new_node(new_nodes.second);

        UpdateForwardSinkingAbility(new_nodes.second);
        return true;
    };

    auto m = std::make_shared<Matcher>(unary_label, "ov::pass::TSUnaryForward");
    register_matcher(m, matcher_pass_callback);
}

TSUnaryBackward::TSUnaryBackward() {
    MATCHER_SCOPE(TSUnaryBackwardMultiConsumers);

    auto unary_restrictions = [](const Output<Node>& output) -> bool {
        return CheckTransposeConsumers(output);
    };

    auto unary_label = wrap_type<UnaryElementwiseArithmetic,
                                 ov::op::v0::Clamp,
                                 ov::op::v0::Elu,
                                 ov::op::v4::SoftPlus,
                                 ov::op::v1::LogicalNot,
                                 ov::op::v0::Convert,
                                 ov::op::v10::IsInf,
                                 ov::op::v10::IsNaN,
                                 ov::op::v10::IsFinite>({any_input()}, unary_restrictions);

    auto transpose_const_label = wrap_type<ov::op::v0::Constant>();

    auto transpose_label = wrap_type<ov::op::v1::Transpose>({unary_label, transpose_const_label});

    ov::matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto transpose_const =
            as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(transpose_const_label).get_node_shared_ptr());
        auto transpose = pattern_to_output.at(transpose_label).get_node_shared_ptr();
        auto unary = pattern_to_output.at(unary_label).get_node_shared_ptr();
        if (transformation_callback(unary)) {
            return false;
        }

        for (auto& new_node : sink_backward::InsertTransposeBeforeNode(unary, transpose_const)) {
            register_new_node(new_node);
        }
        unary->validate_and_infer_types();
        RemoveTransposeConsumers(unary);
        return true;
    };

    auto m = std::make_shared<Matcher>(transpose_label, "ov::pass::TSUnaryBackward");
    register_matcher(m, matcher_pass_callback);
}
