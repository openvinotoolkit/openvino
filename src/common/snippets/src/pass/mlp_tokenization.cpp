// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/mlp_tokenization.hpp"

#include "snippets/itt.hpp"
#include "snippets/op/brgemm.hpp"
#include "snippets/op/subgraph.hpp"
#include "snippets/pass/collapse_subgraph.hpp"
#include "snippets/utils/utils.hpp"
#include "snippets/utils/tokenization_utils.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace snippets {
namespace pass {

using namespace ov::pass::pattern;

namespace {
ov::pass::pattern::op::Predicate act_predicate() {
    return ov::pass::pattern::op::Predicate(
        [](const Output<Node>& output) -> bool {
            return consumers_count(1)(output) && ov::snippets::pass::TokenizeSnippets::AppropriateForSubgraph(output.get_node_shared_ptr());
        },
        "act_predicate");
}

ov::pass::pattern::op::Predicate scalar_predicate() {
    return ov::pass::pattern::op::Predicate(
        [](const Output<Node>& output) -> bool {
            return ov::snippets::utils::is_scalar_constant(output.get_node_shared_ptr());
        },
        "scalar_predicate");
}

ov::pass::pattern::op::Predicate fc_predicate(bool is_down) {
    return ov::pass::pattern::op::Predicate(
        [=](const Output<Node>& output) -> bool {
            const auto node = output.get_node_shared_ptr();
            return ov::is_type<ov::op::v0::MatMul>(node) && ((!is_down && consumers_count(1)(output)) || is_down) &&
                   (op::Brgemm::get_output_type(node->get_input_element_type(0), node->get_input_element_type(1)) != element::dynamic);
        },
        "fc_predicate");
}

}  // namespace

TokenizeMLPSnippets::TokenizeMLPSnippets(const SnippetsTokenization::Config& config) {
    MATCHER_SCOPE(TokenizeMLPSnippets);

    auto make_weights = [](){
        // TODO: Add decompressed weights
        return wrap_type<ov::op::v0::Constant>();
    };

    auto m_input = any_input(ov::pass::pattern::has_static_rank());
    auto m_fc_gate = wrap_type<ov::opset1::MatMul>({m_input, make_weights()}, fc_predicate(false));
    auto m_fc_up = wrap_type<ov::opset1::MatMul>({m_input, make_weights()}, fc_predicate(false));

    auto m_act_unary = wrap_type<ov::op::util::UnaryElementwiseArithmetic, ov::op::v0::HardSigmoid>({m_fc_gate}, act_predicate());
    auto m_act_binary_scalar = wrap_type<ov::op::v0::Constant>(scalar_predicate());
    auto m_act_binary =
        wrap_type<ov::op::util::BinaryElementwiseArithmetic, ov::op::v4::Swish, ov::op::v0::PRelu>({m_fc_gate, m_act_binary_scalar}, act_predicate());
    auto m_act = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{m_act_unary, m_act_binary});

    auto m_mul = wrap_type<ov::opset1::Multiply>({m_act, m_fc_up}, consumers_count(1));
    auto m_fc_down = wrap_type<ov::opset1::MatMul>({m_mul, make_weights()}, fc_predicate(true));

    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(m_fc_down, matcher_name),
        [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher &m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::op::TokenizeMLPSnippets")
        auto& pattern_map = m.get_pattern_value_map();

        const auto fc_gate = pattern_map.at(m_fc_gate).get_node_shared_ptr();
        const auto fc_up = pattern_map.at(m_fc_up).get_node_shared_ptr();
        const auto act = pattern_map.at(m_act).get_node_shared_ptr();
        const auto mul = pattern_map.at(m_mul).get_node_shared_ptr();
        const auto fc_down = pattern_map.at(m_fc_down).get_node_shared_ptr();

        if (transformation_callback(fc_gate) || transformation_callback(fc_up) || transformation_callback(fc_down)) {
            return false;
        }

        static const auto body_params_count = 5; // 2xinput + 3x fc
        static const auto body_result_count = 1; // one output
        static const auto reg_group_count = 5; // upper-bound of possible buffer count

        // TODO [75567]: move this plugin-specific constraint to the plugin callback
        if (body_params_count + body_result_count + reg_group_count > config.get_data_ptr_gpr_count()) {
            return false;
        }

        const auto ordered_ops = ov::NodeVector{fc_gate, fc_up, act, mul, fc_down};
        const auto subgraph = ov::snippets::utils::tokenize_ordered_nodes(ordered_ops);

        // mark the Subgraph as Completed to not allow Snippets to include any nodes into this Subgraph in common Tokenization
        SetSnippetsSubgraphType(subgraph, SnippetsSubgraphType::Completed);
        return true;
    });
}

}  // namespace pass
}  // namespace snippets
}  // namespace ov
