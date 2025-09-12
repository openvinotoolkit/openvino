// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/gated_mlp_tokenization.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>

#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/hard_sigmoid.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/prelu.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/util/binary_elementwise_arithmetic.hpp"
#include "openvino/op/util/unary_elementwise_arithmetic.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/predicate.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/pp.hpp"
#include "snippets/itt.hpp"
#include "snippets/op/brgemm.hpp"
#include "snippets/pass/base_tokenization_config.hpp"
#include "snippets/pass/collapse_subgraph.hpp"
#include "snippets/pass/tokenization.hpp"
#include "snippets/utils/tokenization_utils.hpp"
#include "snippets/utils/utils.hpp"

namespace ov::snippets::pass {

using namespace ov::op;
using namespace ov::pass::pattern;
using namespace ov::pass::pattern::op;

namespace {
Predicate act_predicate() {
    return Predicate(
        [](const Output<Node>& output) -> bool {
            return consumers_count(1)(output) &&
                   ov::snippets::pass::TokenizeSnippets::AppropriateForSubgraph(output.get_node_shared_ptr());
        },
        "act_predicate");
}

Predicate scalar_predicate() {
    return Predicate(
        [](const Output<Node>& output) -> bool {
            return ov::snippets::utils::is_scalar_constant(output.get_node_shared_ptr());
        },
        "scalar_predicate");
}

Predicate fc_predicate(bool is_down) {
    return Predicate(
        [=](const Output<Node>& output) -> bool {
            const auto node = output.get_node_shared_ptr();
            return ov::is_type<v0::MatMul>(node) && ((!is_down && consumers_count(1)(output)) || is_down) &&
                   (op::Brgemm::get_output_type(node->get_input_element_type(0), node->get_input_element_type(1)) !=
                    element::dynamic);
        },
        "fc_predicate");
}

}  // namespace

TokenizeGatedMLPSnippets::TokenizeGatedMLPSnippets(const TokenizationConfig& config) {
    MATCHER_SCOPE(TokenizeGatedMLPSnippets);
    using namespace ov::pass;
    using namespace ov::op::util;

    auto make_weights = []() {
        // TODO: Add decompressed weights
        return wrap_type<v0::Constant>();
    };

    auto m_input = any_input(has_static_rank());
    auto m_fc_gate = wrap_type<v0::MatMul>({m_input, make_weights()}, fc_predicate(false));
    auto m_fc_up = wrap_type<v0::MatMul>({m_input, make_weights()}, fc_predicate(false));

    auto m_act_unary = wrap_type<UnaryElementwiseArithmetic, v0::HardSigmoid, v4::Swish>({m_fc_gate}, act_predicate());
    auto m_act_binary_scalar = wrap_type<v0::Constant>(scalar_predicate());
    auto m_act_binary =
        wrap_type<BinaryElementwiseArithmetic, v0::PRelu, v4::Swish>({m_fc_gate, m_act_binary_scalar}, act_predicate());
    auto m_act = m_act_unary | m_act_binary;

    auto m_mul = wrap_type<v1::Multiply>({m_act, m_fc_up}, consumers_count(1));
    auto m_fc_down = wrap_type<v0::MatMul>({m_mul, make_weights()}, fc_predicate(true));

    register_matcher(std::make_shared<Matcher>(m_fc_down, matcher_name), [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::op::TokenizeGatedMLPSnippets")
        auto& pattern_map = m.get_pattern_value_map();

        const auto fc_gate = pattern_map.at(m_fc_gate).get_node_shared_ptr();
        const auto fc_up = pattern_map.at(m_fc_up).get_node_shared_ptr();
        const auto act = pattern_map.at(m_act).get_node_shared_ptr();
        const auto mul = pattern_map.at(m_mul).get_node_shared_ptr();
        const auto fc_down = pattern_map.at(m_fc_down).get_node_shared_ptr();

        if (transformation_callback(fc_gate) || transformation_callback(fc_up) || transformation_callback(fc_down)) {
            return false;
        }

        const bool allow_shared_params = [&]() {
            const auto mm_gate = ov::as_type_ptr<ov::op::v0::MatMul>(fc_gate);
            const auto mm_up = ov::as_type_ptr<ov::op::v0::MatMul>(fc_up);
            OPENVINO_ASSERT(mm_gate && mm_up, "fc_gate and fc_up must have MatMul type");
            return mm_gate->get_transpose_a() == mm_up->get_transpose_a();
        }();
        // data input (can be shared or not) + 3x matmul weights + result
        const size_t io_count = (allow_shared_params ? 4 : 5) + 1;
        static constexpr size_t n_reg_group = 3;
        // Loop depth could reach 3 because of SplitLoops optimization
        static constexpr size_t n_loops_depth = 3;
        const auto ordered_ops = ov::NodeVector{fc_gate, fc_up, act, mul, fc_down};
        const bool is_dynamic = std::any_of(ordered_ops.begin(), ordered_ops.end(), [](const std::shared_ptr<Node>& n) {
            return n->is_dynamic();
        });

        // TODO [75567]: move this plugin-specific constraint to the plugin callback
        if (!config.is_gprs_count_sufficient(io_count, n_reg_group, n_loops_depth, is_dynamic)) {
            return false;
        }

        const auto subgraph = ov::snippets::utils::tokenize_ordered_nodes(ordered_ops, allow_shared_params);

        // mark the Subgraph as Completed to not allow Snippets to include any nodes into this Subgraph in common
        // Tokenization
        SetSnippetsSubgraphType(subgraph, SnippetsSubgraphType::Completed);
        return true;
    });
}

}  // namespace ov::snippets::pass
