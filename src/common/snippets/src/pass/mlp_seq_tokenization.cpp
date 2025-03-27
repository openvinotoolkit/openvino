// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/mlp_seq_tokenization.hpp"

#include "snippets/itt.hpp"
#include "snippets/op/brgemm.hpp"
#include "snippets/op/subgraph.hpp"
#include "snippets/pass/collapse_subgraph.hpp"
#include "snippets/utils/utils.hpp"
#include "snippets/utils/tokenization_utils.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace snippets {
namespace pass {

namespace {

inline bool has_one_consumer(const std::shared_ptr<ov::Node>& node) {
    return node->get_output_target_inputs(0).size() == 1;
}

size_t get_potential_body_params(const std::shared_ptr<ov::Node>& op) {
    size_t count = 0;
    for (size_t i = 1; i < op->get_input_size(); ++i) {
        const auto input = op->input_value(i);
        const auto parent = input.get_node_shared_ptr();
        const auto constant = ov::as_type_ptr<ov::op::v0::Constant>(parent);
        const auto is_scalar = constant && (ov::shape_size(input.get_shape()) == 1);
        const auto should_be_inside_body = constant && ov::snippets::op::Subgraph::constant_input_should_be_inside_body(op);
        const auto is_fq_weight = constant && ov::is_type<ov::op::v0::FakeQuantize>(op);
        if (!(is_scalar || should_be_inside_body || is_fq_weight)) {
            count++;
        }
    }
    return count;
}

} // namespace

const size_t TokenizeMLPSeqSnippets::m_rank = 2;

bool TokenizeMLPSeqSnippets::is_tensor_supported(const ov::descriptor::Tensor& t) {
    return t.get_partial_shape().rank().is_static() && t.get_partial_shape().size() <= m_rank;
}

bool TokenizeMLPSeqSnippets::is_matmul_supported(const std::shared_ptr<ov::Node>& node) {
    const auto matmul = ov::as_type_ptr<ov::opset1::MatMul>(node);
    if (!matmul || matmul->get_transpose_a() || !ov::is_type<ov::op::v0::Constant>(matmul->get_input_node_shared_ptr(1)) ||
        !is_tensor_supported(matmul->get_input_tensor(0)) || !is_tensor_supported(matmul->get_input_tensor(1))) {
        return false;
    }

    const auto prc = op::Brgemm::get_output_type(matmul->get_input_element_type(0), matmul->get_input_element_type(1));
    return prc != element::dynamic;
}

bool TokenizeMLPSeqSnippets::is_supported_softmax(const std::shared_ptr<ov::Node>& node) {
    int64_t axis = 0;
    const auto rank = node->get_input_partial_shape(0).rank();
    if (const auto softmax_v8 = ov::as_type_ptr<ov::op::v8::Softmax>(node)) {
        axis = ov::util::try_normalize_axis(softmax_v8->get_axis(), rank, *node);
    } else if (const auto softmax_v1 = ov::as_type_ptr<ov::op::v1::Softmax>(node)) {
        axis = softmax_v1->get_axis();
    } else {
        return false;
    }
    return is_tensor_supported(node->get_input_tensor(0)) && axis == (rank.get_length() - 1);
}

bool TokenizeMLPSeqSnippets::is_supported_intermediate_op(const std::shared_ptr<ov::Node>& node) {
    if (!ov::snippets::pass::TokenizeSnippets::AppropriateForSubgraph(node)) {
        return false;
    }
    return is_supported_softmax(node) ||
           ov::is_type_any_of<ov::op::util::UnaryElementwiseArithmetic,
                              ov::op::util::BinaryElementwiseArithmetic,
                              ov::op::v0::FakeQuantize>(node);
}

TokenizeMLPSeqSnippets::TokenizeMLPSeqSnippets(const SnippetsTokenization::Config& config) {
    MATCHER_SCOPE(TokenizeMLPSeqSnippets);

    auto constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto m_matmul0 = ov::pass::pattern::wrap_type<ov::opset1::MatMul>({ov::pass::pattern::any_input(), constant});

    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(m_matmul0, matcher_name),
        [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher &m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::op::TokenizeMLPSeqSnippets")
        auto& pattern_to_output = m.get_pattern_value_map();

        // Two inputs of first MatMul
        size_t potential_body_params_count = 2;
        // After some transformations, a different number of Constants for some operations may be created
        // than the actual number of Constants during tokenization.
        // To avoid unsupported number of non-scalar Constants in the future (plugin specific limitation)
        // we should calculate potential number of non-scalar Constants that will be moved up from body.
        size_t hidden_virtual_ports_count = 0;
        // Heuiristic count of possible buffers - upper-bound value
        const size_t uniqie_buffer_reg_group_count = 2;
        std::string fused_names;
        ov::NodeVector ordered_ops;

        /* ======== Matcher Pass ========== */

        /****** Skeleton ******/
        /* Skeleton on MLP sequence-pattern is (There are hsould be 2 min FC):
         */

        const auto matmul0 = ov::as_type_ptr<ov::opset1::MatMul>(pattern_to_output.at(m_matmul0).get_node_shared_ptr());
        if (!is_matmul_supported(matmul0) || !has_one_consumer(matmul0)) {
            return false;
        }

        if (transformation_callback(matmul0)) {
            return false;
        }

        // Add possible FQ before matmul0
        if (const auto fq = ov::as_type_ptr<ov::op::v0::FakeQuantize>(matmul0->get_input_node_shared_ptr(0))) {
            if (has_one_consumer(fq)) {
                hidden_virtual_ports_count += ov::snippets::utils::get_non_scalar_constant_count_for_fq(fq);
                ordered_ops.push_back(fq);
            }
        }
        ordered_ops.push_back(matmul0);

        // Tokenize Sequence while we can do it
        std::shared_ptr<ov::Node> prev_op = matmul0;
        auto interm_op = prev_op->get_output_target_inputs(0).begin()->get_node()->shared_from_this();
        auto available_regs = config.get_data_ptr_gpr_count();

        while (has_one_consumer(prev_op)) {
            auto possible_param_count = potential_body_params_count;
            auto possible_hidden_virtual_ports_count = hidden_virtual_ports_count;

            if (is_matmul_supported(interm_op) && !transformation_callback(interm_op)) {
                // +1 for weights
                possible_param_count++;
            } else if (is_supported_intermediate_op(interm_op)) {
                possible_param_count += get_potential_body_params(interm_op);
                if (const auto fq = ov::as_type_ptr<ov::op::v0::FakeQuantize>(matmul0->get_input_node_shared_ptr(0))) {
                    possible_hidden_virtual_ports_count += ov::snippets::utils::get_non_scalar_constant_count_for_fq(fq);
                }
            } else {
                // Unsupported op
                break;
            }

            // TODO [75567]: move this plugin-specific constraint to the plugin callback
            // +1 - for result
            if (possible_param_count + possible_hidden_virtual_ports_count + uniqie_buffer_reg_group_count + 1 > available_regs)
                break;

            ordered_ops.push_back(interm_op);
            prev_op = interm_op;
            interm_op = prev_op->get_output_target_inputs(0).begin()->get_node()->shared_from_this();

            // Move counts
            potential_body_params_count = possible_param_count;
            hidden_virtual_ports_count = possible_hidden_virtual_ports_count;
        }

        // Currently, sequence of MLP should contain 2 MatMuls at least
        size_t mm_count = 0;
        for (const auto& op : ordered_ops) {
            mm_count += (ov::is_type<ov::op::v0::MatMul>(op));
        }
        if (mm_count < 2)
            return false;

        /* ================================ */

        /* ====== Subgraph creation ======= */

        const auto subgraph = ov::snippets::utils::tokenize_ordered_nodes(ordered_ops);

        // mark the Subgraph as Completed to not allow Snippets to include any nodes into this Subgraph in common Tokenization
        SetSnippetsSubgraphType(subgraph, SnippetsSubgraphType::Completed);

        return true;

        /* ================================ */
    });
}

}  // namespace pass
}  // namespace snippets
}  // namespace ov
