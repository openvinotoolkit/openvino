// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/mlp_seq_tokenization.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>

#include "openvino/core/descriptor/tensor.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/util/binary_elementwise_arithmetic.hpp"
#include "openvino/op/util/unary_elementwise_arithmetic.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/pp.hpp"
#include "snippets/itt.hpp"
#include "snippets/op/brgemm.hpp"
#include "snippets/pass/collapse_subgraph.hpp"
#include "snippets/pass/tokenization.hpp"
#include "snippets/utils/tokenization_utils.hpp"
#include "snippets/utils/utils.hpp"

namespace ov::snippets::pass {

using namespace ov::snippets::utils;

namespace {
inline bool has_one_consumer(const std::shared_ptr<ov::Node>& node) {
    return node->get_output_target_inputs(0).size() == 1;
}
}  // namespace

const size_t TokenizeMLPSeqSnippets::m_rank = 2;

bool TokenizeMLPSeqSnippets::is_tensor_supported(const ov::descriptor::Tensor& t) {
    return t.get_partial_shape().rank().is_static() && t.get_partial_shape().size() <= m_rank;
}

bool TokenizeMLPSeqSnippets::is_matmul_supported(const std::shared_ptr<ov::Node>& node) {
    const auto matmul = ov::as_type_ptr<ov::opset1::MatMul>(node);
    if (!matmul || matmul->get_transpose_a() ||
        !ov::is_type<ov::op::v0::Constant>(matmul->input_value(1).get_node_shared_ptr()) ||
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
    return is_supported_softmax(node) || ov::is_type_any_of<ov::op::util::UnaryElementwiseArithmetic,
                                                            ov::op::util::BinaryElementwiseArithmetic,
                                                            ov::op::v0::FakeQuantize>(node);
}

TokenizeMLPSeqSnippets::TokenizeMLPSeqSnippets(const SnippetsTokenization::Config& config) {
    MATCHER_SCOPE(TokenizeMLPSeqSnippets);

    auto constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto m_matmul0 = ov::pass::pattern::wrap_type<ov::opset1::MatMul>({ov::pass::pattern::any_input(), constant});

    register_matcher(
        std::make_shared<ov::pass::pattern::Matcher>(m_matmul0, matcher_name),
        [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
            OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::op::TokenizeMLPSeqSnippets")
            auto& pattern_to_output = m.get_pattern_value_map();

            // Two inputs of first MatMul + Result
            size_t io_count = 3;
            ov::NodeVector ordered_ops;

            /* ======== Matcher Pass ========== */

            /****** Skeleton ******/
            /* Skeleton on MLPSeq sequence-pattern is:
             *            MatMul0
             *               |
             *    Eltwise/FakeQuantize (1 or more)
             *               |
             *            MatMul1
             *               |
             *    Eltwise/FakeQuantize (1 or more)
             *               |
             *            MatMul2
             *               |
             *              ...
             */

            const auto matmul0 =
                ov::as_type_ptr<ov::opset1::MatMul>(pattern_to_output.at(m_matmul0).get_node_shared_ptr());
            if (!is_matmul_supported(matmul0) || !has_one_consumer(matmul0)) {
                return false;
            }

            if (transformation_callback(matmul0)) {
                return false;
            }
            const auto& out_shape = matmul0->get_output_partial_shape(0);
            OPENVINO_ASSERT(out_shape.rank().is_static() && out_shape.size() >= 2,
                            "MatMul out shape must be 2D at least");
            const auto& m_dim = *(out_shape.rbegin() + 1);

            bool is_dynamic = matmul0->is_dynamic();
            // Add possible FQ before matmul0
            if (auto fq = ov::as_type_ptr<ov::op::v0::FakeQuantize>(matmul0->get_input_node_shared_ptr(0))) {
                if (has_one_consumer(fq)) {
                    is_dynamic = is_dynamic || fq->is_dynamic();
                    io_count += get_non_scalar_constant_count_for_fq(fq);
                    ordered_ops.push_back(fq);
                }
            }
            ordered_ops.push_back(matmul0);

            // Tokenize Sequence while we can do it
            std::shared_ptr<ov::Node> prev_op = matmul0;
            auto interm_op = prev_op->get_output_target_inputs(0).begin()->get_node()->shared_from_this();
            bool postops_fusion_possible = true;
            std::shared_ptr<ov::op::v0::MatMul> cur_matmul = matmul0;
            while (has_one_consumer(prev_op)) {
                auto current_io_count = io_count;

                if (is_matmul_supported(interm_op) && !transformation_callback(interm_op)) {
                    // +1 for weights
                    current_io_count++;
                    // If MatMul is the first in the sequence, postops fusion status is reset
                    postops_fusion_possible = true;
                    cur_matmul = ov::as_type_ptr<ov::op::v0::MatMul>(interm_op);
                    OPENVINO_ASSERT(cur_matmul, "MatMul is expected");
                } else if (is_supported_intermediate_op(interm_op)) {
                    // Intermediate op contributes to the body params count only if can't be fused as post-op
                    // or if a previous node between MatMul and this op is not supported by post-op fusion
                    if (!postops_fusion_possible || config.get_can_be_fused_as_postop() == nullptr ||
                        !config.get_can_be_fused_as_postop()(cur_matmul, interm_op)) {
                        postops_fusion_possible = false;
                        current_io_count += get_potential_body_params(interm_op);
                    }
                } else {
                    // Unsupported op
                    break;
                }

                // TODO [75567]: move this plugin-specific constraint to the plugin callback
                // Heuiristic count of possible buffers - upper-bound value
                static constexpr size_t n_buffer_reg_groups = 2;
                const bool small_m = m_dim.is_static() && (m_dim.get_length() <= 32);
                // Loop depth could reach 3 because of SplitLoops optimization.
                // If M is small enough, SplitLoops is not needed
                const size_t loops_depth = small_m ? 2 : 3;
                is_dynamic = is_dynamic || interm_op->is_dynamic();
                if (!config.is_gprs_count_sufficient(current_io_count, n_buffer_reg_groups, loops_depth, is_dynamic)) {
                    break;
                }

                ordered_ops.push_back(interm_op);
                prev_op = interm_op;
                interm_op = prev_op->get_output_target_inputs(0).begin()->get_node()->shared_from_this();

                // Move counts
                io_count = current_io_count;
            }

            // Currently, sequence of MLP should contain 2 MatMuls at least
            size_t mm_count = 0;
            for (const auto& op : ordered_ops) {
                mm_count += static_cast<size_t>(ov::is_type<ov::op::v0::MatMul>(op));
            }
            if (mm_count < 2) {
                return false;
            }

            const auto subgraph = tokenize_ordered_nodes(ordered_ops);
            // mark the Subgraph as Completed to not allow Snippets to include any nodes into this Subgraph in common
            // Tokenization
            SetSnippetsSubgraphType(subgraph, SnippetsSubgraphType::Completed);
            return true;
        });
}

}  // namespace ov::snippets::pass
