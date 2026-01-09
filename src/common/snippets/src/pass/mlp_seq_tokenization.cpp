// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/mlp_seq_tokenization.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "openvino/core/descriptor/tensor.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/matmul.hpp"
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
    const auto axis = ov::snippets::utils::get_softmax_axis(node);
    if (!axis) {
        return false;
    }
    const auto rank = static_cast<int64_t>(node->get_input_partial_shape(0).rank().get_length());
    return is_tensor_supported(node->get_input_tensor(0)) && *axis == (rank - 1);
}

bool TokenizeMLPSeqSnippets::is_supported_intermediate_op(const std::shared_ptr<ov::Node>& node) {
    if (!ov::snippets::pass::TokenizeSnippets::AppropriateForSubgraph(node)) {
        return false;
    }
    return is_supported_softmax(node) || ov::is_type_any_of<ov::op::util::UnaryElementwiseArithmetic,
                                                            ov::op::util::BinaryElementwiseArithmetic,
                                                            ov::op::v0::FakeQuantize,
                                                            ov::op::v0::Convert>(node);
}

TokenizeMLPSeqSnippets::TokenizeMLPSeqSnippets(const Config& config) {
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
            // Non fused postops between MatMuls increase number of GPRs for Snippets kernel execution
            bool non_fused_postops_between_matmuls = false;
            auto cur_matmul = matmul0;
            while (has_one_consumer(prev_op)) {
                auto current_io_count = io_count;

                if (is_matmul_supported(interm_op) && !transformation_callback(interm_op)) {
                    if (!postops_fusion_possible) {
                        non_fused_postops_between_matmuls = true;
                    }
                    // +1 for weights
                    current_io_count++;
                    // If MatMul is the first in the sequence, postops fusion status is reset
                    postops_fusion_possible = true;
                    cur_matmul = ov::as_type_ptr<ov::op::v0::MatMul>(interm_op);
                    OPENVINO_ASSERT(cur_matmul, "MatMul is expected");
                } else if (is_supported_intermediate_op(interm_op)) {
                    // Intermediate op contributes to the body params count only if can't be fused as post-op
                    // or if a previous node between MatMul and this op is not supported by post-op fusion
                    if (!postops_fusion_possible || !config.get_can_be_fused_as_postop()(cur_matmul, interm_op)) {
                        postops_fusion_possible = false;
                        current_io_count += get_potential_body_params(interm_op);
                    }
                } else {
                    // Unsupported op
                    break;
                }

                auto compute_gpr_params = [non_fused_postops_between_matmuls]() {
                    if (non_fused_postops_between_matmuls) {
                        // Loop depth could reach 4 because of SplitLoops optimization (M and N loops are split).
                        constexpr size_t loop_depth = 4;
                        // In case of SplitLoops, 3 register groups are needed:
                        // 1. Buffer before intermediate matmul
                        // 2. Buffer inside N block right after intermediate matmul before postops
                        // 3. Buffer outside of N block after postops
                        constexpr size_t reg_groups = 3;
                        return std::make_pair(loop_depth, reg_groups);
                    }
                    // If all postops between matmuls are fused, there will be no split loop by N dimension
                    // In this case, maximal loop depth is less, and GPRs, used for common buffer pointer,
                    // can be more efficiently reused
                    constexpr size_t loop_depth = 3;
                    constexpr size_t reg_groups = 1;
                    return std::make_pair(loop_depth, reg_groups);
                };

                const auto& [loops_depth, n_buffer_reg_groups] = compute_gpr_params();
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
