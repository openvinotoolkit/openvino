// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sdpa.hpp"

#include <functional>
#include <iostream>
#include <numeric>

#include "../../logging.hpp"
#include "../online/group.hpp"     // online::Group
#include "../online/snapshot.hpp"  // online::Snapshot
#include "openvino/op/ops.hpp"
#include "openvino/pass/pattern/op/label.hpp"  // any_input
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/common_util.hpp"

namespace ov {
namespace npuw {
namespace patterns {
namespace attn {

namespace opp = ov::pass::pattern;

AttentionParams extractAttentionParamsFromSDPAPattern(const std::shared_ptr<ov::Node>& matmul1,
                                                      const std::shared_ptr<ov::Node>& matmul2,
                                                      const std::shared_ptr<ov::Node>& softmax_node,
                                                      const std::shared_ptr<ov::Node>& add_node) {
    AttentionParams params;

    try {
        // Extract shapes from matmul1 (Q * K^T)
        // matmul1 inputs: Q tensor and K tensor (after reshape)
        auto matmul1_input0_shape = matmul1->get_input_shape(0);   // Q shape
        auto matmul1_input1_shape = matmul1->get_input_shape(1);   // K shape (after reshape)
        auto matmul1_output_shape = matmul1->get_output_shape(0);  // attention scores shape

        // Extract shapes from matmul2 (attention_scores * V)
        auto matmul2_input1_shape = matmul2->get_input_shape(1);   // V shape (after reshape)
        auto matmul2_output_shape = matmul2->get_output_shape(0);  // output shape

        // Extract shapes from softmax
        auto softmax_shape = softmax_node->get_output_shape(0);

        LOG_DEBUG("SDPA Shape Analysis:");
        LOG_DEBUG("  MatMul1 input0 (Q) shape: " << ov::util::vector_to_string(matmul1_input0_shape));
        LOG_DEBUG("  MatMul1 input1 (K) shape: " << ov::util::vector_to_string(matmul1_input1_shape));
        LOG_DEBUG("  MatMul1 output shape: " << ov::util::vector_to_string(matmul1_output_shape));
        LOG_DEBUG("  MatMul2 input1 (V) shape: " << ov::util::vector_to_string(matmul2_input1_shape));
        LOG_DEBUG("  MatMul2 output shape: " << ov::util::vector_to_string(matmul2_output_shape));
        LOG_DEBUG("  Softmax shape: " << ov::util::vector_to_string(softmax_shape));

        // For typical SDPA pattern, shapes are:
        // Q: [batch, num_heads, seq_len, head_dim]
        // K: [batch, num_heads, head_dim, seq_len] (transposed)
        // V: [batch, num_heads, seq_len, head_dim]
        // attention_scores: [batch, num_heads, seq_len, seq_len]

        // Analyze attention scores shape (matmul1 output or softmax shape)
        auto attn_scores_shape = softmax_shape;

        // Check MatMul transpose parameters
        bool matmul1_transpose_a = false;
        bool matmul1_transpose_b = false;
        bool matmul2_transpose_a = false;
        bool matmul2_transpose_b = false;

        if (auto matmul1_op = std::dynamic_pointer_cast<ov::op::v0::MatMul>(matmul1)) {
            matmul1_transpose_a = matmul1_op->get_transpose_a();
            matmul1_transpose_b = matmul1_op->get_transpose_b();
            LOG_DEBUG("MatMul1 transpose_a: " << matmul1_transpose_a << ", transpose_b: " << matmul1_transpose_b);
        }

        if (auto matmul2_op = std::dynamic_pointer_cast<ov::op::v0::MatMul>(matmul2)) {
            matmul2_transpose_a = matmul2_op->get_transpose_a();
            matmul2_transpose_b = matmul2_op->get_transpose_b();
            LOG_DEBUG("MatMul2 transpose_a: " << matmul2_transpose_a << ", transpose_b: " << matmul2_transpose_b);
        }

        // Only handle 4D tensors for now
        if (attn_scores_shape.size() != 4) {
            LOG_WARN("Only 4D tensors are supported, found " << attn_scores_shape.size() << "D attention scores");
            return AttentionParams{};
        }

        params.batch_size = attn_scores_shape[0];
        params.num_heads = attn_scores_shape[1];
        params.sequence_length = attn_scores_shape[2];

        // Set dimension mappings based on actual tensor layouts and transpose settings
        // For Q (MatMul1 input A): typically [batch, num_heads, seq_len, head_dim]
        if (matmul1_transpose_a) {
            // Q is transposed: [batch, num_heads, head_dim, seq_len]
            params.q_dims = {0, 1, 3, 2};  // sequence and head_dim swapped
        } else {
            // Q is not transposed: [batch, num_heads, seq_len, head_dim]
            params.q_dims = {0, 1, 2, 3};  // standard layout
        }

        // For K (MatMul1 input B): the effective layout after reshape operations
        // In SDPA, K is typically transposed to [batch, num_heads, head_dim, seq_len] for Q*K^T
        if (matmul1_transpose_b) {
            // K input is further transposed: if K was [batch, num_heads, head_dim, seq_len],
            // transpose_b makes it [batch, num_heads, seq_len, head_dim]
            params.k_dims = {0, 1, 2, 3};
        } else {
            // K input is not transposed: [batch, num_heads, head_dim, seq_len] (typical for attention)
            params.k_dims = {0, 1, 3, 2};  // head_dim and sequence swapped
        }

        // For V (MatMul2 input B): typically [batch, num_heads, seq_len, head_dim]
        if (matmul2_transpose_b) {
            // V is transposed: [batch, num_heads, head_dim, seq_len]
            params.v_dims = {0, 1, 3, 2};  // sequence and head_dim swapped
        } else {
            // V is not transposed: [batch, num_heads, seq_len, head_dim]
            params.v_dims = {0, 1, 2, 3};  // standard layout
        }

        LOG_DEBUG("Dimension mappings:");
        LOG_DEBUG("  Q dims: [" << params.q_dims.batch_dim << ", " << params.q_dims.heads_dim << ", "
                                << params.q_dims.sequence_dim << ", " << params.q_dims.head_dim_idx << "]");
        LOG_DEBUG("  K dims: [" << params.k_dims.batch_dim << ", " << params.k_dims.heads_dim << ", "
                                << params.k_dims.sequence_dim << ", " << params.k_dims.head_dim_idx << "]");
        LOG_DEBUG("  V dims: [" << params.v_dims.batch_dim << ", " << params.v_dims.heads_dim << ", "
                                << params.v_dims.sequence_dim << ", " << params.v_dims.head_dim_idx << "]");

        // Extract head dimension from Q tensor (input0 of matmul1)
        if (matmul1_input0_shape.size() == 4) {
            if (matmul1_transpose_a) {
                // Q is transposed: [batch, num_heads, head_dim, seq_len]
                params.head_dim = matmul1_input0_shape[2];
            } else {
                // Q is not transposed: [batch, num_heads, seq_len, head_dim]
                params.head_dim = matmul1_input0_shape[3];
            }
        } else {
            LOG_WARN("Q tensor is not 4D, cannot extract head dimension reliably");
            return AttentionParams{};  // Return empty params for unsupported cases
        }

        // Verify consistency: Q*K^T should produce [batch, heads, seq_len, seq_len]
        auto expected_scores_size =
            params.batch_size * params.num_heads * params.sequence_length * params.sequence_length;
        auto actual_scores_size =
            std::accumulate(attn_scores_shape.begin(), attn_scores_shape.end(), size_t(1), std::multiplies<size_t>());

        if (expected_scores_size != actual_scores_size && expected_scores_size > 0) {
            LOG_DEBUG("Shape consistency warning: expected " << expected_scores_size << " elements, got "
                                                             << actual_scores_size);
        }

        // Store original shapes for reference
        params.q_shape = matmul1_input0_shape;
        params.k_shape = matmul1_input1_shape;
        params.v_shape = matmul2_input1_shape;
        params.output_shape = matmul2_output_shape;

        // Get data type from the first input
        params.data_type = matmul1->get_input_element_type(0);

        // Check if bias is present (add node should have bias as second input)
        params.has_bias = (add_node && add_node->get_input_size() > 1);
        if (params.has_bias && add_node) {
            auto bias_shape = add_node->get_input_shape(1);
            LOG_DEBUG("  Bias shape: " << ov::util::vector_to_string(bias_shape));
        }

        LOG_INFO("Extracted Attention Parameters:");
        LOG_INFO("  Batch size: " << params.batch_size);
        LOG_INFO("  Number of heads: " << params.num_heads);
        LOG_INFO("  Sequence length: " << params.sequence_length);
        LOG_INFO("  Head dimension: " << params.head_dim);
        LOG_INFO("  Data type: " << params.data_type);
        LOG_INFO("  Has bias: " << (params.has_bias ? "Yes" : "No"));
        LOG_INFO("  MatMul1 transpose_a: " << matmul1_transpose_a << ", transpose_b: " << matmul1_transpose_b);
        LOG_INFO("  MatMul2 transpose_a: " << matmul2_transpose_a << ", transpose_b: " << matmul2_transpose_b);
        LOG_INFO("  Total parameters: " << (params.batch_size * params.num_heads * params.sequence_length *
                                            params.head_dim));
    } catch (const std::exception& e) {
        LOG_DEBUG("Error extracting attention parameters: " << e.what());
        // Return default/empty parameters on error
        params = AttentionParams{};
    }

    return params;
}

/*
    SDPA Pattern:
            Convert
                \       /
                 Concat
                    |
                Unsqueeze
                    |
                Broadcast   Convert
                    |       \       /
                Reshape       Concat
        \           /           |
            MatMul           Unsqueeze
    \       /                   |
       Add                   Broadcast
        |                       |
     Softmax                Reshape
            \               /
                  MatMul
                    |
                Transpose
                    |
                Reshape
                    |
*/

SDPA::SDPA(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& isol_tag) {
    auto convert1 = opp::wrap_type<ov::op::v0::Convert>({opp::any_input()});
    auto concat1 = opp::wrap_type<ov::op::v0::Concat>({convert1, opp::any_input()});
    auto unsqueeze1 = opp::wrap_type<ov::op::v0::Unsqueeze>({concat1, opp::any_input()});
    auto broadcast1 = opp::wrap_type<ov::op::v3::Broadcast>({unsqueeze1, opp::any_input()});
    auto reshape1 = opp::wrap_type<ov::op::v1::Reshape>({broadcast1, opp::any_input()});

    auto convert2 = opp::wrap_type<ov::op::v0::Convert>({opp::any_input()});
    auto concat2 = opp::wrap_type<ov::op::v0::Concat>({convert2, opp::any_input()});
    auto unsqueeze2 = opp::wrap_type<ov::op::v0::Unsqueeze>({concat2, opp::any_input()});
    auto broadcast2 = opp::wrap_type<ov::op::v3::Broadcast>({unsqueeze2, opp::any_input()});
    auto reshape2 = opp::wrap_type<ov::op::v1::Reshape>({broadcast2, opp::any_input()});

    auto matmul1 = opp::wrap_type<ov::op::v0::MatMul>({opp::any_input(), reshape1});
    auto add = opp::wrap_type<ov::op::v1::Add>({matmul1, opp::any_input()});
    auto softmax = opp::wrap_type<ov::op::v8::Softmax>({add});

    auto matmul2 = opp::wrap_type<ov::op::v0::MatMul>({softmax, reshape2});
    auto transpose = opp::wrap_type<ov::op::v1::Transpose>({matmul2, opp::any_input()});
    auto reshape3 = opp::wrap_type<ov::op::v1::Reshape>({transpose, opp::any_input()});

    auto node_to_gptr = snapshot->getNodeToGroupMap();

    // Note: Use [=] to make sure the above objects stay alive in the callback
    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();

        auto matched_softmax = node_to_output.at(softmax).get_node_shared_ptr();

        // Check softmax shape: skip if second-to-last dimension is 1 so that not isolate for kvcache model
        // FIXME: check for a more proper condition
        auto softmax_shape = matched_softmax->get_output_shape(0);
        if (softmax_shape.size() < 2) {
            return false;
        }
        auto second_to_last_dim = softmax_shape[softmax_shape.size() - 2];
        if (second_to_last_dim == 1) {
            LOG_DEBUG("SDPA pattern skipped: softmax second-to-last dimension is 1");
            return false;
        }

        LOG_INFO("SDPA pattern matched!");

        auto matched_convert1 = node_to_output.at(convert1).get_node_shared_ptr();
        auto matched_concat1 = node_to_output.at(concat1).get_node_shared_ptr();
        auto matched_unsqueeze1 = node_to_output.at(unsqueeze1).get_node_shared_ptr();
        auto matched_broadcast1 = node_to_output.at(broadcast1).get_node_shared_ptr();
        auto matched_reshape1 = node_to_output.at(reshape1).get_node_shared_ptr();

        auto matched_convert2 = node_to_output.at(convert2).get_node_shared_ptr();
        auto matched_concat2 = node_to_output.at(concat2).get_node_shared_ptr();
        auto matched_unsqueeze2 = node_to_output.at(unsqueeze2).get_node_shared_ptr();
        auto matched_broadcast2 = node_to_output.at(broadcast2).get_node_shared_ptr();
        auto matched_reshape2 = node_to_output.at(reshape2).get_node_shared_ptr();

        auto matched_matmul1 = node_to_output.at(matmul1).get_node_shared_ptr();
        auto matched_add = node_to_output.at(add).get_node_shared_ptr();
        auto matched_matmul2 = node_to_output.at(matmul2).get_node_shared_ptr();
        auto matched_transpose = node_to_output.at(transpose).get_node_shared_ptr();
        auto matched_reshape3 = node_to_output.at(reshape3).get_node_shared_ptr();

        // Isolate all matched nodes with the given tag
        node_to_gptr->at(matched_convert1)->isolate(isol_tag);
        node_to_gptr->at(matched_concat1)->isolate(isol_tag);
        node_to_gptr->at(matched_unsqueeze1)->isolate(isol_tag);
        node_to_gptr->at(matched_broadcast1)->isolate(isol_tag);
        node_to_gptr->at(matched_reshape1)->isolate(isol_tag);

        node_to_gptr->at(matched_convert2)->isolate(isol_tag);
        node_to_gptr->at(matched_concat2)->isolate(isol_tag);
        node_to_gptr->at(matched_unsqueeze2)->isolate(isol_tag);
        node_to_gptr->at(matched_broadcast2)->isolate(isol_tag);
        node_to_gptr->at(matched_reshape2)->isolate(isol_tag);

        node_to_gptr->at(matched_matmul1)->isolate(isol_tag);
        node_to_gptr->at(matched_add)->isolate(isol_tag);
        node_to_gptr->at(matched_softmax)->isolate(isol_tag);
        node_to_gptr->at(matched_matmul2)->isolate(isol_tag);
        node_to_gptr->at(matched_transpose)->isolate(isol_tag);
        node_to_gptr->at(matched_reshape3)->isolate(isol_tag);

        return false;  // root hasn't changed
    };

    register_matcher(std::make_shared<opp::Matcher>(reshape3, "TagSDPA"), std::move(callback));
}

}  // namespace attn
}  // namespace patterns
}  // namespace npuw
}  // namespace ov
