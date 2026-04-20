// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_gated_delta_net.hpp"

#include <string_view>

#include "dimension_util.hpp"
#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/op.hpp"
#include "paged_gated_delta_net_shape_inference.hpp"

namespace {

// Validates input rank and type for a node input.
inline void input_check(const ov::Node* node,
                        size_t idx,
                        const std::string_view input_name,
                        std::initializer_list<ov::Rank>&& allowed_ranks,
                        const std::vector<ov::element::Type>& allowed_types) {
    using namespace ov;
    using namespace ov::util;
    using namespace ov::element;

    const auto& rank = node->get_input_partial_shape(idx).rank();
    const auto& tp = node->get_input_element_type(idx);

    auto rank_check = [&](const Rank& rank) {
        return rank.is_dynamic() || is_rank_compatible_any_of(rank.get_length(), allowed_ranks);
    };

    auto type_check = [&](const Type& type) {
        auto it = std::find(allowed_types.begin(), allowed_types.end(), type);
        return type.is_dynamic() || allowed_types.empty() || it != allowed_types.end();
    };

    NODE_VALIDATION_CHECK(node,
                          rank_check(rank),
                          "Rank of `",
                          input_name,
                          "` input must be one of [",
                          join(allowed_ranks),
                          "]. Got: ",
                          rank,
                          ".");

    NODE_VALIDATION_CHECK(node,
                          type_check(tp),
                          "Element type of `",
                          input_name,
                          "` input must be one of [",
                          join(allowed_types),
                          "]. Got: ",
                          tp,
                          ".");
}
}  // namespace

namespace ov::op::internal {

PagedGatedDeltaNet::PagedGatedDeltaNet(const Output<Node>& query,
                                       const Output<Node>& key,
                                       const Output<Node>& value,
                                       const Output<Node>& recurrent_state_table,
                                       const Output<Node>& gate,
                                       const Output<Node>& beta,
                                       const Output<Node>& subsequence_begins,
                                       const Output<Node>& la_block_indices,
                                       const Output<Node>& la_block_indices_begins,
                                       const Output<Node>& processed_tokens,
                                       const Output<Node>& cache_interval,
                                       bool use_qk_l2norm,
                                       float q_l2_norm_eps,
                                       float k_l2_norm_eps)
    : Op({query,
          key,
          value,
          recurrent_state_table,
          gate,
          beta,
          subsequence_begins,
          la_block_indices,
          la_block_indices_begins,
          processed_tokens,
          cache_interval}),
      m_use_qk_l2norm(use_qk_l2norm),
      m_q_l2_norm_eps(q_l2_norm_eps),
      m_k_l2_norm_eps(k_l2_norm_eps) {
    constructor_validate_and_infer_types();
}

PagedGatedDeltaNet::PagedGatedDeltaNet(const ov::OutputVector& args,
                                       bool use_qk_l2norm,
                                       float q_l2_norm_eps,
                                       float k_l2_norm_eps)
    : ov::op::Op(args),
      m_use_qk_l2norm(use_qk_l2norm),
      m_q_l2_norm_eps(q_l2_norm_eps),
      m_k_l2_norm_eps(k_l2_norm_eps) {
    constructor_validate_and_infer_types();
}

void PagedGatedDeltaNet::validate_and_infer_types() {
    OV_OP_SCOPE(PagedGatedDeltaNet_validate_and_infer_types);

    NODE_VALIDATION_CHECK(this,
                          get_input_size() == 11,
                          "PagedGatedDeltaNet expects 11 inputs. Got: ",
                          get_input_size());

    static const std::vector<ov::element::Type> float_types = {ov::element::f32, ov::element::f16, ov::element::bf16};
    static const std::vector<ov::element::Type> integer_types = {ov::element::i32, ov::element::i64};

    input_check(this, 0, "query", {3}, float_types);
    input_check(this, 1, "key", {3}, float_types);
    input_check(this, 2, "value", {3}, float_types);
    input_check(this, 3, "recurrent_state_table", {4}, float_types);
    input_check(this, 4, "gate", {2}, float_types);
    input_check(this, 5, "beta", {2}, float_types);
    input_check(this, 6, "subsequence_begins", {1}, integer_types);
    input_check(this, 7, "la_block_indices", {1}, integer_types);
    input_check(this, 8, "la_block_indices_begins", {1}, integer_types);
    input_check(this, 9, "processed_tokens", {1}, integer_types);
    input_check(this, 10, "cache_interval", {1}, integer_types);

    ov::element::Type common_float_type = get_input_element_type(0);
    NODE_VALIDATION_CHECK(
        this,
        ov::element::Type::merge(common_float_type, common_float_type, get_input_element_type(1)) &&
            ov::element::Type::merge(common_float_type, common_float_type, get_input_element_type(2)) &&
            ov::element::Type::merge(common_float_type, common_float_type, get_input_element_type(3)) &&
            ov::element::Type::merge(common_float_type, common_float_type, get_input_element_type(4)) &&
            ov::element::Type::merge(common_float_type, common_float_type, get_input_element_type(5)),
        "PagedGatedDeltaNet expects query, key, value, recurrent_state_table, gate, and beta to "
        "have the same floating-point element type.");

    NODE_VALIDATION_CHECK(this,
                          m_q_l2_norm_eps > 0.0f,
                          "Attribute 'q_l2_norm_eps' must be a positive floating-point number. Got: ",
                          m_q_l2_norm_eps);
    NODE_VALIDATION_CHECK(this,
                          m_k_l2_norm_eps > 0.0f,
                          "Attribute 'k_l2_norm_eps' must be a positive floating-point number. Got: ",
                          m_k_l2_norm_eps);

    const auto output_shapes = shape_infer(this, ov::util::get_node_input_partial_shapes(*this));
    set_output_type(0, common_float_type, output_shapes[0]);
}

bool PagedGatedDeltaNet::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(PagedGatedDeltaNet_visit_attributes);
    visitor.on_attribute("use_qk_l2norm", m_use_qk_l2norm);
    visitor.on_attribute("q_l2_norm_eps", m_q_l2_norm_eps);
    visitor.on_attribute("k_l2_norm_eps", m_k_l2_norm_eps);
    return true;
}

std::shared_ptr<ov::Node> PagedGatedDeltaNet::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OV_OP_SCOPE(PagedGatedDeltaNet_clone_with_new_inputs);
    return std::make_shared<PagedGatedDeltaNet>(new_args, m_use_qk_l2norm, m_q_l2_norm_eps, m_k_l2_norm_eps);
}

}  // namespace ov::op::internal
