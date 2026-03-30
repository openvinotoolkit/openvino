// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_gated_delta_net.hpp"

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
        return !rank.is_dynamic() && is_rank_compatible_any_of(rank.get_length(), allowed_ranks);
    };

    auto type_check = [&](const Type& type) {
        auto it = std::find(allowed_types.begin(), allowed_types.end(), tp);
        return !type.is_dynamic() && (allowed_types.empty() || it != allowed_types.end());
    };

    NODE_VALIDATION_CHECK(node,
                          rank_check(rank),
                          "Rank of `",
                          input_name,
                          "` input should be in [",
                          join(allowed_ranks),
                          "] list, but it is ",
                          rank,
                          ".");

    NODE_VALIDATION_CHECK(node,
                          type_check(tp),
                          "Element type of `",
                          input_name,
                          "` input should be in [",
                          join(allowed_types),
                          "] list, but it is ",
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
                                       const Output<Node>& block_indices,
                                       const Output<Node>& block_indices_begins,
                                       const Output<Node>& past_lens,
                                       const Output<Node>& cache_interval,
                                       bool fuse_qk_l2norm,
                                       float q_l2_norm_eps,
                                       float k_l2_norm_eps)
    : Op({query,
          key,
          value,
          recurrent_state_table,
          gate,
          beta,
          subsequence_begins,
          block_indices,
          block_indices_begins,
          past_lens,
          cache_interval}),
      m_fuse_qk_l2norm(fuse_qk_l2norm),
      m_q_l2_norm_eps(q_l2_norm_eps),
      m_k_l2_norm_eps(k_l2_norm_eps) {
    constructor_validate_and_infer_types();
}

PagedGatedDeltaNet::PagedGatedDeltaNet(const ov::OutputVector& args,
                                       bool fuse_qk_l2norm,
                                       float q_l2_norm_eps,
                                       float k_l2_norm_eps)
    : ov::op::Op(args),
      m_fuse_qk_l2norm(fuse_qk_l2norm),
      m_q_l2_norm_eps(q_l2_norm_eps),
      m_k_l2_norm_eps(k_l2_norm_eps) {
    constructor_validate_and_infer_types();
}

void PagedGatedDeltaNet::validate_and_infer_types() {
    OV_OP_SCOPE(PagedGatedDeltaNet_validate_and_infer_types);

    NODE_VALIDATION_CHECK(this,
                          get_input_size() == 11,
                          "PagedGatedDeltaNet expects 11 inputs, but it has ",
                          get_input_size());

    const std::vector<ov::element::Type> float_types = {ov::element::f32, ov::element::f16, ov::element::bf16};

    input_check(this, 0, "query", {3}, float_types);
    input_check(this, 1, "key", {3}, float_types);
    input_check(this, 2, "value", {3}, float_types);
    input_check(this, 3, "recurrent_state_table", {4}, float_types);
    input_check(this, 4, "gate", {2}, float_types);
    input_check(this, 5, "beta", {2}, float_types);
    input_check(this, 6, "subsequence_begins", {1}, {ov::element::i32});
    input_check(this, 7, "block_indices", {1}, {ov::element::i32});
    input_check(this, 8, "block_indices_begins", {1}, {ov::element::i32});
    input_check(this, 9, "past_lens", {1}, {ov::element::i32});
    input_check(this, 10, "cache_interval", {1}, {ov::element::i32});

    const auto output_shapes = shape_infer(this, ov::util::get_node_input_partial_shapes(*this));
    set_output_type(0, get_input_element_type(0), output_shapes[0]);
}

bool PagedGatedDeltaNet::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(PagedGatedDeltaNet_visit_attributes);
    visitor.on_attribute("fuse_qk_l2norm", m_fuse_qk_l2norm);
    visitor.on_attribute("q_l2_norm_eps", m_q_l2_norm_eps);
    visitor.on_attribute("k_l2_norm_eps", m_k_l2_norm_eps);
    return true;
}

std::shared_ptr<ov::Node> PagedGatedDeltaNet::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OV_OP_SCOPE(PagedGatedDeltaNet_clone_with_new_inputs);
    return std::make_shared<PagedGatedDeltaNet>(new_args, m_fuse_qk_l2norm, m_q_l2_norm_eps, m_k_l2_norm_eps);
}

}  // namespace ov::op::internal
