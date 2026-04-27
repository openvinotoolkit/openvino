// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_gated_delta_net.hpp"

#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/op.hpp"
#include "paged_gated_delta_net_shape_inference.hpp"

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

    NODE_VALIDATION_CHECK(this, get_input_size() == 11);

    ov::element::Type common_float_type = get_input_element_type(0);
    const bool float_types_merge =
        ov::element::Type::merge(common_float_type, common_float_type, get_input_element_type(1)) &&
        ov::element::Type::merge(common_float_type, common_float_type, get_input_element_type(2)) &&
        ov::element::Type::merge(common_float_type, common_float_type, get_input_element_type(3)) &&
        ov::element::Type::merge(common_float_type, common_float_type, get_input_element_type(4)) &&
        ov::element::Type::merge(common_float_type, common_float_type, get_input_element_type(5));
    NODE_VALIDATION_CHECK(this,
                          float_types_merge,
                          "PagedGatedDeltaNet expects query, key, value, recurrent_state_table, gate, and beta to "
                          "have the same element type.");
    NODE_VALIDATION_CHECK(this,
                          common_float_type.is_dynamic() || common_float_type == ov::element::f32 ||
                              common_float_type == ov::element::f16 || common_float_type == ov::element::bf16,
                          "Float inputs must have f32, f16, or bf16 element type.");
    for (size_t i = 6; i < 11; ++i) {
        const auto& et = get_input_element_type(i);
        NODE_VALIDATION_CHECK(this,
                              et.is_dynamic() || et == ov::element::i32 || et == ov::element::i64,
                              "Integer inputs must have i32 or i64 element type.");
    }

    NODE_VALIDATION_CHECK(this,
                          m_q_l2_norm_eps > 0.0f,
                          "Attribute 'q_l2_norm_eps' must be a positive floating-point number.");
    NODE_VALIDATION_CHECK(this,
                          m_k_l2_norm_eps > 0.0f,
                          "Attribute 'k_l2_norm_eps' must be a positive floating-point number.");

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
    check_new_args_count(this, new_args);
    return std::make_shared<PagedGatedDeltaNet>(new_args, m_use_qk_l2norm, m_q_l2_norm_eps, m_k_l2_norm_eps);
}

}  // namespace ov::op::internal
