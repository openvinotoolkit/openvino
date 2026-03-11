// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/gated_delta_net.hpp"

#include "itt.hpp"

using namespace std;

ov::op::internal::GatedDeltaNet::GatedDeltaNet(const Output<Node>& query,
                                                const Output<Node>& key,
                                                const Output<Node>& value,
                                                const Output<Node>& recurrent_state,
                                                const Output<Node>& gate,
                                                const Output<Node>& beta)
    : Op({query, key, value, recurrent_state, gate, beta}) {
    constructor_validate_and_infer_types();
}

bool ov::op::internal::GatedDeltaNet::visit_attributes(AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(internal_GatedDeltaNet_visit_attributes);
    return true;
}

void ov::op::internal::GatedDeltaNet::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(internal_GatedDeltaNet_validate_and_infer_types);

    // Validate element types (all must be floating-point and compatible)
    auto result_et = element::dynamic;
    for (size_t i = 0; i < 6; ++i) {
        NODE_VALIDATION_CHECK(this,
                              element::Type::merge(result_et, result_et, get_input_element_type(i)),
                              "Element types for all inputs must be compatible floating-point types.");
        NODE_VALIDATION_CHECK(this,
                              get_input_element_type(i).is_dynamic() || get_input_element_type(i).is_real(),
                              "Input ",
                              i,
                              " must be a floating-point type, got: ",
                              get_input_element_type(i));
    }

    // query: [batch, seq_len, num_heads, key_head_dim]
    const auto& query_shape = get_input_partial_shape(0);
    // key: [batch, seq_len, num_heads, key_head_dim]
    const auto& key_shape = get_input_partial_shape(1);
    // value: [batch, seq_len, num_heads, value_head_dim]
    const auto& value_shape = get_input_partial_shape(2);
    // recurrent_state: [batch, num_heads, key_head_dim, value_head_dim]
    const auto& state_shape = get_input_partial_shape(3);
    // gate: [batch, seq_len, num_heads]
    const auto& gate_shape = get_input_partial_shape(4);
    // beta: [batch, seq_len, num_heads]
    const auto& beta_shape = get_input_partial_shape(5);

    NODE_VALIDATION_CHECK(this,
                          query_shape.rank().is_dynamic() || query_shape.rank().get_length() == 4,
                          "query must be a 4D tensor, got rank: ",
                          query_shape.rank());
    NODE_VALIDATION_CHECK(this,
                          key_shape.rank().is_dynamic() || key_shape.rank().get_length() == 4,
                          "key must be a 4D tensor, got rank: ",
                          key_shape.rank());
    NODE_VALIDATION_CHECK(this,
                          value_shape.rank().is_dynamic() || value_shape.rank().get_length() == 4,
                          "value must be a 4D tensor, got rank: ",
                          value_shape.rank());
    NODE_VALIDATION_CHECK(this,
                          state_shape.rank().is_dynamic() || state_shape.rank().get_length() == 4,
                          "recurrent_state must be a 4D tensor, got rank: ",
                          state_shape.rank());
    NODE_VALIDATION_CHECK(this,
                          gate_shape.rank().is_dynamic() || gate_shape.rank().get_length() == 3,
                          "gate must be a 3D tensor, got rank: ",
                          gate_shape.rank());
    NODE_VALIDATION_CHECK(this,
                          beta_shape.rank().is_dynamic() || beta_shape.rank().get_length() == 3,
                          "beta must be a 3D tensor, got rank: ",
                          beta_shape.rank());

    // Infer output shapes
    // output_attn: [batch, seq_len, num_heads, value_head_dim]
    ov::PartialShape attn_shape = ov::PartialShape::dynamic(4);
    // output_recurrent_state: [batch, num_heads, key_head_dim, value_head_dim]
    ov::PartialShape out_state_shape = ov::PartialShape::dynamic(4);

    if (query_shape.rank().is_static() && value_shape.rank().is_static()) {
        // batch from query[0], seq_len from query[1], num_heads from query[2]
        // value_head_dim from value[3]
        attn_shape = ov::PartialShape{query_shape[0], query_shape[1], query_shape[2], value_shape[3]};
    }

    if (state_shape.rank().is_static()) {
        // batch, num_heads, key_head_dim, value_head_dim all from recurrent_state
        out_state_shape = state_shape;
    }

    set_output_type(0, result_et, attn_shape);
    set_output_type(1, result_et, out_state_shape);
}

shared_ptr<ov::Node> ov::op::internal::GatedDeltaNet::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(internal_GatedDeltaNet_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<GatedDeltaNet>(new_args.at(0),
                                      new_args.at(1),
                                      new_args.at(2),
                                      new_args.at(3),
                                      new_args.at(4),
                                      new_args.at(5));
}
