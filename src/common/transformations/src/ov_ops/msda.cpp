// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/msda.hpp"

#include "itt.hpp"
namespace ov {
namespace op {
namespace internal {

MSDA::MSDA(const OutputVector& inputs) : Op(inputs) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<ov::Node> MSDA::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(internal_MSDA_clone_with_new_inputs);
    return std::make_shared<MSDA>(new_args);
}

bool MSDA::visit_attributes(ov::AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(internal_MSDA_visit_attributes);
    return true;
}

// Inputs:
//     value : (bs, num_keys, num_heads, embed_dims)
//     value_spatial_shapes : (num_levels, 2), last dimension 2 represent (h, w)
//     level_start_index : (num_levels, ) and can be represented
//     as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
//     sampling_locations : (bs ,num_queries, num_heads, num_levels, num_points, 2),
//         the last dimension 2 represent (x, y).
//     attention_weights : The weight of sampling points used
//         when calculate the attention, has shape
//         (bs, num_queries, num_heads, num_levels, num_points),

// Returns:
//     output: has shape (bs, num_queries, num_heads * embed_dims)
void MSDA::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(internal_MSDA_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this, get_input_size() == 5, "MSDA must have 5 inputs whereas it has ", get_input_size());

    const auto& value_ps = get_input_partial_shape(0);
    const auto& locations_ps = get_input_partial_shape(3);
    const auto& attention_weights_ps = get_input_partial_shape(4);
    NODE_VALIDATION_CHECK(this,
                          value_ps.rank().compatible(4),
                          "MSDA value input must be 4D (bs, num_keys, num_heads, embed_dims), got ",
                          value_ps);
    NODE_VALIDATION_CHECK(this,
                          locations_ps.rank().compatible(6),
                          "MSDA sampling_locations input must be 6D, got ",
                          locations_ps);
    NODE_VALIDATION_CHECK(this,
                          attention_weights_ps.rank().compatible(5),
                          "MSDA attention_weights input must be 5D, got ",
                          attention_weights_ps);

    // Dimensions of inputs with a dynamic rank stay dynamic in the output.
    ov::Dimension batch, num_queries, channels;
    if (value_ps.rank().is_static()) {
        batch = value_ps[0];
        channels = value_ps[2] * value_ps[3];
    }
    if (attention_weights_ps.rank().is_static())
        num_queries = attention_weights_ps[1];
    set_output_type(0, get_input_element_type(0), ov::PartialShape{batch, num_queries, channels});
}

}  // namespace internal
}  // namespace op
}  // namespace ov