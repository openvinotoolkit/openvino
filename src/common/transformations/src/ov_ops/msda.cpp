// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/msda.hpp"

#include "augru_sequence_shape_inference.hpp"
#include "itt.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "ov_ops/augru_sequence.hpp"
#include "scaled_dot_product_attention_shape_inference.hpp"
namespace ov {
namespace op {
namespace internal {

namespace {
// Overload << operator for vectors
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
    os << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        os << vec[i];
        if (i != vec.size() - 1) {
            os << ", ";
        }
    }
    os << "]";
    return os;
}
};  // namespace

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
    OPENVINO_ASSERT(get_input_size() == 5, "MSDA must have 5 inputs whereas it has ", get_input_size());

    auto value_ps = get_input_partial_shape(0);
    auto attention_weights_ps = get_input_partial_shape(4);
    set_output_type(0, get_input_element_type(0), {value_ps[0], attention_weights_ps[1], value_ps[2] * value_ps[3]});
}

}  // namespace internal
}  // namespace op
}  // namespace ov