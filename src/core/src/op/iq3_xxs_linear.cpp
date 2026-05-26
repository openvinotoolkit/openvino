// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/iq3_xxs_linear.hpp"

#include "openvino/core/validation_util.hpp"

namespace ov {
namespace op {
namespace internal {

IQ3XXSLinear::IQ3XXSLinear(const Output<Node>& activation,
                           const Output<Node>& compressed_weights,
                           const ov::Shape& weight_shape,
                           int64_t block_size,
                           int64_t bytes_per_block)
    : Op({activation, compressed_weights}),
      m_weight_shape(weight_shape),
      m_block_size(block_size),
      m_bytes_per_block(bytes_per_block) {
    constructor_validate_and_infer_types();
}

bool IQ3XXSLinear::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("weight_shape", m_weight_shape);
    visitor.on_attribute("block_size", m_block_size);
    visitor.on_attribute("bytes_per_block", m_bytes_per_block);
    return true;
}

void IQ3XXSLinear::validate_and_infer_types() {
    // Input 0: activation [M, K] or [batch..., M, K]
    const auto& activation_type = get_input_element_type(0);
    const auto& activation_pshape = get_input_partial_shape(0);

    // Input 1: compressed weights blob [total_bytes] - must be u8
    const auto& weights_type = get_input_element_type(1);
    NODE_VALIDATION_CHECK(this,
                          weights_type == element::u8,
                          "Compressed weights must be u8 type, got: ",
                          weights_type);

    // Validate weight_shape: [N, K]
    NODE_VALIDATION_CHECK(this,
                          m_weight_shape.size() == 2,
                          "weight_shape must be 2D [N, K], got rank: ",
                          m_weight_shape.size());

    const int64_t N = static_cast<int64_t>(m_weight_shape[0]);
    const int64_t K = static_cast<int64_t>(m_weight_shape[1]);

    // Validate K is compatible with block_size
    NODE_VALIDATION_CHECK(this,
                          K % m_block_size == 0,
                          "K (", K, ") must be divisible by block_size (", m_block_size, ")");

    // Validate compressed data size
    const int64_t blocks_per_row = K / m_block_size;
    const int64_t expected_bytes = N * blocks_per_row * m_bytes_per_block;
    if (get_input_partial_shape(1).is_static()) {
        const auto& weights_shape = get_input_partial_shape(1).to_shape();
        NODE_VALIDATION_CHECK(this,
                              weights_shape.size() == 1,
                              "Compressed weights must be 1D blob");
        NODE_VALIDATION_CHECK(this,
                              static_cast<int64_t>(weights_shape[0]) == expected_bytes,
                              "Compressed weights size mismatch: expected ",
                              expected_bytes, " bytes, got ", weights_shape[0]);
    }

    // Output shape: activation leading dims + N
    // activation: [..., M, K] -> output: [..., M, N]
    if (activation_pshape.rank().is_dynamic()) {
        set_output_type(0, activation_type, ov::PartialShape::dynamic());
    } else {
        auto output_pshape = activation_pshape;
        // Last dim of activation (K) replaced by N (from weight_shape[0])
        output_pshape[output_pshape.rank().get_length() - 1] = N;
        set_output_type(0, activation_type, output_pshape);
    }
}

std::shared_ptr<Node> IQ3XXSLinear::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<IQ3XXSLinear>(new_args[0],
                                          new_args[1],
                                          m_weight_shape,
                                          m_block_size,
                                          m_bytes_per_block);
}

}  // namespace internal
}  // namespace op
}  // namespace ov
