// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/block_sparse_attention.hpp"

#include "block_sparse_attention_shape_inference.hpp"
#include "itt.hpp"

namespace ov::op::v17 {

BlockSparseAttention::BlockSparseAttention(const OutputVector& inputs, int64_t block_size, bool causal)
    : Op(inputs),
      m_block_size(block_size),
      m_causal(causal) {
    constructor_validate_and_infer_types();
}

BlockSparseAttention::BlockSparseAttention(const Output<Node>& query,
                                           const Output<Node>& key,
                                           const Output<Node>& value,
                                           const Output<Node>& block_indices,
                                           const Output<Node>& block_indices_mask,
                                           const Output<Node>& scale,
                                           int64_t block_size,
                                           bool causal)
    : BlockSparseAttention({query, key, value, block_indices, block_indices_mask, scale}, block_size, causal) {}

BlockSparseAttention::BlockSparseAttention(const Output<Node>& query,
                                           const Output<Node>& key,
                                           const Output<Node>& value,
                                           const Output<Node>& block_indices,
                                           const Output<Node>& block_indices_mask,
                                           int64_t block_size,
                                           bool causal)
    : BlockSparseAttention({query, key, value, block_indices, block_indices_mask}, block_size, causal) {}

BlockSparseAttention::BlockSparseAttention(const Output<Node>& query,
                                           const Output<Node>& key,
                                           const Output<Node>& value,
                                           const Output<Node>& block_indices,
                                           int64_t block_size,
                                           bool causal)
    : BlockSparseAttention({query, key, value, block_indices}, block_size, causal) {}

void BlockSparseAttention::validate_and_infer_types() {
    OV_OP_SCOPE(v17_BlockSparseAttention_validate_and_infer_types);

    const auto& input_size = get_input_size();
    NODE_VALIDATION_CHECK(this,
                          input_size >= 4 && input_size <= 6,
                          "BlockSparseAttention expects 4 to 6 inputs (query, key, value, block_indices, "
                          "[block_indices_mask], [scale]), got ",
                          input_size,
                          ".");
    NODE_VALIDATION_CHECK(this, m_block_size > 0, "The 'block_size' attribute must be a positive integer.");

    const bool has_mask = input_size >= 5;
    const bool has_scale = input_size == 6;

    // query / key / value must share a floating-point type; block_indices is a separate
    // integral input; block_indices_mask (if present) is boolean-valued (u8 is also accepted,
    // since plugins such as CPU normalize boolean tensors to u8 storage -- e.g. via
    // ov::pass::ConvertPrecision -- before an op ever sees them, and the two are byte-for-byte
    // interchangeable for a strictly 0/1 mask); scale (if present) must merge with the
    // floating-point data type.
    auto data_type = get_input_element_type(0);
    for (size_t i : {1u, 2u}) {
        NODE_VALIDATION_CHECK(this,
                              element::Type::merge(data_type, data_type, get_input_element_type(i)),
                              "The element types of query, key and value must match.");
    }
    NODE_VALIDATION_CHECK(this,
                          data_type.is_dynamic() || data_type.is_real(),
                          "The element type of query, key and value must be a floating-point type.");

    const auto& block_indices_type = get_input_element_type(3);
    NODE_VALIDATION_CHECK(this,
                          block_indices_type.is_dynamic() || block_indices_type.is_integral_number(),
                          "The element type of 'block_indices' must be an integer type.");

    if (has_mask) {
        const auto& mask_type = get_input_element_type(4);
        NODE_VALIDATION_CHECK(
            this,
            mask_type.is_dynamic() || mask_type == element::boolean || mask_type == element::u8,
            "The element type of 'block_indices_mask' must be boolean (u8 is also accepted).");
    }
    if (has_scale) {
        NODE_VALIDATION_CHECK(this,
                              element::Type::merge(data_type, data_type, get_input_element_type(5)),
                              "The element type of 'scale' must match query/key/value.");
    }

    const auto& input_shapes = ov::util::get_node_input_partial_shapes(*this);
    const auto output_shapes = shape_infer(this, input_shapes);
    set_output_type(0, data_type, output_shapes[0]);
}

bool BlockSparseAttention::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v17_BlockSparseAttention_visit_attributes);
    visitor.on_attribute("block_size", m_block_size);
    visitor.on_attribute("causal", m_causal);
    return true;
}

std::shared_ptr<Node> BlockSparseAttention::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v17_BlockSparseAttention_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<BlockSparseAttention>(new_args, m_block_size, m_causal);
}

}  // namespace ov::op::v17
