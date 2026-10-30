// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/sparse_attention.hpp"

#include "itt.hpp"
#include "sparse_attention_shape_inference.hpp"

namespace ov::op::v17 {

SparseAttention::SparseAttention(const OutputVector& inputs,
                                 int64_t key_block_size,
                                 int64_t query_block_size,
                                 bool causal)
    : Op(inputs),
      m_key_block_size(key_block_size),
      m_query_block_size(query_block_size),
      m_causal(causal) {
    constructor_validate_and_infer_types();
}

SparseAttention::SparseAttention(const Output<Node>& query,
                                 const Output<Node>& key,
                                 const Output<Node>& value,
                                 const Output<Node>& block_indices,
                                 const Output<Node>& block_count,
                                 int64_t key_block_size,
                                 int64_t query_block_size,
                                 bool causal)
    : SparseAttention(OutputVector{query, key, value, block_indices, block_count},
                      key_block_size,
                      query_block_size,
                      causal) {}

SparseAttention::SparseAttention(const Output<Node>& query,
                                 const Output<Node>& key,
                                 const Output<Node>& value,
                                 const Output<Node>& block_indices,
                                 const Output<Node>& block_count,
                                 const Output<Node>& key_length,
                                 int64_t key_block_size,
                                 int64_t query_block_size,
                                 bool causal)
    : SparseAttention(OutputVector{query, key, value, block_indices, block_count, key_length},
                      key_block_size,
                      query_block_size,
                      causal) {}

SparseAttention::SparseAttention(const Output<Node>& query,
                                 const Output<Node>& key,
                                 const Output<Node>& value,
                                 const Output<Node>& block_indices,
                                 const Output<Node>& block_count,
                                 const Output<Node>& key_length,
                                 const Output<Node>& scale,
                                 int64_t key_block_size,
                                 int64_t query_block_size,
                                 bool causal)
    : SparseAttention(OutputVector{query, key, value, block_indices, block_count, key_length, scale},
                      key_block_size,
                      query_block_size,
                      causal) {}

SparseAttention::SparseAttention(const Output<Node>& query,
                                 const Output<Node>& key,
                                 const Output<Node>& value,
                                 const Output<Node>& block_indices,
                                 const Output<Node>& block_count,
                                 const Output<Node>& key_length,
                                 const Output<Node>& scale,
                                 const Output<Node>& sinks,
                                 int64_t key_block_size,
                                 int64_t query_block_size,
                                 bool causal)
    : SparseAttention(OutputVector{query, key, value, block_indices, block_count, key_length, scale, sinks},
                      key_block_size,
                      query_block_size,
                      causal) {}

void SparseAttention::validate_and_infer_types() {
    OV_OP_SCOPE(v17_SparseAttention_validate_and_infer_types);
    const auto input_count = get_input_size();
    NODE_VALIDATION_CHECK(this,
                          5 <= input_count && input_count <= 8,
                          "The number of inputs must be from 5 to 8. Got: ",
                          input_count);

    element::Type data_et = get_input_element_type(0);
    NODE_VALIDATION_CHECK(this,
                          element::Type::merge(data_et, data_et, get_input_element_type(1)) &&
                              element::Type::merge(data_et, data_et, get_input_element_type(2)),
                          "The element types of query, key and value must be the same. Got: ",
                          get_input_element_type(0),
                          ", ",
                          get_input_element_type(1),
                          ", ",
                          get_input_element_type(2));
    NODE_VALIDATION_CHECK(this,
                          data_et.is_dynamic() || data_et.is_real(),
                          "The element type of query, key and value must be a floating point type. Got: ",
                          data_et);

    static constexpr const char* index_input_names[] = {"block_indices", "block_count", "key_length"};
    for (size_t i = 3; i < input_count && i < 6; ++i) {
        const auto& et = get_input_element_type(i);
        NODE_VALIDATION_CHECK(this,
                              et.is_dynamic() || et == element::i32 || et == element::i64,
                              "The element type of ",
                              index_input_names[i - 3],
                              " must be i32 or i64. Got: ",
                              et);
    }

    if (input_count >= 7) {
        const auto& scale_et = get_input_element_type(6);
        NODE_VALIDATION_CHECK(
            this,
            (scale_et.is_dynamic() || scale_et.is_real()) && element::Type::merge(data_et, data_et, scale_et),
            "The element type of scale must be a floating point type, the same as of query, key "
            "and value. Got: ",
            scale_et);
    }

    // The sink logits are not scaled and may be kept in a higher precision than the activations.
    if (input_count == 8) {
        const auto& sinks_et = get_input_element_type(7);
        NODE_VALIDATION_CHECK(this,
                              sinks_et.is_dynamic() || sinks_et.is_real(),
                              "The element type of sinks must be a floating point type. Got: ",
                              sinks_et);
    }

    const auto output_shapes = shape_infer(this, ov::util::get_node_input_partial_shapes(*this));
    set_output_type(0, data_et, output_shapes[0]);
}

bool SparseAttention::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v17_SparseAttention_visit_attributes);
    visitor.on_attribute("key_block_size", m_key_block_size);
    visitor.on_attribute("query_block_size", m_query_block_size);
    visitor.on_attribute("causal", m_causal);
    return true;
}

std::shared_ptr<Node> SparseAttention::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v17_SparseAttention_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<SparseAttention>(new_args, m_key_block_size, m_query_block_size, m_causal);
}

}  // namespace ov::op::v17
