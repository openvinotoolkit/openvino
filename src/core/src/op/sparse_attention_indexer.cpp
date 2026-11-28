// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/sparse_attention_indexer.hpp"

#include "itt.hpp"
#include "sparse_attention_indexer_shape_inference.hpp"

namespace ov::op::v17 {

SparseAttentionIndexer::SparseAttentionIndexer(const OutputVector& inputs,
                                               int64_t key_block_size,
                                               int64_t token_budget,
                                               int64_t num_selection_heads)
    : Op(inputs),
      m_key_block_size(key_block_size),
      m_token_budget(token_budget),
      m_num_selection_heads(num_selection_heads) {
    constructor_validate_and_infer_types();
}

SparseAttentionIndexer::SparseAttentionIndexer(const Output<Node>& index_query,
                                               const Output<Node>& block_index_keys,
                                               const Output<Node>& key_length,
                                               int64_t key_block_size,
                                               int64_t token_budget,
                                               int64_t num_selection_heads)
    : SparseAttentionIndexer(OutputVector{index_query, block_index_keys, key_length},
                             key_block_size,
                             token_budget,
                             num_selection_heads) {}

SparseAttentionIndexer::SparseAttentionIndexer(const Output<Node>& index_query,
                                               const Output<Node>& block_index_keys,
                                               const Output<Node>& key_length,
                                               const Output<Node>& index_weights,
                                               int64_t key_block_size,
                                               int64_t token_budget,
                                               int64_t num_selection_heads)
    : SparseAttentionIndexer(OutputVector{index_query, block_index_keys, key_length, index_weights},
                             key_block_size,
                             token_budget,
                             num_selection_heads) {}

void SparseAttentionIndexer::validate_and_infer_types() {
    OV_OP_SCOPE(v17_SparseAttentionIndexer_validate_and_infer_types);
    const auto input_count = get_input_size();
    NODE_VALIDATION_CHECK(this,
                          input_count == 3 || input_count == 4,
                          "The number of inputs must be 3 or 4. Got: ",
                          input_count);

    element::Type data_et = get_input_element_type(0);
    NODE_VALIDATION_CHECK(this,
                          element::Type::merge(data_et, data_et, get_input_element_type(1)),
                          "The element types of index_query and block_index_keys must be the same. Got: ",
                          get_input_element_type(0),
                          ", ",
                          get_input_element_type(1));
    if (input_count == 4) {
        NODE_VALIDATION_CHECK(this,
                              element::Type::merge(data_et, data_et, get_input_element_type(3)),
                              "The element type of index_weights must be the same as of index_query. Got: ",
                              get_input_element_type(3));
    }
    NODE_VALIDATION_CHECK(this,
                          data_et.is_dynamic() || data_et.is_real(),
                          "The element type of index_query and block_index_keys must be a floating point type. Got: ",
                          data_et);

    const auto& key_length_et = get_input_element_type(2);
    NODE_VALIDATION_CHECK(this,
                          key_length_et.is_dynamic() || key_length_et == element::i32 || key_length_et == element::i64,
                          "The element type of key_length must be i32 or i64. Got: ",
                          key_length_et);

    const auto output_shapes = shape_infer(this, ov::util::get_node_input_partial_shapes(*this));
    set_output_type(0, element::i32, output_shapes[0]);
    set_output_type(1, element::i32, output_shapes[1]);
}

bool SparseAttentionIndexer::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v17_SparseAttentionIndexer_visit_attributes);
    visitor.on_attribute("key_block_size", m_key_block_size);
    visitor.on_attribute("token_budget", m_token_budget);
    visitor.on_attribute("num_selection_heads", m_num_selection_heads);
    return true;
}

std::shared_ptr<Node> SparseAttentionIndexer::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v17_SparseAttentionIndexer_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<SparseAttentionIndexer>(new_args, m_key_block_size, m_token_budget, m_num_selection_heads);
}

}  // namespace ov::op::v17
