// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/experimental/block_sparse_attention.hpp"

#include "block_sparse_attention_shape_inference.hpp"
#include "itt.hpp"
#include "openvino/core/validation_util.hpp"

namespace {

// Validates input rank. Dynamic rank always passes.
void input_check(const ov::Node* node,
                 size_t idx,
                 std::string_view input_name,
                 std::initializer_list<ov::Rank> allowed_ranks) {
    const auto& rank = node->get_input_partial_shape(idx).rank();
    NODE_VALIDATION_CHECK(node,
                          rank.is_dynamic() || ov::util::is_rank_compatible_any_of(rank, allowed_ranks),
                          "Rank of `",
                          input_name,
                          "` input is not compatible with the expected rank, got: ",
                          rank,
                          ".");
}

}  // namespace

namespace ov::op::experimental::v1 {

BlockSparseAttention::BlockSparseAttention(const Output<Node>& query,
                                           const Output<Node>& key,
                                           const Output<Node>& value,
                                           const Output<Node>& block_mask,
                                           int64_t block_size_q,
                                           int64_t block_size_kv,
                                           bool causal)
    : Op({query, key, value, block_mask}),
      m_causal(causal),
      m_block_size_q(block_size_q),
      m_block_size_kv(block_size_kv) {
    constructor_validate_and_infer_types();
}

BlockSparseAttention::BlockSparseAttention(const Output<Node>& query,
                                           const Output<Node>& key,
                                           const Output<Node>& value,
                                           const Output<Node>& block_mask,
                                           const Output<Node>& scale,
                                           int64_t block_size_q,
                                           int64_t block_size_kv,
                                           bool causal)
    : Op({query, key, value, block_mask, scale}),
      m_causal(causal),
      m_block_size_q(block_size_q),
      m_block_size_kv(block_size_kv) {
    constructor_validate_and_infer_types();
}

void BlockSparseAttention::validate_and_infer_types() {
    OV_OP_SCOPE(BlockSparseAttention_validate_and_infer_types);

    const auto& input_size = get_input_size();
    NODE_VALIDATION_CHECK(this,
                          input_size == 4 || input_size == 5,
                          "BlockSparseAttention expects 4 or 5 inputs, but it has ",
                          input_size);
    NODE_VALIDATION_CHECK(this, m_block_size_q > 0, "block_size_q must be positive, got: ", m_block_size_q);
    NODE_VALIDATION_CHECK(this, m_block_size_kv > 0, "block_size_kv must be positive, got: ", m_block_size_kv);

    input_check(this, 0, "query", {4});
    input_check(this, 1, "key", {4});
    input_check(this, 2, "value", {4});
    input_check(this, 3, "block_mask", {4});
    if (input_size == 5) {
        input_check(this, 4, "scale", {0});
    }

    // `element::u8` is accepted alongside `element::boolean`: plugins (e.g. intel_gpu's
    // ConvertPrecision pass) may rewrite `boolean` block_mask edges to `u8` ahead of this op's
    // own validation, and both types share the same one-byte-per-element 0/1 layout the reference
    // and GPU kernels read as raw bytes.
    const auto& mask_type = get_input_element_type(3);
    NODE_VALIDATION_CHECK(this,
                          mask_type.is_dynamic() || mask_type == element::boolean || mask_type == element::u8,
                          "BlockSparseAttention expects a boolean (or u8) block_mask element type.");

    auto out_type = get_input_element_type(0);
    for (size_t i = 1; i < input_size; ++i) {
        if (i == 3) {
            continue;  // block_mask is boolean, excluded from the merged compute dtype
        }
        NODE_VALIDATION_CHECK(this,
                              element::Type::merge(out_type, out_type, get_input_element_type(i)),
                              "Mixed input types are not supported.");
    }
    NODE_VALIDATION_CHECK(this,
                          out_type.is_real() || out_type.is_dynamic(),
                          "The element type of query/key/value/scale must be floating-point.");

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);
    const auto output_shapes = shape_infer(this, input_shapes);
    set_output_type(0, out_type, output_shapes[0]);
}

bool BlockSparseAttention::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(BlockSparseAttention_visit_attributes);
    visitor.on_attribute("causal", m_causal);
    visitor.on_attribute("block_size_q", m_block_size_q);
    visitor.on_attribute("block_size_kv", m_block_size_kv);
    return true;
}

std::shared_ptr<ov::Node> BlockSparseAttention::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OV_OP_SCOPE(BlockSparseAttention_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    if (new_args.size() == 5) {
        return std::make_shared<BlockSparseAttention>(new_args.at(0),
                                                      new_args.at(1),
                                                      new_args.at(2),
                                                      new_args.at(3),
                                                      new_args.at(4),
                                                      m_block_size_q,
                                                      m_block_size_kv,
                                                      m_causal);
    }
    return std::make_shared<BlockSparseAttention>(new_args.at(0),
                                                  new_args.at(1),
                                                  new_args.at(2),
                                                  new_args.at(3),
                                                  m_block_size_q,
                                                  m_block_size_kv,
                                                  m_causal);
}

}  // namespace ov::op::experimental::v1
