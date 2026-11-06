// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/experimental/draft_block_mask.hpp"

#include <cstdint>

#include "draft_block_mask_shape_inference.hpp"
#include "itt.hpp"
#include "openvino/core/validation_util.hpp"

namespace {

// Validates input rank. Dynamic rank always passes. Mirrors BlockSparseAttention's own file-local
// helper of the same name/contract.
void dbm_input_check(const ov::Node* node,
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

DraftBlockMask::DraftBlockMask(const Output<Node>& q_windows,
                               const Output<Node>& k_windows,
                               const Output<Node>& local_mask,
                               int64_t num_heads,
                               int64_t seqlen,
                               int64_t topk,
                               int64_t batch_size)
    : Op({q_windows, k_windows, local_mask}),
      m_num_heads(num_heads),
      m_seqlen(seqlen),
      m_topk(topk),
      m_batch_size(batch_size) {
    constructor_validate_and_infer_types();
}

DraftBlockMask::DraftBlockMask(const Output<Node>& q_windows,
                               const Output<Node>& k_windows,
                               const Output<Node>& local_mask,
                               const Output<Node>& seqlen,
                               const Output<Node>& topk,
                               int64_t num_heads,
                               int64_t batch_size)
    : Op({q_windows, k_windows, local_mask, seqlen, topk}),
      m_num_heads(num_heads),
      m_seqlen(-1),
      m_topk(-1),
      m_batch_size(batch_size) {
    constructor_validate_and_infer_types();
}

void DraftBlockMask::validate_and_infer_types() {
    OV_OP_SCOPE(DraftBlockMask_validate_and_infer_types);

    const auto& input_size = get_input_size();
    NODE_VALIDATION_CHECK(this,
                          input_size == 3 || input_size == 5,
                          "DraftBlockMask expects 3 or 5 inputs, but it has ",
                          input_size);
    NODE_VALIDATION_CHECK(this, m_num_heads > 0, "num_heads must be positive, got: ", m_num_heads);
    NODE_VALIDATION_CHECK(this, m_batch_size > 0, "batch_size must be positive, got: ", m_batch_size);
    if (input_size == 3) {
        NODE_VALIDATION_CHECK(this, m_seqlen > 0, "seqlen must be positive, got: ", m_seqlen);
        NODE_VALIDATION_CHECK(this, m_topk > 0, "topk must be positive, got: ", m_topk);
    } else {
        dbm_input_check(this, 3, "seqlen", {0, 1});
        dbm_input_check(this, 4, "topk", {0, 1});
        for (const size_t idx : {size_t{3}, size_t{4}}) {
            const auto& t = get_input_element_type(idx);
            NODE_VALIDATION_CHECK(this,
                                  t.is_dynamic() || t == element::i32 || t == element::i64,
                                  "DraftBlockMask expects an i32 or i64 seqlen/topk element type.");
        }
    }

    const auto& q_type = get_input_element_type(0);
    const auto& k_type = get_input_element_type(1);
    NODE_VALIDATION_CHECK(this,
                          q_type.is_dynamic() || q_type.is_real(),
                          "DraftBlockMask expects a floating-point q_windows element type.");
    element::Type merged_type;
    NODE_VALIDATION_CHECK(this,
                          element::Type::merge(merged_type, q_type, k_type),
                          "q_windows and k_windows must have the same element type.");

    // `element::u8` is accepted alongside `element::boolean`: mirrors BlockSparseAttention's own
    // dual-accept precedent -- plugins (e.g. intel_gpu's ConvertPrecision pass) may rewrite `boolean`
    // mask edges to `u8` ahead of this op's own validation, and both types share the same
    // one-byte-per-element 0/1 layout the reference/GPU kernels read as raw bytes.
    const auto& mask_type = get_input_element_type(2);
    NODE_VALIDATION_CHECK(this,
                          mask_type.is_dynamic() || mask_type == element::boolean || mask_type == element::u8,
                          "DraftBlockMask expects a boolean (or u8) local_mask element type.");

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);
    const auto output_shapes = shape_infer(this, input_shapes);
    set_output_type(0, element::boolean, output_shapes[0]);
}

bool DraftBlockMask::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(DraftBlockMask_visit_attributes);
    visitor.on_attribute("num_heads", m_num_heads);
    // `seqlen`/`topk` are `-1` sentinels when this op was built via the runtime-input (5-input)
    // constructor -- still visited unconditionally for a stable serialization schema.
    visitor.on_attribute("seqlen", m_seqlen);
    visitor.on_attribute("topk", m_topk);
    visitor.on_attribute("batch_size", m_batch_size);
    return true;
}

std::shared_ptr<ov::Node> DraftBlockMask::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OV_OP_SCOPE(DraftBlockMask_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    if (new_args.size() == 5) {
        return std::make_shared<DraftBlockMask>(new_args.at(0),
                                                new_args.at(1),
                                                new_args.at(2),
                                                new_args.at(3),
                                                new_args.at(4),
                                                m_num_heads,
                                                m_batch_size);
    }
    return std::make_shared<DraftBlockMask>(new_args.at(0),
                                            new_args.at(1),
                                            new_args.at(2),
                                            m_num_heads,
                                            m_seqlen,
                                            m_topk,
                                            m_batch_size);
}

}  // namespace ov::op::experimental::v1
