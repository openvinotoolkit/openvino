// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_attention.hpp"

#include "dimension_util.hpp"
#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/op.hpp"
#include "paged_attention_shape_inference.hpp"

namespace {

// Validates input rank and type for a node input.
// We consider that dynamic rank/type are always valid case.
// Empty {} means any rank/type
inline void input_check(const ov::Node* node,
                        size_t idx,
                        const std::string_view input_name,
                        std::initializer_list<ov::Rank>&& allowed_ranks,
                        const std::vector<ov::element::Type>& allowed_types) {
    using namespace ov;
    using namespace ov::util;
    using namespace ov::element;

    const auto& rank = node->get_input_partial_shape(idx).rank();
    const auto& tp = node->get_input_element_type(idx);

    auto rank_check = [&](const Rank& rnk) {
        return rnk.is_dynamic() || empty(allowed_ranks) || is_rank_compatible_any_of(rnk.get_length(), allowed_ranks);
    };

    auto type_check = [&](const Type& t) {
        return t.is_dynamic() || allowed_types.empty() ||
               std::find(allowed_types.begin(), allowed_types.end(), t) != allowed_types.end();
    };

    NODE_VALIDATION_CHECK(node,
                          rank_check(rank),
                          "Rank of `",
                          input_name,
                          "` input should be in [dynamic, ",
                          join(allowed_ranks),
                          "] list, but it is ",
                          rank,
                          ".");

    NODE_VALIDATION_CHECK(node,
                          type_check(tp),
                          "Element type of `",
                          input_name,
                          "` input should be in [dynamic, ",
                          join(allowed_types),
                          "] list, but it is ",
                          tp,
                          ".");
}

std::vector<ov::element::Type> get_real_types() {
    std::vector<ov::element::Type> real_types;
    for (const auto& type : ov::element::Type::get_known_types()) {
        if (type->is_real()) {
            real_types.push_back(*type);
        }
    }
    return real_types;
}

}  // namespace

namespace ov {
namespace op {
PagedAttentionExtension::PagedAttentionExtension(const ov::OutputVector& args) : ov::op::Op(args) {
    constructor_validate_and_infer_types();
}

PagedAttentionExtension::PagedAttentionExtension(const Output<Node>& query,
                                                 const Output<Node>& key,
                                                 const Output<Node>& value,
                                                 const Output<Node>& key_cache,
                                                 const Output<Node>& value_cache,
                                                 const Output<Node>& past_lens,
                                                 const Output<Node>& subsequence_begins,
                                                 const Output<Node>& block_indices,
                                                 const Output<Node>& block_indices_begins,
                                                 const Output<Node>& scale,
                                                 const Output<Node>& sliding_window,
                                                 const Output<Node>& alibi_slopes,
                                                 const Output<Node>& max_context_len)
    : Op({query,
          key,
          value,
          key_cache,
          value_cache,
          past_lens,
          subsequence_begins,
          block_indices,
          block_indices_begins,
          scale,
          sliding_window,
          alibi_slopes,
          max_context_len}) {
    constructor_validate_and_infer_types();
}

PagedAttentionExtension::PagedAttentionExtension(const Output<Node>& query,
                                                 const Output<Node>& key,
                                                 const Output<Node>& value,
                                                 const Output<Node>& key_cache,
                                                 const Output<Node>& value_cache,
                                                 const Output<Node>& past_lens,
                                                 const Output<Node>& subsequence_begins,
                                                 const Output<Node>& block_indices,
                                                 const Output<Node>& block_indices_begins,
                                                 const Output<Node>& scale,
                                                 const Output<Node>& sliding_window,
                                                 const Output<Node>& alibi_slopes,
                                                 const Output<Node>& max_context_len,
                                                 const Output<Node>& rotated_block_indices,
                                                 const Output<Node>& rotation_deltas,
                                                 const Output<Node>& rotation_trig_lut)
    : Op({query,
          key,
          value,
          key_cache,
          value_cache,
          past_lens,
          subsequence_begins,
          block_indices,
          block_indices_begins,
          scale,
          sliding_window,
          alibi_slopes,
          max_context_len,
          rotated_block_indices,
          rotation_deltas,
          rotation_trig_lut}) {
    constructor_validate_and_infer_types();
}

void PagedAttentionExtension::validate_and_infer_types() {
    OV_OP_SCOPE(PagedAttentionExtension_validate_and_infer_types);

    NODE_VALIDATION_CHECK(this,
                          get_input_size() == 20,
                          "PagedAttensionExtension expects 20 inputs, but it has ",
                          get_input_size());

    // format: Node*, input_idx, name, {rank_list}, {type_list}
    input_check(this, 0, "query", {2}, {});
    input_check(this, 1, "key", {2}, {});
    input_check(this, 2, "value", {2}, {});
    input_check(this, 3, "key_cache", {4}, {});
    input_check(this, 4, "value_cache", {4}, {});
    input_check(this, 5, "past_lens", {1}, {element::i32});
    input_check(this, 6, "subsequence_begins", {1}, {element::i32});
    input_check(this, 7, "block_indices", {1}, {element::i32});
    input_check(this, 8, "block_indices_begins", {1}, {element::i32});
    input_check(this, 9, "scale", {0, 1}, get_real_types());
    input_check(this, 10, "sliding_window", {0}, {element::i32});
    input_check(this, 11, "alibi_slopes", {1}, get_real_types());
    input_check(this, 12, "max_context_len", {0}, {element::i32});
    input_check(this, 13, "score_aggregation_window", {0, 1}, {element::i32});
    input_check(this, 14, "rotated_block_indices", {1}, {element::i32});
    input_check(this, 15, "rotation_deltas", {1, 2}, {element::i32});
    input_check(this, 16, "rotation_trig_lut", {1, 2}, {element::f16, element::f32});
    input_check(this, 17, "xattention_threshold", {1}, {element::f16, element::f32});
    input_check(this, 18, "xattention_block_size", {0}, {element::i32});
    input_check(this, 19, "xattention_stride", {0}, {element::i32});

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);
    const auto output_shapes = shape_infer(this, input_shapes);

    set_output_type(0, get_input_element_type(0), output_shapes[0]);
    set_output_type(1, get_input_element_type(0), output_shapes[1]);
}

std::shared_ptr<ov::Node> PagedAttentionExtension::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OV_OP_SCOPE(PagedAttentionExtension_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<PagedAttentionExtension>(new_args);
}

const ov::element::Type PagedAttentionExtension::get_out_type(int index) const {
    OPENVINO_ASSERT(index < 2, "Output index should be 0 or 1, but got " + std::to_string(index));
    return m_output_type[index];
}

void PagedAttentionExtension::set_out_type(int index, const ov::element::Type& output_type) {
    OPENVINO_ASSERT(index < 2, "Output index should be 0 or 1, but got " + std::to_string(index));
    m_output_type[index] = output_type;
}
}  // namespace op
}  // namespace ov
