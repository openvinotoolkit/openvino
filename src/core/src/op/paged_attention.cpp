// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_attention.hpp"

#include "dimension_util.hpp"
#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/op.hpp"

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

    auto rank_check = [&](const Rank& rank) {
        return rank.is_dynamic() || empty(allowed_ranks) || is_rank_compatible_any_of(rank.get_length(), allowed_ranks);
    };

    auto type_check = [&](const Type& type) {
        auto it = std::find(allowed_types.begin(), allowed_types.end(), tp);
        return type.is_dynamic() || allowed_types.empty() || it != allowed_types.end();
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
                                                 const Output<Node>& max_context_len,
                                                 const Output<Node>& free_block_indices,
                                                 const Output<Node>& max_blocks)
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
          free_block_indices,
          max_blocks}) {
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
                                                 const Output<Node>& rotation_trig_lut,
                                                 const Output<Node>& free_block_indices,
                                                 const Output<Node>& max_blocks)
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
          rotation_trig_lut,
          free_block_indices,
          max_blocks}) {
    constructor_validate_and_infer_types();
}

void PagedAttentionExtension::validate_and_infer_types() {
    OV_OP_SCOPE(PagedAttentionExtension_validate_and_infer_types);

    NODE_VALIDATION_CHECK(this,
                          get_input_size() == 15 || get_input_size() == 18,
                          "PagedAttensionExtension expects 15 or 18 inputs, but it has ",
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

    if (get_input_size() == 15) {
        input_check(this, 13, "free_block_indices", {1}, {element::i32});
        input_check(this, 14, "max_blocks", {1}, {element::i32});
    } else {
        input_check(this, 13, "rotated_block_indices", {1}, {element::i32});
        input_check(this, 14, "rotation_deltas", {2}, {element::i32});
        input_check(this, 15, "rotation_trig_lut", {2}, {element::f16, element::f32});
        input_check(this, 16, "free_block_indices", {1}, {element::i32});
        input_check(this, 17, "max_blocks", {1}, {element::i32});
    }

    // value head_size may be not same with key
    auto out_ps = get_input_partial_shape(0);
    const auto& key_ps = get_input_partial_shape(1);
    const auto& value_ps = get_input_partial_shape(2);
    const auto& block_ps = get_input_partial_shape(7);
    if (out_ps.rank().is_static()) {
        if (key_ps.rank().is_static() && value_ps.rank().is_static() && key_ps[1].is_static()) {
            // The dim of out_ps[1] should be `num_heads * v_head_size`, it can be got from:
            // because:
            //   q: query_ps[1] = num_heads * head_size
            //   k: key_ps[1] = num_kv_heads * head_size
            //   v: value_ps[1] = num_kv_heads * v_head_size
            // therefore:
            //   q * v / k = (num_heads * head_size) * (num_kv_heads * v_head_size) /
            //               (num_kv_heads * head_size) = num_heads * v_head_size
            out_ps[1] = out_ps[1] * value_ps[1] / key_ps[1].get_length();
            NODE_VALIDATION_CHECK(this,
                                  !ov::util::dim::is_empty(out_ps[1]),
                                  "The last dimension of output should not be empty.");
        } else {
            out_ps[1] = Dimension::dynamic();
        }
    }

    if (m_output_type[0].is_dynamic()) {
        set_output_type(0, get_input_element_type(0), out_ps);
    } else {
        set_output_type(0, m_output_type[0], out_ps);
    }

    if (m_output_type[1].is_dynamic()) {
        set_output_type(1, get_input_element_type(0), {Dimension::dynamic()});
    } else {
        set_output_type(1, m_output_type[1], {Dimension::dynamic()});
    }

    set_output_type(2, m_output_type[2], {block_ps});
    set_output_type(3, m_output_type[3], {block_ps});
    set_output_type(3, m_output_type[4], {block_ps});
}

std::shared_ptr<ov::Node> PagedAttentionExtension::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OV_OP_SCOPE(PagedAttentionExtension_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<PagedAttentionExtension>(new_args);
}

void PagedAttentionExtension::set_out_type(int index, const ov::element::Type& output_type) {
    OPENVINO_ASSERT(index < 5, "Output index should be 0, 1, 2, 3, or 4, but got ", std::to_string(index));
    m_output_type[index] = output_type;
}

const ov::element::Type PagedAttentionExtension::get_out_type(int index) const {
    OPENVINO_ASSERT(index < 5, "Output index should be 0, 1, 2, 3, or 4, but got ", std::to_string(index));
    return m_output_type[index];
}

}  // namespace op
}  // namespace ov
