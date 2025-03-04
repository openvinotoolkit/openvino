// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_attention.hpp"

#include "dimension_util.hpp"
#include "itt.hpp"
#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v16 {

PagedAttention::PagedAttention(const Output<Node>& query,
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

PagedAttention::PagedAttention(const Output<Node>& query,
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

void PagedAttention::validate_and_infer_types() {
    OV_OP_SCOPE(v16_PagedAttention_validate_and_infer_types);

    NODE_VALIDATION_CHECK(this,
                          get_input_size() == 13 || get_input_size() == 16,
                          "PagedAttensionExtension expects 13 or 16 inputs, but it has ",
                          get_input_size());

    NODE_VALIDATION_CHECK(
        this,
        get_input_partial_shape(0).rank().is_dynamic() || get_input_partial_shape(0).rank().get_length() == 2,
        "Rank of `query` input should be 2, but it is ",
        get_input_partial_shape(0).rank().get_length(),
        ".");
    NODE_VALIDATION_CHECK(
        this,
        get_input_partial_shape(1).rank().is_dynamic() || get_input_partial_shape(1).rank().get_length() == 2,
        "Rank of `key` input should be 2, but it is ",
        get_input_partial_shape(1).rank().get_length(),
        ".");
    NODE_VALIDATION_CHECK(
        this,
        get_input_partial_shape(2).rank().is_dynamic() || get_input_partial_shape(2).rank().get_length() == 2,
        "Rank of `value` input should be 2, but it is ",
        get_input_partial_shape(2).rank().get_length(),
        ".");

    NODE_VALIDATION_CHECK(
        this,
        get_input_partial_shape(3).rank().is_dynamic() || get_input_partial_shape(3).rank().get_length() >= 2,
        "Rank of `key_cache` input should be at least 2, but it is ",
        get_input_partial_shape(3).rank().get_length(),
        ".");
    NODE_VALIDATION_CHECK(
        this,
        get_input_partial_shape(4).rank().is_dynamic() || get_input_partial_shape(4).rank().get_length() >= 2,
        "Rank of `value_cache` input should be at least 2, but it is ",
        get_input_partial_shape(4).rank().get_length(),
        ".");

    NODE_VALIDATION_CHECK(
        this,
        get_input_partial_shape(5).rank().is_dynamic() || get_input_partial_shape(5).rank().get_length() == 1,
        "Rank of `past_lens` input should be 1, but it is ",
        get_input_partial_shape(5).rank().get_length(),
        ".");
    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(5).is_dynamic() || get_input_element_type(5) == element::i32,
                          "Element type of `past_lens` input should be i32, but it is ",
                          get_input_element_type(5),
                          ".");
    NODE_VALIDATION_CHECK(
        this,
        get_input_partial_shape(6).rank().is_dynamic() || get_input_partial_shape(6).rank().get_length() == 1,
        "Rank of `subsequence_begins` input should be 1, but it is ",
        get_input_partial_shape(6).rank().get_length(),
        ".");
    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(6).is_dynamic() || get_input_element_type(6) == element::i32,
                          "Element type of `subsequence_begins` input should be i32, but it is ",
                          get_input_element_type(6),
                          ".");

    NODE_VALIDATION_CHECK(
        this,
        get_input_partial_shape(7).rank().is_dynamic() || get_input_partial_shape(7).rank().get_length() == 1,
        "Rank of `block_indices` input should be 1, but it is ",
        get_input_partial_shape(7).rank().get_length(),
        ".");
    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(7).is_dynamic() || get_input_element_type(7) == element::i32,
                          "Element type of `block_indices` input should be i32, but it is ",
                          get_input_element_type(7),
                          ".");
    NODE_VALIDATION_CHECK(
        this,
        get_input_partial_shape(8).rank().is_dynamic() || get_input_partial_shape(8).rank().get_length() == 1,
        "Rank of `block_indices_begins` input should be 1, but it is ",
        get_input_partial_shape(8).rank().get_length(),
        ".");
    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(8).is_dynamic() || get_input_element_type(8) == element::i32,
                          "Element type of `block_indices_begins` input should be i32, but it is ",
                          get_input_element_type(8),
                          ".");

    NODE_VALIDATION_CHECK(
        this,
        get_input_partial_shape(9).rank().is_dynamic() || get_input_partial_shape(9).rank().get_length() == 0,
        "Input `scale` should be a scalar but it has rank ",
        get_input_partial_shape(9).rank().get_length(),
        ".");
    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(9).is_dynamic() || get_input_element_type(9).is_real(),
                          "Element type of `scale` input should be a floating type, but it is ",
                          get_input_element_type(9),
                          ".");
    NODE_VALIDATION_CHECK(
        this,
        get_input_partial_shape(10).rank().is_dynamic() || get_input_partial_shape(10).rank().get_length() == 0,
        "Input `sliding_window` should be a scalar but it has rank ",
        get_input_partial_shape(10).rank().get_length(),
        ".");
    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(10).is_dynamic() || get_input_element_type(10) == element::i32,
                          "Element type of `sliding_window` input should be i32, but it is ",
                          get_input_element_type(10),
                          ".");

    NODE_VALIDATION_CHECK(
        this,
        get_input_partial_shape(11).rank().is_dynamic() || get_input_partial_shape(11).rank().get_length() == 1,
        "Rank of `alibi_slopes` input should be 1, but it is ",
        get_input_partial_shape(11).rank().get_length(),
        ".");
    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(11).is_dynamic() || get_input_element_type(11).is_real(),
                          "Element type of `alibi_slopes` input should be a floating type, but it is ",
                          get_input_element_type(11),
                          ".");
    NODE_VALIDATION_CHECK(
        this,
        get_input_partial_shape(12).rank().is_dynamic() || get_input_partial_shape(12).rank().get_length() == 0,
        "Input `max_context_len` should be a scalar but it has rank ",
        get_input_partial_shape(12).rank().get_length(),
        ".");
    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(12).is_dynamic() || get_input_element_type(12) == element::i32,
                          "Element type of `max_context_len` input should be i32, but it is ",
                          get_input_element_type(12),
                          ".");

    // value head_size may be not same with key
    auto out_ps = get_input_partial_shape(0);
    const auto& key_ps = get_input_partial_shape(1);
    const auto& value_ps = get_input_partial_shape(2);
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

    set_output_type(0, m_output_type[0], out_ps);
    set_output_type(1, m_output_type[1], {Dimension::dynamic()});
}

std::shared_ptr<ov::Node> PagedAttention::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OV_OP_SCOPE(v16_PagedAttention_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    if (new_args.size() == 13) {
        return std::make_shared<PagedAttention>(new_args.at(0),
                                                new_args.at(1),
                                                new_args.at(2),
                                                new_args.at(3),
                                                new_args.at(4),
                                                new_args.at(5),
                                                new_args.at(6),
                                                new_args.at(7),
                                                new_args.at(8),
                                                new_args.at(9),
                                                new_args.at(10),
                                                new_args.at(11),
                                                new_args.at(12));
    } else if (new_args.size() == 16) {
        return std::make_shared<PagedAttention>(new_args.at(0),
                                                new_args.at(1),
                                                new_args.at(2),
                                                new_args.at(3),
                                                new_args.at(4),
                                                new_args.at(5),
                                                new_args.at(6),
                                                new_args.at(7),
                                                new_args.at(8),
                                                new_args.at(9),
                                                new_args.at(10),
                                                new_args.at(11),
                                                new_args.at(12),
                                                new_args.at(13),
                                                new_args.at(14),
                                                new_args.at(15));
    }
    OPENVINO_ASSERT(false, "PagedAttention requires either 13 or 16 inputs");

}

void PagedAttention::set_out_type(int index, const ov::element::Type& output_type) {
    OPENVINO_ASSERT(index < 2, "Output index should be 0 or 1, but got " + std::to_string(index));
    m_output_type[index] = output_type;
}

const ov::element::Type PagedAttention::get_out_type(int index) const {
    OPENVINO_ASSERT(index < 2, "Output index should be 0 or 1, but got " + std::to_string(index));
    return m_output_type[index];
}

}  // namespace v16
}  // namespace op
}  // namespace ov
