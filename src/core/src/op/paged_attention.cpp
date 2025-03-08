// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_attention.hpp"

#include "dimension_util.hpp"
#include "itt.hpp"
#include "openvino/op/op.hpp"

namespace ov {
namespace op {

PagedAttentionExtension::PagedAttentionExtension(const ov::OutputVector& args) : ov::op::Op(args) {
    constructor_validate_and_infer_types();
}

void PagedAttentionExtension::validate_and_infer_types() {
    OV_OP_SCOPE(PagedAttentionExtension_validate_and_infer_types);

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

    if (get_input_size() == 16) {
        NODE_VALIDATION_CHECK(
            this,
            get_input_partial_shape(13).rank().is_dynamic() || get_input_partial_shape(13).rank().get_length() == 1,
            "Input `rotated_block_indices` should either have rank 1 or be omitted, but it has rank ",
            get_input_partial_shape(13).rank().get_length(),
            ".");
        NODE_VALIDATION_CHECK(this,
                              get_input_element_type(13).is_dynamic() || get_input_element_type(13) == element::i32,
                              "Element type of `rotated_block_indices` input should be i32, but it is ",
                              get_input_element_type(13),
                              ".");
        NODE_VALIDATION_CHECK(
            this,
            get_input_partial_shape(14).rank().is_dynamic() || get_input_partial_shape(14).rank().get_length() == 2,
            "Input `rotation_deltas` should either have rank 2 or be omitted, but it has rank ",
            get_input_partial_shape(14).rank().get_length(),
            ".");
        NODE_VALIDATION_CHECK(this,
                              get_input_element_type(14).is_dynamic() || get_input_element_type(14) == element::i32,
                              "Element type of `rotation_deltas` input should be i32, but it is ",
                              get_input_element_type(14),
                              ".");
        NODE_VALIDATION_CHECK(
            this,
            get_input_partial_shape(15).rank().is_dynamic() || get_input_partial_shape(15).rank().get_length() == 2,
            "Input `rotation_trig_lut` should either have rank 2 or be omitted, but it has rank ",
            get_input_partial_shape(15).rank().get_length(),
            ".");
        NODE_VALIDATION_CHECK(this,
                              get_input_element_type(15).is_dynamic() || get_input_element_type(15) == element::f32 ||
                                  get_input_element_type(15) == element::f16,
                              "Element type of `rotation_trig_lut` input should be f32 or f16, but it is ",
                              get_input_element_type(15),
                              ".");
    }

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
}

std::shared_ptr<ov::Node> PagedAttentionExtension::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    return std::make_shared<PagedAttentionExtension>(new_args);
}

void PagedAttentionExtension::set_out_type(int index, const ov::element::Type& output_type) {
    OPENVINO_ASSERT(index < 2, "Output index should be 0 or 1, but got " + std::to_string(index));
    m_output_type[index] = output_type;
}

}  // namespace op
}  // namespace ov
