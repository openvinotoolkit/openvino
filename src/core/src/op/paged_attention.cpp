// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_attention.hpp"

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
                          get_input_size() == 13,
                          "PagedAttensionExtension expects 13 inputs, but it has ",
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

    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
    set_output_type(1, get_input_element_type(0), {Dimension::dynamic()});
}

std::shared_ptr<ov::Node> PagedAttentionExtension::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    return std::make_shared<PagedAttentionExtension>(new_args);
}

}  // namespace op
}  // namespace ov
