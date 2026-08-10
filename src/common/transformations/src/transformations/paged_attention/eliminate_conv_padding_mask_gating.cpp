// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/paged_attention/eliminate_conv_padding_mask_gating.hpp"

#include "itt.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

using ov::pass::pattern::wrap_type;

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v8 = ov::op::v8;

namespace {

bool has_static_rank(const ov::Output<ov::Node>& output, const int64_t expected_rank) {
    const auto& rank = output.get_partial_shape().rank();
    return rank.is_static() && rank.get_length() == expected_rank;
}

bool is_attention_mask_slice(ov::Output<ov::Node> output) {
    auto node = output.get_node_shared_ptr();
    if (ov::is_type<v0::Convert>(node)) {
        node = node->input_value(0).get_node_shared_ptr();
    }
    if (ov::is_type<v0::Unsqueeze>(node)) {
        node = node->input_value(0).get_node_shared_ptr();
    }

    const auto slice = ov::as_type_ptr<v8::Slice>(node);
    if (!slice) {
        return false;
    }
    const auto parameter = ov::as_type_ptr<v0::Parameter>(slice->get_input_node_shared_ptr(0));
    return parameter && parameter->output(0).get_names().count("attention_mask");
}

bool is_attention_mask_expression(const ov::Output<ov::Node>& output) {
    if (is_attention_mask_slice(output)) {
        return true;
    }

    const auto add = ov::as_type_ptr<v1::Add>(output.get_node_shared_ptr());
    if (!add) {
        return false;
    }
    for (const auto& add_input : add->input_values()) {
        const auto multiply = ov::as_type_ptr<v1::Multiply>(add_input.get_node_shared_ptr());
        if (!multiply) {
            continue;
        }
        for (const auto& multiply_input : multiply->input_values()) {
            if (is_attention_mask_slice(multiply_input)) {
                return true;
            }
        }
    }
    return false;
}

}  // namespace

namespace ov::pass {

EliminateConvPaddingMaskGating::EliminateConvPaddingMaskGating() {
    MATCHER_SCOPE(EliminateConvPaddingMaskGating);

    auto multiply_pattern = wrap_type<v1::Multiply>();
    ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& matcher) {
        const auto multiply = ov::as_type_ptr<v1::Multiply>(matcher.get_match_root());
        for (size_t hidden_index = 0; hidden_index < 2; ++hidden_index) {
            const auto hidden_states = multiply->input_value(hidden_index);
            const auto mask_expression = multiply->input_value(1 - hidden_index);
            if (has_static_rank(hidden_states, 3) && is_attention_mask_expression(mask_expression)) {
                multiply->output(0).replace(hidden_states);
                return true;
            }
        }
        return false;
    };

    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(multiply_pattern, matcher_name);
    this->register_matcher(matcher, callback);
}

}  // namespace ov::pass
