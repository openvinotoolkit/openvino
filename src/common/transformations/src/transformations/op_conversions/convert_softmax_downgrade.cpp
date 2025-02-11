// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_softmax_downgrade.hpp"

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

ov::pass::ConvertSoftMax8ToSoftMax1::ConvertSoftMax8ToSoftMax1() {
    MATCHER_SCOPE(ConvertSoftMax8ToSoftMax1);

    auto input = pattern::any_input(pattern::has_static_rank());
    auto softmax_v8_pattern = pattern::wrap_type<ov::op::v8::Softmax>({input});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto softmax_v8_node = ov::as_type_ptr<ov::op::v8::Softmax>(m.get_match_root());
        if (!softmax_v8_node)
            return false;

        auto v8_axis = softmax_v8_node->get_axis();
        auto rank = softmax_v8_node->get_input_partial_shape(0).rank().get_length();
        auto v1_axis = ov::util::try_normalize_axis(v8_axis, rank, *softmax_v8_node);

        auto softmax_v1_node = std::make_shared<ov::op::v1::Softmax>(softmax_v8_node->input_value(0), v1_axis);
        softmax_v1_node->set_friendly_name(softmax_v8_node->get_friendly_name());
        copy_runtime_info(softmax_v8_node, softmax_v1_node);
        replace_node(softmax_v8_node, softmax_v1_node);

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(softmax_v8_pattern, matcher_name);
    register_matcher(m, callback);
}
