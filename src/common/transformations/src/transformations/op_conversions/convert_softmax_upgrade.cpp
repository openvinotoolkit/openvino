// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_softmax_upgrade.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>

#include "itt.hpp"

ngraph::pass::ConvertSoftMax1ToSoftMax8::ConvertSoftMax1ToSoftMax8() {
    MATCHER_SCOPE(ConvertSoftMax1ToSoftMax8);

    auto softmax_v1_pattern = pattern::wrap_type<opset1::Softmax>();

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto softmax_v1_node = std::dynamic_pointer_cast<opset1::Softmax>(m.get_match_root());
        if (!softmax_v1_node)
            return false;

        auto axis = static_cast<int64_t>(softmax_v1_node->get_axis());
        auto softmax_v8_node = std::make_shared<opset8::Softmax>(softmax_v1_node->input_value(0), axis);
        softmax_v8_node->set_friendly_name(softmax_v1_node->get_friendly_name());
        copy_runtime_info(softmax_v1_node, softmax_v8_node);
        replace_node(softmax_v1_node, softmax_v8_node);

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(softmax_v1_pattern, matcher_name);
    register_matcher(m, callback);
}
