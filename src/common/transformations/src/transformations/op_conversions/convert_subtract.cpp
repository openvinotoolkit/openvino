// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_subtract.hpp"

#include <memory>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <openvino/opsets/opset1.hpp>
#include <vector>

#include "itt.hpp"
#include "transformations/rt_info/dequantization_node.hpp"

ov::pass::ConvertSubtract::ConvertSubtract() {
    MATCHER_SCOPE(ConvertSubtract);
    auto sub = ngraph::pattern::wrap_type<ov::opset1::Subtract>();

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto sub = std::dynamic_pointer_cast<ov::opset1::Subtract>(m.get_match_root());
        if (!sub) {
            return false;
        }

        if (ov::is_dequantization_node(sub)) {
            return false;
        }

        if (!sub->get_input_element_type(1).is_signed()) {
            return false;
        }

        auto neg = std::make_shared<ov::opset1::Multiply>(
            sub->input(1).get_source_output(),
            opset1::Constant::create(sub->get_input_element_type(1), Shape{}, {-1}));

        auto add = std::make_shared<ov::opset1::Add>(sub->input(0).get_source_output(), neg);

        add->set_friendly_name(sub->get_friendly_name());
        ngraph::copy_runtime_info(sub, {neg, add});
        ngraph::replace_node(sub, add);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(sub, matcher_name);
    this->register_matcher(m, callback);
}
