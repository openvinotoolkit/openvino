// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "itt.hpp"
#include "transformations/op_conversions/convert_negative.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

NGRAPH_RTTI_DEFINITION(ov::pass::ConvertNegative, "ConvertNegative", 0);

ov::pass::ConvertNegative::ConvertNegative() {
    MATCHER_SCOPE(ConvertNegative);
    auto neg = ov::pattern::wrap_type<ov::opset1::Negative>();

    ov::matcher_pass_callback callback = [](pattern::Matcher& m) {
        auto neg = std::dynamic_pointer_cast<ov::opset1::Negative> (m.get_match_root());
        if (!neg) {
            return false;
        }

        auto mul = std::make_shared<ov::opset1::Multiply>(neg->input(0).get_source_output(),
                                                              opset1::Constant::create(neg->get_element_type(), Shape{}, {-1}));
        mul->set_friendly_name(neg->get_friendly_name());
        ov::copy_runtime_info(neg, mul);
        ov::replace_node(neg, mul);
        return true;
    };

    auto m = std::make_shared<ov::pattern::Matcher>(neg, matcher_name);
    this->register_matcher(m, callback);
}
