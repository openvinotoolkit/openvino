// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "itt.hpp"
#include "transformations/op_conversions/convert_divide.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

NGRAPH_RTTI_DEFINITION(ov::pass::ConvertDivide, "ConvertDivide", 0);

ov::pass::ConvertDivide::ConvertDivide() {
    MATCHER_SCOPE(ConvertDivide);
    auto div = ov::pattern::wrap_type<ov::opset1::Divide>();

    ov::matcher_pass_callback callback = [](pattern::Matcher& m) {
        auto div = std::dynamic_pointer_cast<ov::opset1::Divide> (m.get_match_root());
        // We can not apply this transformation in case with integer input data type
        if (!div || div->input(0).get_element_type().is_integral()) {
            return false;
        }

        auto pow = std::make_shared<ov::opset1::Power>(div->input(1).get_source_output(),
                                                           op::Constant::create(div->get_input_element_type(1), Shape{}, {-1}));

        auto mul = std::make_shared<ov::opset1::Multiply>(div->input(0).get_source_output(), pow);

        mul->set_friendly_name(div->get_friendly_name());
        ov::copy_runtime_info(div, {pow, mul});
        ov::replace_node(div, mul);
        return true;
    };

    auto m = std::make_shared<ov::pattern::Matcher>(div, matcher_name);
    this->register_matcher(m, callback);
}
