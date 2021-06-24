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

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertDivide, "ConvertDivide", 0);

ngraph::pass::ConvertDivide::ConvertDivide() {
    MATCHER_SCOPE(ConvertDivide);
    auto div = ngraph::pattern::wrap_type<ngraph::opset1::Divide>();

    ngraph::matcher_pass_callback callback = [](pattern::Matcher& m) {
        auto div = std::dynamic_pointer_cast<ngraph::opset1::Divide> (m.get_match_root());
        // We can not apply this transformation in case with integer input data type
        if (!div || div->input(0).get_element_type().is_integral()) {
            return false;
        }

        auto pow = std::make_shared<ngraph::opset1::Power>(div->input(1).get_source_output(),
                                                           op::Constant::create(div->get_input_element_type(1), Shape{}, {-1}));

        auto mul = std::make_shared<ngraph::opset1::Multiply>(div->input(0).get_source_output(), pow);

        mul->set_friendly_name(div->get_friendly_name());
        ngraph::copy_runtime_info(div, {pow, mul});
        ngraph::replace_node(div, mul);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(div, matcher_name);
    this->register_matcher(m, callback);
}
