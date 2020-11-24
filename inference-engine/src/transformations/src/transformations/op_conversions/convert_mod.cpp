// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_mod.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertMod, "ConvertMod", 0);

ngraph::pass::ConvertMod::ConvertMod() {
    auto mod = ngraph::pattern::wrap_type<opset1::Mod>();

    ngraph::matcher_pass_callback callback = [this](pattern::Matcher& m) {
        auto mod = std::dynamic_pointer_cast<ngraph::opset1::Mod> (m.get_match_root());
        if (!mod) {
            return false;
        }

        const auto dividend = std::make_shared<opset1::Abs>(mod->input_value(0));
        const auto dividend_sign = std::make_shared<opset1::Sign>(mod->input_value(0));
        const auto dividend_et = dividend->get_element_type();
        const auto divisor = std::make_shared<opset1::Abs>(mod->input_value(1));

        // truncated(a / b)
        auto div = register_new_node<opset1::Divide>(dividend, divisor);
        auto convert_to_i64 = std::make_shared<opset1::Convert>(div, ngraph::element::i64);
        auto convert = std::make_shared<opset1::Convert>(convert_to_i64, dividend_et);
        // truncated(a / b) * b
        auto multiplication = std::make_shared<opset1::Multiply>(convert, divisor);
        // a mod b = a - truncated(a / b) * b
        auto sub = register_new_node<opset1::Subtract>(dividend, multiplication);

        // apply sign of dividend
        auto mul = std::make_shared<opset1::Multiply>(dividend_sign, sub);

        mul->set_friendly_name(mod->get_friendly_name());
        ngraph::copy_runtime_info(mod, {dividend, dividend_sign, divisor, div, convert_to_i64, convert, multiplication, sub, mul});
        ngraph::replace_node(mod, mul);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(mod, "ConvertMod");
    this->register_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
