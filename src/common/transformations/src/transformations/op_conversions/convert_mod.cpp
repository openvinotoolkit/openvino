// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_mod.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/mod.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/sign.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

ov::pass::ConvertMod::ConvertMod() {
    MATCHER_SCOPE(ConvertMod);
    auto mod = ov::pass::pattern::wrap_type<ov::op::v1::Mod>();

    matcher_pass_callback callback = [this](pattern::Matcher& m) {
        auto mod = ov::as_type_ptr<ov::op::v1::Mod>(m.get_match_root());
        if (!mod) {
            return false;
        }

        const auto dividend = std::make_shared<ov::op::v0::Abs>(mod->input_value(0));
        const auto dividend_sign = std::make_shared<ov::op::v0::Sign>(mod->input_value(0));
        const auto dividend_et = dividend->get_element_type();
        const auto divisor = std::make_shared<ov::op::v0::Abs>(mod->input_value(1));

        // truncated(a / b)
        auto div = register_new_node<ov::op::v1::Divide>(dividend, divisor);
        auto convert_to_i64 = std::make_shared<ov::op::v0::Convert>(div, ov::element::i64);
        auto convert = std::make_shared<ov::op::v0::Convert>(convert_to_i64, dividend_et);
        // truncated(a / b) * b
        auto multiplication = std::make_shared<ov::op::v1::Multiply>(convert, divisor);
        // a mod b = a - truncated(a / b) * b
        auto sub = register_new_node<ov::op::v1::Subtract>(dividend, multiplication);

        // apply sign of dividend
        auto mul = std::make_shared<ov::op::v1::Multiply>(dividend_sign, sub);

        mul->set_friendly_name(mod->get_friendly_name());
        ov::copy_runtime_info(
            mod,
            {dividend, dividend_sign, divisor, div, convert_to_i64, convert, multiplication, sub, mul});
        ov::replace_node(mod, mul);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(mod, matcher_name);
    this->register_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
