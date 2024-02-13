// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "fuse_trunc_divide.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/xor.hpp"
#include "openvino/op/constant.hpp"


#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov {
namespace intel_cpu {

FuseTruncDivide::FuseTruncDivide() {
    std::function<bool(Output<Node>)> out_is_integral = [=](Output<Node> output) -> bool {
        return output.get_element_type().is_integral_number();
    };

    auto input_1 = pass::pattern::any_input(out_is_integral);
    auto input_2 = pass::pattern::any_input(out_is_integral);

    auto zero_const = pass::pattern::wrap_type<op::v0::Constant>(out_is_integral);
    auto minus_one_const = pass::pattern::wrap_type<op::v0::Constant>(out_is_integral);

    auto x_less_cond = pass::pattern::wrap_type<op::v1::Less>({input_1, zero_const});
    auto y_less_cond = pass::pattern::wrap_type<op::v1::Less>({input_2, zero_const});
    auto xor_cond = pass::pattern::wrap_type<op::v1::LogicalXor>({x_less_cond, y_less_cond});

    auto div = pass::pattern::wrap_type<op::v1::Divide>({input_1, input_2});

    auto mod_xy = pass::pattern::wrap_type<op::v1::Mod>({input_1, input_2});
    auto cond_mod = pass::pattern::wrap_type<op::v1::NotEqual>({mod_xy, zero_const});

    auto cond = pass::pattern::wrap_type<op::v1::LogicalAnd>({cond_mod, xor_cond});
    auto reminder = pass::pattern::wrap_type<op::v1::Select>({cond, minus_one_const, zero_const});
    auto trunc_div_pattern = pass::pattern::wrap_type<op::v1::Add>({div, reminder});

    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(trunc_div_pattern, "FuseTruncDivide"),
         [=](ov::pass::pattern::Matcher& m) {
            if (!m.get_match_root())
                return false;

            const auto& pattern_to_output = m.get_pattern_map();
            auto old_div = pattern_to_output.at(div);
            auto truc_div = pattern_to_output.at(trunc_div_pattern);

            const auto zero_const_val = as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(zero_const));
            const auto minus_one_const_val = as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(minus_one_const));
            // todo: check if value is really zero and minus one
            // for (auto val : minus_one_const_val->get_vector<float16>()){ }

            auto new_divide = std::make_shared<ov::opset1::Divide>(old_div->input_value(0), old_div->input_value(1));
            new_divide->set_friendly_name(truc_div->get_friendly_name());
            ov::copy_runtime_info(truc_div, new_divide);
            ov::replace_node(truc_div, new_divide);
            return true;
         });
}

}   // namespace intel_cpu
}   // namespace ov
