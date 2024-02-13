// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "fuse_floor_divide.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/xor.hpp"
#include "openvino/op/constant.hpp"

#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"

using namespace ov::pass;

namespace ov {
namespace intel_cpu {

FuseFloorDivide::FuseFloorDivide() {
    std::function<bool(Output<Node>)> is_out_int = [=](Output<Node> output) -> bool {
        return output.get_element_type().is_integral();
    };

    auto matches_to_value = [](int value) -> std::function<bool(Output<Node>)>{
        return [=](Output<Node> const_out) -> bool {
            if (!const_out.get_element_type().is_integral()) {
                return false;
                std::cout << "didn't match" << std::endl;
            }
            auto values = as_type_ptr<ov::op::v0::Constant>(const_out.get_node_shared_ptr())->cast_vector<int>();
            if (!std::all_of(values.begin(), values.end(), [&](const int val) {return val == value;})) {
                return false;
                std::cout << "didn't match" << std::endl;
            }
            return true;
        };
    };

    auto input_1 = pattern::any_input(is_out_int);
    auto input_2 = pattern::any_input(is_out_int);

    // each zeros can be a different object
    std::vector<std::shared_ptr<ov::Node>> zeros_pattern;
    for (int i = 0; i < 4; i++)
        zeros_pattern.push_back(pattern::wrap_type<op::v0::Constant>(matches_to_value(0)));

    auto minus_one_const = pattern::wrap_type<op::v0::Constant>(matches_to_value(-1));

    // Less condition itself might be constant folded to a boolean Const
    auto x_less = pattern::wrap_type<op::v1::Less>({input_1, zeros_pattern[0]});
    auto x_less_cond = std::make_shared<pattern::op::Or>(OutputVector{x_less, pattern::wrap_type<op::v0::Constant>()});
    auto y_less = pattern::wrap_type<op::v1::Less>({input_2, zeros_pattern[1]});
    auto y_less_cond = std::make_shared<pattern::op::Or>(OutputVector{y_less, pattern::wrap_type<op::v0::Constant>()});

    auto xor_cond = pattern::wrap_type<op::v1::LogicalXor>({x_less_cond, y_less_cond});

    auto div = pattern::wrap_type<op::v1::Divide>({input_1, input_2});
    auto mod_xy = pattern::wrap_type<op::v1::Mod>({input_1, input_2});
    auto cond_mod = pattern::wrap_type<op::v1::NotEqual>({mod_xy, zeros_pattern[2]});

    auto cond = pattern::wrap_type<op::v1::LogicalAnd>({cond_mod, xor_cond});
    auto reminder = pattern::wrap_type<op::v1::Select>({cond, minus_one_const, zeros_pattern[3]});
    auto floor_div_pattern = pattern::wrap_type<op::v1::Add>({div, reminder});

    register_matcher(std::make_shared<pattern::Matcher>(floor_div_pattern, "FuseFloorDivide"),
         [=](pattern::Matcher& m) {
            if (!m.get_match_root())
                return false;
            std::cout << "MATCHED" << std::endl;
            auto pattern_map = m.get_pattern_map();

            auto old_div = pattern_map.at(div);
            auto truc_div = pattern_map.at(floor_div_pattern);

            // At least on the Less(x, 0), Less(y, 0) should be non-constant.
            // pattern can match if both x_less and y_less are Constant, but in that case
            // it's not FloorDiv subgraph, we should skip it.
            if (pattern_map.find(x_less) == pattern_map.end() && pattern_map.find(y_less) == pattern_map.end()) {
                std::cout << "need to exit" << std::endl;
                return false;
            }

            auto new_divide = std::make_shared<ov::opset1::Divide>(old_div->input_value(0), old_div->input_value(1));
            new_divide->set_friendly_name(truc_div->get_friendly_name());
            ov::copy_runtime_info(truc_div, new_divide);
            ov::replace_node(truc_div, new_divide);
            return true;
         });
}

}   // namespace intel_cpu
}   // namespace ov
