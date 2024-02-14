// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "transformations/op_conversions/decompose_integer_floordiv.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/logical_xor.hpp"
#include "openvino/op/mod.hpp"
#include "openvino/op/not_equal.hpp"
#include "openvino/op/select.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov {
namespace pass {

DecomposeIntegerFloorDivide::DecomposeIntegerFloorDivide() {
    auto div_pattern = ov::pass::pattern::wrap_type<ov::op::v1::Divide>();
    auto floor_div_pattern = ov::pass::pattern::wrap_type<ov::op::v0::Floor>({div_pattern});

    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(floor_div_pattern, "DecomposeIntegerFloorDivide"),
                     [](ov::pass::pattern::Matcher& m) {
                         auto floor_div = std::dynamic_pointer_cast<ov::op::v0::Floor>(m.get_match_root());
                         if (!floor_div)
                             return false;

                         auto old_div = std::dynamic_pointer_cast<ov::op::v1::Divide>(
                             floor_div->input_value(0).get_node_shared_ptr());
                         if (!old_div)
                             return false;

                         auto out_type = floor_div->output(0).get_element_type();
                         if (!out_type.is_integral_number() && out_type.is_signed())  // todo: check if is needed
                             return false;

                         auto x = old_div->input_value(0);
                         auto y = old_div->input_value(1);

                         auto zero_const = std::make_shared<op::v0::Constant>(out_type, Shape{}, 0);
                         auto minus_one_const = std::make_shared<op::v0::Constant>(out_type, Shape{}, -1);

                         // when integer inputs have different signs remainder should be taken into account
                         // res = x / y; if x > 0 and y > 0
                         // res = x / y - 1; if (x < 0 XOR y < 0) and (x mod y != 0)
                         auto x_less_cond = std::make_shared<op::v1::Less>(x, zero_const);
                         auto y_less_cond = std::make_shared<op::v1::Less>(y, zero_const);
                         auto xor_cond = std::make_shared<op::v1::LogicalXor>(x_less_cond, y_less_cond);

                         auto div = std::make_shared<op::v1::Divide>(x, y, false);
                         auto mod_xy = std::make_shared<op::v1::Mod>(x, y);

                         auto cond_mod = std::make_shared<op::v1::NotEqual>(mod_xy, zero_const);
                         auto cond = std::make_shared<op::v1::LogicalAnd>(cond_mod, xor_cond);
                         auto reminder = std::make_shared<op::v1::Select>(cond, minus_one_const, zero_const);
                         auto new_int_floor_div = std::make_shared<op::v1::Add>(div, reminder);

                         new_int_floor_div->set_friendly_name(floor_div->get_friendly_name());
                         std::vector<std::shared_ptr<ov::Node>> new_nodes = {zero_const,
                                                                             minus_one_const,
                                                                             x_less_cond,
                                                                             y_less_cond,
                                                                             xor_cond,
                                                                             div,
                                                                             mod_xy,
                                                                             cond_mod,
                                                                             cond,
                                                                             reminder,
                                                                             new_int_floor_div};
                         ov::copy_runtime_info(floor_div, new_nodes);
                         ov::replace_node(floor_div, new_int_floor_div);
                         return true;
                     });
}
}  // namespace pass
}  // namespace ov
