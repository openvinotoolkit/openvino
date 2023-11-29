// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "common_op_table.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/atan.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/subtract.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_Atan2_op(const NodeContext& node) {
    default_op_checks(node, 2, {"Atan2"});
    auto y = node.get_input(0);
    auto x = node.get_input(1);

    // handle the first condition : x>0
    auto div_y_x = make_shared<v1::Divide>(y, x);
    auto atan = make_shared<v0::Atan>(div_y_x);
    auto const_zero = create_same_type_const_scalar<int32_t>(x, 0);
    auto result = atan->output(0);

    // handle the second condition : x<0 && y>=0
    auto const_pi = create_same_type_const_scalar<double>(x, std::atan(1.0) * 4);
    auto is_x_negative = make_shared<v1::Less>(x, const_zero);
    auto y_non_negative = make_shared<v1::GreaterEqual>(y, const_zero);
    auto cond1 = make_shared<v1::LogicalAnd>(is_x_negative, y_non_negative);
    auto atan_y_x_plus_pi = make_shared<v1::Add>(atan, const_pi);
    result = make_shared<v1::Select>(cond1, atan_y_x_plus_pi, result);

    // handle the third consition : x<0 && y<0
    auto is_y_negative = make_shared<v1::Less>(y, const_zero);
    auto cond2 = make_shared<v1::LogicalAnd>(is_x_negative, is_y_negative);
    auto atan_y_x_minus_pi = make_shared<v1::Subtract>(atan, const_pi);
    result = make_shared<v1::Select>(cond2, atan_y_x_minus_pi, result);

    // handle the fourth condition : x=0 && y>0
    auto is_x_zero = make_shared<v1::Equal>(x, const_zero);
    auto is_y_positive = make_shared<v1::Greater>(y, const_zero);
    auto cond3 = make_shared<v1::LogicalAnd>(is_x_zero, is_y_positive);
    auto const_two = create_same_type_const_scalar<int32_t>(x, 2);
    auto pi_div_two = make_shared<v1::Divide>(const_pi, const_two);
    result = make_shared<v1::Select>(cond3, pi_div_two, result);

    // handle the fifth condition : x=0 && y<0
    auto cond4 = make_shared<v1::LogicalAnd>(is_x_zero, is_y_negative);
    auto const_minus_two = create_same_type_const_scalar<int32_t>(x, -2);
    auto pi_div_minus_two = make_shared<v1::Divide>(const_pi, const_minus_two);
    result = make_shared<v1::Select>(cond4, pi_div_two, result);

    set_node_name(node.get_name(), result.get_node_shared_ptr());
    return {result};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov

+
