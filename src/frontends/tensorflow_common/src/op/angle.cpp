// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "common_op_table.hpp"
#include "helper_ops/complex_type_mark.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/atan.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/gather.hpp"
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

OutputVector translate_angle_op(const NodeContext& node) {
    default_op_checks(node, 1, {"Angle"}, true);
    auto complex = node.get_input(0);
    auto result_type = node.get_attribute<ov::element::Type>("Tout");

    auto complex_type_mark = as_type_ptr<ComplexTypeMark>(complex.get_node_shared_ptr());

    TENSORFLOW_OP_VALIDATION(
        node,
        complex_type_mark,
        "[TensorFlow Frontend] inconsistent model: Angle operation expects complex type tensor on input");

    complex = complex_type_mark->input_value(0);
    auto real_index = make_shared<v0::Constant>(element::i32, Shape{}, 0);
    auto imag_index = make_shared<v0::Constant>(element::i32, Shape{}, 1);
    auto gather_axis = make_shared<v0::Constant>(element::i32, Shape{1}, -1);

    auto x = make_shared<v8::Gather>(complex, real_index, gather_axis)->output(0);
    auto y = make_shared<v8::Gather>(complex, imag_index, gather_axis)->output(0);

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

    // handle the third condition : x<0 && y<0
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
    auto result_changed_type = make_shared<v0::Convert>(result, result_type)->output(0);

    set_node_name(node.get_name(), result_changed_type.get_node_shared_ptr());
    return {result_changed_type};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
