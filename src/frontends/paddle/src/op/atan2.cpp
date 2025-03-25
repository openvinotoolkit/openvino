// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/frontend/paddle/node_context.hpp"
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
#include "openvino/opsets/opset6.hpp"
using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace paddle {

template <typename T>
ov::Output<ov::Node> create_same_type_const_scalar(const ov::Output<ov::Node>& same_type_output, const T& value) {
    if (same_type_output.get_element_type().is_static()) {
        return std::make_shared<ov::op::v0::Constant>(same_type_output.get_element_type(), ov::Shape{}, value);
    } else {
        ov::Output<ov::Node> const_res =
            std::make_shared<ov::op::v0::Constant>(ov::element::from<T>(), ov::Shape{}, value);
        const_res = std::make_shared<ov::op::v1::ConvertLike>(const_res, same_type_output);
        return const_res;
    }
}

namespace op {
NamedOutputs atan2(const NodeContext& node) {
    //    default_op_checks(node, 2, {"Atan2"});
    auto y = node.get_input("X1");
    auto x = node.get_input("X2");

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
    NamedOutputs named_outputs;
    named_outputs["Out"] = {result};
    return named_outputs;
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
