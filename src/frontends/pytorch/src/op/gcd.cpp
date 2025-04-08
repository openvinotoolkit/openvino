// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/mod.hpp"
#include "openvino/op/not_equal.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_logical_or.hpp"
#include "openvino/op/select.hpp"
#include "openvino/openvino.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_gcd(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    Output<Node> x;
    Output<Node> y;
    std::tie(x, y) = get_inputs_with_promoted_types(context, 0, 1);
    auto zero_i32 = ov::op::v0::Constant::create(element::i32, Shape{}, {0});

    auto trip_count = std::make_shared<v0::Constant>(element::i32, Shape{}, 1000);
    auto exec_condition = std::make_shared<v0::Constant>(element::boolean, Shape{}, true);

    auto loop = std::make_shared<op::v5::Loop>(trip_count, exec_condition);

    auto x_input = std::make_shared<v0::Parameter>(x.get_element_type(), x.get_partial_shape());
    auto y_input = std::make_shared<v0::Parameter>(y.get_element_type(), y.get_partial_shape());

    x_input->set_element_type(x.get_element_type());
    y_input->set_element_type(y.get_element_type());
    auto zero = std::make_shared<ov::op::v1::ConvertLike>(zero_i32, x_input);
    auto condition = std::make_shared<v1::NotEqual>(y_input, zero);
    auto mod = std::make_shared<v1::Mod>(x_input, y_input);
    auto new_x = std::make_shared<v1::Select>(condition, y_input, x_input);
    auto new_y = std::make_shared<v1::Select>(condition, mod, zero);

    auto reduced_condition = std::make_shared<v1::ReduceLogicalOr>(condition, zero);

    auto body =
        std::make_shared<Model>(OutputVector{new_x, new_y, reduced_condition}, ParameterVector{x_input, y_input});
    loop->set_function(body);

    loop->set_special_body_ports({-1, 2});

    loop->set_merged_input(x_input, x, new_x);
    loop->set_merged_input(y_input, y, new_y);

    auto gcd_output = loop->get_iter_value(new_x, -1);
    auto gcd_node = gcd_output.get_node_shared_ptr();

    auto marked_gcd_node = context.mark_node(gcd_node);
    return {marked_gcd_node};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
