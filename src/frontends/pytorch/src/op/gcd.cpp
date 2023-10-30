// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/mod.hpp"
#include "openvino/op/not_equal.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/loop.hpp"

#include "utils.hpp"
#include <memory>
#include <ngraph/function.hpp>
#include <openvino/openvino.hpp>

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;



OutputVector translate_gcd(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto x = context.get_input(0).get_node_shared_ptr();
    auto y = context.get_input(1).get_node_shared_ptr();

    auto zero = std::make_shared<v0::Constant>(x->get_element_type(), Shape{}, 0);

    auto trip_count = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, 1000);
    auto exec_condition = std::make_shared<ov::op::v0::Constant>(ov::element::boolean, ov::Shape{}, true);

    auto loop = std::make_shared<op::v5::Loop>(trip_count, exec_condition);
    loop->set_special_body_ports({-1, 0});

    auto x_input = std::make_shared<v0::Parameter>(x->get_element_type(), x->get_shape());
    auto y_input = std::make_shared<v0::Parameter>(y->get_element_type(), y->get_shape());

    auto condition = std::make_shared<v1::NotEqual>(y_input, zero);
    auto mod = std::make_shared<v1::Mod>(x_input, y_input);
    auto new_x = std::make_shared<v1::Select>(condition, y_input, x_input);
    auto new_y = std::make_shared<v1::Select>(condition, mod, zero);

    auto body = std::make_shared<ngraph::Function>(OutputVector{new_x, new_y}, ParameterVector{x_input, y_input});
    loop->set_function(body);

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
