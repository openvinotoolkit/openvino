// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/gelu.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/tanh.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_gelu(NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto x = context.get_input(0);
    auto approximate = context.const_input<std::string>(1);
    if (approximate == "none") {
        return {context.mark_node(std::make_shared<ov::op::v7::Gelu>(x))};
    }
    // GELU(x)=0.5∗x∗(1+Tanh(sqrt(2/pi)∗(x+0.044715∗x3)))
    if (approximate == "tanh") {
        auto pi = context.mark_node(ov::op::v0::Constant::create(element::f32, Shape{}, {3.141592653589793}));
        auto kappa = context.mark_node(ov::op::v0::Constant::create(element::f32, Shape{}, {0.044715}));
        auto two = context.mark_node(ov::op::v0::Constant::create(element::f32, Shape{}, {2}));
        auto three = context.mark_node(ov::op::v0::Constant::create(element::f32, Shape{}, {3}));
        auto one = context.mark_node(ov::op::v0::Constant::create(element::f32, Shape{}, {1}));
        auto half = context.mark_node(ov::op::v0::Constant::create(element::f32, Shape{}, {0.5}));
        auto beta = context.mark_node(std::make_shared<ov::op::v1::Divide>(two, pi));
        auto beta_sqrt = context.mark_node(std::make_shared<ov::op::v0::Sqrt>(beta));
        three = context.mark_node(std::make_shared<ov::op::v1::ConvertLike>(three, x));
        auto x3 = context.mark_node(std::make_shared<ov::op::v1::Power>(x, three));
        kappa = context.mark_node(std::make_shared<ov::op::v1::ConvertLike>(kappa, x));
        auto kappa_x3 = context.mark_node(std::make_shared<ov::op::v1::Multiply>(kappa, x3));
        auto x_kappa_x3 = context.mark_node(std::make_shared<ov::op::v1::Add>(x, kappa_x3));
        beta_sqrt = context.mark_node(std::make_shared<ov::op::v1::ConvertLike>(beta_sqrt, x));
        auto inner_prod = context.mark_node(std::make_shared<ov::op::v1::Multiply>(beta_sqrt, x_kappa_x3));
        auto inner_tanh = context.mark_node(std::make_shared<ov::op::v0::Tanh>(inner_prod));
        one = context.mark_node(std::make_shared<ov::op::v1::ConvertLike>(one, x));
        auto inner = context.mark_node(std::make_shared<ov::op::v1::Add>(one, inner_tanh));
        half = context.mark_node(std::make_shared<ov::op::v1::ConvertLike>(half, x));
        auto half_x = context.mark_node(std::make_shared<ov::op::v1::Multiply>(half, x));
        auto res = context.mark_node(std::make_shared<ov::op::v1::Multiply>(half_x, inner));
        return {res};
    }
    FRONT_END_OP_CONVERSION_CHECK(false, "Unsupported approximate for Gelu: ", approximate);
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov