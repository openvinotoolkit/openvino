// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/elu.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_celu(const NodeContext& context) {
    // aten::celu(%x_copy.1, %self.alpha)
    num_inputs_check(context, 1, 2);
    auto x = context.get_input(0);
    Output<Node> alpha;
    if (context.input_is_none(1)) {
        alpha = context.mark_node(v0::Constant::create(element::f32, Shape{}, {1.}));
    } else {
        alpha = context.get_input(1);
    }

    // CELU(x)=max(0,x)+min(0,a*(exp(x/a)-1))
    auto zero = context.mark_node(v0::Constant::create(element::f32, Shape{}, {0.}));
    zero = context.mark_node(std::make_shared<v1::ConvertLike>(zero, x));
    auto x_max = context.mark_node(std::make_shared<v1::Maximum>(x, zero));

    alpha = context.mark_node(std::make_shared<v1::ConvertLike>(alpha, x));
    auto divide_node = context.mark_node(std::make_shared<v1::Divide>(x, alpha));
    auto exp_node = context.mark_node(std::make_shared<v0::Exp>(divide_node));
    auto one = context.mark_node(v0::Constant::create(element::f32, Shape{}, {1.}));
    one = context.mark_node(std::make_shared<v1::ConvertLike>(one, x));
    auto exp_minus_one = context.mark_node(std::make_shared<v1::Subtract>(exp_node, one));
    auto elu_node = context.mark_node(std::make_shared<v1::Multiply>(alpha, exp_minus_one));
    auto min_node = context.mark_node(std::make_shared<v1::Minimum>(elu_node, zero));

    auto elu = context.mark_node(std::make_shared<v1::Add>(x_max, min_node));
    return {elu};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov