// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/elu.hpp"
#include "openvino/op/multiply.hpp"
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

    alpha = context.mark_node(std::make_shared<v1::ConvertLike>(alpha, x));
    auto divide_node = context.mark_node(std::make_shared<v1::Divide>(x, alpha));
    auto elu_node = context.mark_node(std::make_shared<v0::Elu>(divide_node, 1.));

    auto elu = context.mark_node(std::make_shared<v1::Multiply>(alpha, elu_node));
    return {elu};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov