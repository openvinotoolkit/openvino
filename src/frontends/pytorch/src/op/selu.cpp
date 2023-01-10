// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset8.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_selu(NodeContext& context) {
    auto x = context.get_input(0);
    auto alpha = context.mark_node(opset8::Constant::create(element::f64, Shape{}, {1.6732632423543772848170429916717}));
    auto lambda = context.mark_node(opset8::Constant::create(element::f64, Shape{}, {1.0507009873554804934193349852946}));
    auto x_f64 = context.mark_node(std::make_shared<opset8::Convert>(x, element::f64));
    auto res = context.mark_node(std::make_shared<opset8::Selu>(x, alpha, lambda));
    return {context.mark_node(std::make_shared<opset8::ConvertLike>(res, x)};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov