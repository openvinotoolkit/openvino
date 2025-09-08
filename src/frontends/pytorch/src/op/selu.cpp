// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/selu.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_selu(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    auto x = context.get_input(0);
    auto alpha = context.mark_node(v0::Constant::create(element::f64, Shape{}, {1.6732632423543772848170429916717}));
    auto lambda = context.mark_node(v0::Constant::create(element::f64, Shape{}, {1.0507009873554804934193349852946}));
    alpha = context.mark_node(std::make_shared<v1::ConvertLike>(alpha, x));
    lambda = context.mark_node(std::make_shared<v1::ConvertLike>(lambda, x));
    return {context.mark_node(std::make_shared<v0::Selu>(x, alpha, lambda))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov