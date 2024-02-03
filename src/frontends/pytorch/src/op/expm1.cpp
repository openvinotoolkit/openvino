// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/opsets/opset7.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;


OutputVector translate_expm1(const NodeContext& context) {
    num_inputs_check(context, 1, false);
    // "aten::expm1(Tensor self) -> Tensor"

    auto input = context.mark_node(std::make_shared<v0::Convert>(context.get_input(0), element::f32));

    auto exp = context.mark_node(std::make_shared<opset7::Exp>(input));
    auto const_1 = v0::Constant::create(element::f32, Shape{}, {1});
    auto expm1 = context.mark_node(std::make_shared<v1::Subtract>(exp, const_1));
    return {expm1};
}

} // namespace op
} // namespace pytorch
} // namespace frontend
} // namespace ov
