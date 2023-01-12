// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset8.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_reciprocal(NodeContext& context) {
    auto x = context.get_input(0);
    auto const_neg_1 = opset8::Constant::create(element::i32, Shape{}, {-1});
    auto cast = std::make_shared<opset8::ConvertLike>(const_neg_1, x);
    auto power = std::make_shared<opset8::Power>(x, cast);
    return {context.mark_node(power)};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov