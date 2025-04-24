// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/multiply.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_neg(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    auto x = context.get_input(0);
    auto const_neg_1 = context.mark_node(v0::Constant::create(element::i32, Shape{}, {-1}));
    auto cast = context.mark_node(std::make_shared<v1::ConvertLike>(const_neg_1, x));
    return {context.mark_node(std::make_shared<v1::Multiply>(x, cast))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov