// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/power.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_reciprocal(NodeContext& context) {
    num_inputs_check(context, 1, 1);
    auto x = context.get_input(0);
    auto const_neg_1 = context.mark_node(v0::Constant::create(element::i32, Shape{}, {-1}));
    auto cast = context.mark_node(std::make_shared<v1::ConvertLike>(const_neg_1, x));
    auto power = context.mark_node(std::make_shared<v1::Power>(x, cast));
    return {context.mark_node(power)};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov