// Copyright (C) 2018-2025 Intel Corporation
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

OutputVector translate_reciprocal(const NodeContext& context) {
    num_inputs_check(context, 1, 2);
    auto x = get_input_with_floating_type(context, 0);
    auto const_neg_1 = context.mark_node(v0::Constant::create(element::f32, Shape{}, {-1}))->output(0);
    const_neg_1 = context.mark_node(std::make_shared<v1::ConvertLike>(const_neg_1, x));
    auto power = context.mark_node(std::make_shared<v1::Power>(x, const_neg_1));
    return {context.mark_node(power)};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
