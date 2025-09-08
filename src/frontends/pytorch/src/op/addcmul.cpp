// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <climits>

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/multiply.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

namespace {
OutputVector addcmul_common(const NodeContext& context, const Output<Node>& value) {
    const auto eltwise_mult = std::make_shared<v1::Multiply>(context.get_input(1), context.get_input(2));
    const auto converted_value = std::make_shared<v1::ConvertLike>(value, context.get_input(1));
    const auto scalar_mult = std::make_shared<v1::Multiply>(eltwise_mult, converted_value);
    context.mark_nodes({eltwise_mult, converted_value, scalar_mult});
    return {context.mark_node(std::make_shared<v1::Add>(context.get_input(0), scalar_mult))};
};
}  // namespace

OutputVector translate_addcmul(const NodeContext& context) {
    num_inputs_check(context, 4, 4);
    const auto value = context.get_input(3);
    return addcmul_common(context, value);
};

OutputVector translate_addcmul_fx(const NodeContext& context) {
    num_inputs_check(context, 3, 3);
    Output<Node> value = context.mark_node(v0::Constant::create(element::f32, Shape{}, {1}));
    if (context.has_attribute("value")) {
        value = context.get_input("value");
    }
    return addcmul_common(context, value);
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
