// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <climits>

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset8.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_addcmul(NodeContext& context) {
    const auto eltwise_mult = std::make_shared<opset8::Multiply>(context.get_input(1), context.get_input(2));
    const auto elem_type = context.get_input(3).get_element_type();
    const auto value = context.const_input<float>(3);
    const auto const_value = opset8::Constant::create(elem_type, Shape{}, {value});

    const auto scalar_mult = std::make_shared<opset8::Multiply>(eltwise_mult, const_value);
    context.mark_nodes({eltwise_mult, const_value, scalar_mult});
    return {context.mark_node(std::make_shared<opset8::Add>(context.get_input(0), scalar_mult))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
