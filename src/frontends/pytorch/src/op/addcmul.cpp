// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <climits>

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset10.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_addcmul(NodeContext& context) {
    const auto eltwise_mult = std::make_shared<opset10::Multiply>(context.get_input(1), context.get_input(2));
    const auto value = context.get_input(3);
    const auto converted_value = std::make_shared<opset10::ConvertLike>(value, context.get_input(1));
    const auto scalar_mult = std::make_shared<opset10::Multiply>(eltwise_mult, converted_value);
    context.mark_nodes({eltwise_mult, converted_value, scalar_mult});
    return {context.mark_node(std::make_shared<opset10::Add>(context.get_input(0), scalar_mult))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
