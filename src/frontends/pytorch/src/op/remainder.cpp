// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset10.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_remainder(NodeContext& context) {
    auto x = context.get_input(0);
    auto y = context.get_input(1);
    auto div = context.mark_node(std::make_shared<opset10::Divide>(x, y, true));
    auto floor = context.mark_node(std::make_shared<opset10::Floor>(div));
    auto quo = context.mark_node(std::make_shared<opset10::Multiply>(floor, y));
    return {context.mark_node(std::make_shared<opset10::Subtract>(x, quo))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov