// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset10.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_sub(NodeContext& context) {
    auto x = context.get_input(0);
    auto y = context.get_input(1);
    // default alpha is 1 so no need to multiply if alpha is not provided
    if (!context.input_is_none(2)) {
        auto alpha = context.get_input(2);
        auto casted_alpha = context.mark_node(std::make_shared<opset10::ConvertLike>(alpha, y));
        y = context.mark_node(std::make_shared<opset10::Multiply>(casted_alpha, y));
    }
    return {context.mark_node(std::make_shared<opset10::Subtract>(x, y))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov