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

OutputVector translate_add(NodeContext& context) {
    auto rhs = context.get_input(1);
    if (!context.input_is_none(2)) {
        auto converted_alpha = std::make_shared<opset10::ConvertLike>(context.get_input(2), rhs);
        rhs = std::make_shared<opset10::Multiply>(converted_alpha, rhs);
    }
    return {context.mark_node(std::make_shared<opset10::Add>(context.get_input(0), rhs))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov