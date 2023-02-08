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

OutputVector translate_rsub(NodeContext& context) {
    auto self = context.get_input(0);
    auto other = context.get_input(1);
    auto alpha = context.get_input(2);
    align_eltwise_input_types(context, self, other);
    // reverse aten::sub other - self * alpha
    auto alpha_casted = context.mark_node(std::make_shared<opset10::ConvertLike>(alpha, self));
    auto alpha_mul = context.mark_node(std::make_shared<opset10::Multiply>(self, alpha_casted));
    return {context.mark_node(std::make_shared<opset10::Subtract>(other, alpha_mul))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov