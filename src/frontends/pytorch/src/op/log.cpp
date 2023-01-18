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

OutputVector translate_log(NodeContext& context) {
    auto x = context.get_input(0);
    if (x.get_element_type().is_integral()) {
        x = context.mark_node(std::make_shared<opset10::Convert>(x, element::f32));
    }
    auto log = context.mark_node(std::make_shared<opset10::Log>(x));
    return {log};
};

OutputVector translate_log2(NodeContext& context) {
    auto x = context.get_input(0);
    auto two = context.mark_node(opset10::Constant::create(element::f32, Shape{}, {2}));
    if (x.get_element_type().is_integral()) {
        x = context.mark_node(std::make_shared<opset10::Convert>(x, element::f32));
    } else {
        two = context.mark_node(std::make_shared<opset10::ConvertLike>(two, x));
    }
    auto log2 = context.mark_node(std::make_shared<opset10::Log>(two));
    auto log = context.mark_node(std::make_shared<opset10::Log>(x));
    auto res = context.mark_node(std::make_shared<opset10::Divide>(log, log2));
    return {res};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov