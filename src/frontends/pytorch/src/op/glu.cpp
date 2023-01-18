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

OutputVector translate_glu(NodeContext& context) {
    auto x = context.get_input(0);
    auto dim = context.input_is_none(1) ? context.mark_node(opset10::Constant::create(element::i64, Shape{}, {-1}))
                                        : context.get_input(1);
    auto split = context.mark_node(std::make_shared<opset10::Split>(x, dim, 2));
    auto first = split->output(0);
    auto second = split->output(1);
    auto sigmoid = context.mark_node(std::make_shared<opset10::Sigmoid>(second));
    return {context.mark_node(std::make_shared<opset10::Multiply>(first, sigmoid))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov