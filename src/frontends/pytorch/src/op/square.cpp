// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset8.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_square(NodeContext& context) {
    auto input_0 = context.get_input(0);
    auto const_2 = context.mark_node(opset8::Constant::create(input_0.get_element_type(), Shape{1}, {2}));
    return {context.mark_node(std::make_shared<opset8::Power>(input_0, const_2))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov