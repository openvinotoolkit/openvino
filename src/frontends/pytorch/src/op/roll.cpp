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

OutputVector translate_roll(NodeContext& context) {
    const auto data = context.get_input(0);
    const auto shifts = context.get_input(1);
    const auto axes = context.get_input(2);
    const auto shifts_pshape = shifts.get_partial_shape();
    const auto axes_pshape = axes.get_partial_shape();
    const auto match_dims = axes_pshape.compatible(shifts_pshape);
    if (!match_dims) {
        const auto const_minus_1 = opset10::Constant::create(element::i32, Shape{1}, {-1});
        const auto axis_0 = opset10::Constant::create(element::i32, Shape{1}, {0});
        const auto flat = std::make_shared<opset10::Reshape>(data, const_minus_1, false);
        const auto roll = std::make_shared<opset10::Roll>(flat, shifts, axis_0);
        const auto shape_of_data = std::make_shared<opset10::ShapeOf>(data);
        const auto reshape = std::make_shared<opset10::Reshape>(roll, shape_of_data, false);
        context.mark_nodes({const_minus_1, flat, roll, shape_of_data, reshape});
        return {reshape};
    }
    return {context.mark_node(std::make_shared<opset10::Roll>(data, shifts, axes))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
