// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/roll.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_roll(const NodeContext& context) {
    num_inputs_check(context, 2, 3);
    const auto data = context.get_input(0);
    const auto shifts = get_input_concat_if_list(context, 1);
    Output<Node> axes;
    bool on_flattened = context.input_is_none(2);
    if (!on_flattened) {
        axes = context.get_input(2);
        const auto& shifts_pshape = shifts.get_partial_shape();
        const auto& axes_pshape = axes.get_partial_shape();
        on_flattened = !axes_pshape.compatible(shifts_pshape);
    }
    if (on_flattened) {
        const auto const_minus_1 = v0::Constant::create(element::i32, Shape{1}, {-1});
        const auto axis_0 = v0::Constant::create(element::i32, Shape{1}, {0});
        const auto flat = std::make_shared<v1::Reshape>(data, const_minus_1, false);
        const auto roll = std::make_shared<v7::Roll>(flat, shifts, axis_0);
        const auto shape_of_data = std::make_shared<v3::ShapeOf>(data, element::i32);
        const auto reshape = std::make_shared<v1::Reshape>(roll, shape_of_data, false);
        context.mark_nodes({const_minus_1, flat, roll, shape_of_data, reshape});
        return {reshape};
    }
    return {context.mark_node(std::make_shared<v7::Roll>(data, shifts, axes))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
