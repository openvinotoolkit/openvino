// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_reshape_as(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto input_tensor = context.get_input(0);
    auto shape_tesnor = context.get_input(1);
    auto desired_shape = context.mark_node(std::make_shared<ov::op::v3::ShapeOf>(shape_tesnor, element::i32));
    return {context.mark_node(std::make_shared<ov::op::v1::Reshape>(input_tensor, desired_shape, false))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov