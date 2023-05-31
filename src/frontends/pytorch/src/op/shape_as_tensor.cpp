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

using namespace ov::op;

OutputVector translate_shape_as_tensor(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    auto input = context.get_input(0);
    auto shape = context.mark_node(std::make_shared<opset10::ShapeOf>(input));
    return {shape};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov