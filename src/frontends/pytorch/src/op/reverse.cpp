// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/reverse.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"

#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_reverse(const NodeContext& context) {
    // aten::reverse(Tensor input, int[] dims=None)
    num_inputs_check(context, 1, 2);

    auto data = context.get_input(0);
    ov::Output<ov::Node> axes;
    if(!context.input_is_none(1)) {
        axes = context.get_input(1);
    } else {
        axes = v0::Constant::create(element::i32, Shape{1}, {0});
    }
    auto reverse_node = std::make_shared<ov::op::v1::Reverse>(data, axes, ov::op::v1::Reverse::Mode::INDEX);

    return {context.mark_node(reverse_node)};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
