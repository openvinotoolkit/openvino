// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_sum(NodeContext& context) {
    num_inputs_check(context, 1, 3);
    bool keep_dims = false;
    ov::Output<ov::Node> axes;
    auto data = context.get_input(0);
    if (context.input_is_none(1)) {
        axes = get_axes_range(context, 0);
    } else {
        axes = context.get_input(1);
    }
    if (!context.input_is_none(2)) {
        keep_dims = context.const_input<bool>(2);
    }

    return {context.mark_node(std::make_shared<ov::op::v1::ReduceSum>(data, axes, keep_dims))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov