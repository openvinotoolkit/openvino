// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_mean(NodeContext& context) {
    num_inputs_check(context, 3, 4);
    auto x = context.get_input(0);
    auto y = context.get_input(1);
    auto keep_dims = context.const_input<bool>(2);
    FRONT_END_OP_CONVERSION_CHECK(context.input_is_none(3),
                                  "Only False is supported for input with index 3 for aten::mean");
    return {context.mark_node(std::make_shared<ov::op::v1::ReduceMean>(x, y, keep_dims))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov