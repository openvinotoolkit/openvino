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

OutputVector translate_mean(NodeContext& context) {
    auto x = context.get_input(0);
    auto y = context.get_input(1);
    auto keep_dims = context.const_input<bool>(2);
    OV_FRONTEND_REQUIRE(context.input_is_none(3));
    return {context.mark_node(std::make_shared<opset8::ReduceMean>(x, y, keep_dims))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov