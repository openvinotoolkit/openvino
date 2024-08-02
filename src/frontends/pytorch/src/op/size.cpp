// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/shape_of.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_size(const NodeContext& context) {
    num_inputs_check(context, 1, 2);
    auto shape = context.mark_node(std::make_shared<v3::ShapeOf>(context.get_input(0), element::i64));
    if (context.input_is_none(1)) {
        return shape->outputs();
    } else {
        auto axis_0 = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
        return {context.mark_node(std::make_shared<v8::Gather>(shape, context.get_input(1), axis_0))};
    }
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
