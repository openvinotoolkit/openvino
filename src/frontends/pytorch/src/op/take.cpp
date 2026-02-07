// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/reshape.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_take(const NodeContext& context) {
    // aten::take(self, index)
    num_inputs_check(context, 2, 2);
    auto self = context.get_input(0);
    auto index = context.get_input(1);
    
    auto minus_one = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-1}));
    auto flattened = context.mark_node(std::make_shared<v1::Reshape>(self, minus_one, false));
    
    auto axis = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {0}));
    auto gather = context.mark_node(std::make_shared<v8::Gather>(flattened, index, axis));
    
    return {gather};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
