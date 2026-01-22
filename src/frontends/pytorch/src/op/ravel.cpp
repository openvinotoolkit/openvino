// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/reshape.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_ravel(const NodeContext& context) {
    // aten::ravel(input)
    // "Return a contiguous flattened tensor."
    // Equivalent to input.reshape(-1)
    
    num_inputs_check(context, 1, 1);
    auto input = context.get_input(0);
    
    auto shape = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-1}));
    auto reshape = context.mark_node(std::make_shared<v1::Reshape>(input, shape, false));
    
    return {reshape};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
