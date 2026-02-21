// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_is_nested(const NodeContext& context) {
    // prim::is_nested checks if a tensor is a nested tensor.
    return {context.mark_node(v0::Constant::create(element::boolean, Shape{}, {false}))};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
