// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_tolist(const NodeContext& context) {
    // prim::tolist converts a tensor to a list.
    // works with tensors and the conversion to Python lists happens at runtime.
    num_inputs_check(context, 1, 1);
    return {context.get_input(0)};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
