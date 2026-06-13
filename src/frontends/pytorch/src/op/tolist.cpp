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
    // prim::tolist conceptually converts a tensor to a Python list in PyTorch,
    // but in OpenVINO this is intentionally modeled as a no-op/pass-through:
    // the tensor is forwarded as-is and any actual list conversion is handled
    // at runtime on the Python side. Note: the op table currently maps
    // prim::tolist to op::skip_node, so this translator is not registered/used
    // and is kept here for clarity and potential future use.
    num_inputs_check(context, 1, 1);
    return {context.get_input(0)};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
