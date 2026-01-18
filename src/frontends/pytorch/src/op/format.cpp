// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_format(const NodeContext& context) {
    // aten::format(str self, ...) -> str
    num_inputs_check(context, 1, context.get_input_size());
    return {context.get_input(0)};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
