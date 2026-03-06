// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_delete(const NodeContext& context) {
    // aten::Delete(t[](a!) self, int idx) -> ()
    // Treated as a no-op in static graphs. mutate_input keeps the mutation
    // tracking consistent so subsequent reads see the correct list mapping.
    num_inputs_check(context, 2, 2);
    context.mutate_input(0, context.get_input(0));
    return {};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
