// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_delete(const NodeContext& context) {
    // aten::Delete(container, key)
    // No-Op for static graph.
    return {};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
