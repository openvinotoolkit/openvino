// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reduce_ops.hpp"

namespace ov::frontend::paddle::op {
NamedOutputs reduce_any(const NodeContext& node_context) {
    return reduce_ops<default_opset::ReduceLogicalOr>(node_context);
}

}  // namespace ov::frontend::paddle::op
