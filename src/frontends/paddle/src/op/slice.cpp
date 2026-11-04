// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "slice_ops.hpp"

namespace ov::frontend::paddle::op {
NamedOutputs slice(const NodeContext& node) {
    return slice_op(node, false);
}
}  // namespace ov::frontend::paddle::op
