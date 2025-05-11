// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "slice_ops.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs slice(const NodeContext& node) {
    return slice_op(node, false);
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
