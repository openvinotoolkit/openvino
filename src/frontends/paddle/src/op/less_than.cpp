// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "elementwise_ops.hpp"

namespace ov::frontend::paddle::op {
NamedOutputs less_than(const NodeContext& node) {
    return elementwise_ops<default_opset::Less>(node);
}
}  // namespace ov::frontend::paddle::op
