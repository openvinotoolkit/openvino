// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reverse_op.hpp"

namespace ov::frontend::paddle::op {
NamedOutputs flip(const NodeContext& node) {
    return reverse_op(node);
}
}  // namespace ov::frontend::paddle::op
