// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reverse_op.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs reverse(const NodeContext& node) {
    return reverse_op(node);
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov