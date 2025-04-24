// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "elementwise_ops.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs greater_than(const NodeContext& node) {
    return elementwise_ops<default_opset::Greater>(node);
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
