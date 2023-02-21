// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "reduce_ops.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs mean(const NodeContext& node_context) {
    return reduce_ops<default_opset::ReduceMean>(node_context);
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov

