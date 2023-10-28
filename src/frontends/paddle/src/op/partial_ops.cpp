// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "partial_ops.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {

NamedOutputs elementwise_sum(const NodeContext& node_context) {
    return partial_ops(node, "sum");
}

NamedOutputs elementwise_concat(const NodeContext& node_context) {
    return partial_ops(node, "concat");
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
