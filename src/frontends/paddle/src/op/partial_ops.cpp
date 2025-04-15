// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "partial_ops.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {

NamedOutputs partial_sum(const NodeContext& node) {
    return partial_ops(node, "sum");
}

NamedOutputs partial_concat(const NodeContext& node) {
    return partial_ops(node, "concat");
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
