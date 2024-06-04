// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {

NamedOutputs skip(const NodeContext& node) {
    const auto x = node.get_ng_inputs("X");
    return NamedOutputs({{"Out", x}});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
