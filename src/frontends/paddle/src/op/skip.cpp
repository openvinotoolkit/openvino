// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov::frontend::paddle::op {

NamedOutputs skip(const NodeContext& node) {
    const auto x = node.get_ng_inputs("X");
    return NamedOutputs({{"Out", x}});
}

}  // namespace ov::frontend::paddle::op
