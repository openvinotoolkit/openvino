// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs share_data(const NodeContext& node) {
    auto x = node.get_input("X");
    PADDLE_OP_CHECK(node,
                    x.get_element_type() == node.get_out_port_type("Out"),
                    "Input and output type should be the same");
    NamedOutputs named_outputs;
    named_outputs["Out"] = OutputVector{x};
    return named_outputs;
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
