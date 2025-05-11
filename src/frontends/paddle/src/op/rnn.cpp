// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/frontend/paddle/visibility.hpp"
#include "openvino/opsets/opset6.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs lstm(const NodeContext& node);
NamedOutputs rnn(const NodeContext& node) {
    auto mode = node.get_attribute<std::string>("mode");
    PADDLE_OP_CHECK(node,
                    mode == "LSTM",
                    "[Paddle Frontend]RNN Only Supports LSTM Ops Conversion now, don't "
                    "support " +
                        mode);
    return lstm(node);
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
