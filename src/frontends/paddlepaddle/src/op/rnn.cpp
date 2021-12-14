// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paddlepaddle_frontend/node_context.hpp"

#include "openvino/opsets/opset6.hpp"
#include "paddlepaddle_frontend/utility.hpp"

namespace ov {
namespace frontend {
namespace pdpd {
namespace op {
NamedOutputs lstm(const NodeContext& node);
NamedOutputs rnn(const NodeContext& node) {
    auto mode = node.get_attribute<std::string>("mode");
    PDPD_ASSERT(mode == "LSTM",
                "[Paddle Frontend]RNN Only Supports LSTM Ops Conversion now, don't "
                "support " +
                    mode);
    return lstm(node);
}

}  // namespace op
}  // namespace pdpd
}  // namespace frontend
}  // namespace ov
