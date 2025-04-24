// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs softplus(const NodeContext& node) {
    auto data = node.get_input("X");
    auto beta = node.get_attribute<float>("beta");
    auto threshold = node.get_attribute<float>("threshold");
    float supported_beta = 1.0;
    float supported_threshold = 20.0;
    const float EPSINON = 1e-6f;

    if (!(std::fabs(beta - supported_beta) <= EPSINON) || !(std::fabs(threshold - supported_threshold) <= EPSINON)) {
        PADDLE_OP_CHECK(node, false, "only support beta==1.0 && threshold==20.0");
    }
    return node.default_single_output_mapping({std::make_shared<default_opset::SoftPlus>(data)}, {"Out"});
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
