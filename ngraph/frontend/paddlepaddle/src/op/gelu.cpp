// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset7.hpp>
#include <node_context.hpp>

namespace ngraph {
namespace frontend {
namespace pdpd {
namespace op {
NamedOutputs gelu(const NodeContext& node) {
    auto data = node.get_ng_input("X");
    auto approximate = node.get_attribute<bool>("approximate", false);
    auto mode = approximate ? ngraph::op::GeluApproximationMode::TANH : ngraph::op::GeluApproximationMode::ERF;

    return node.default_single_output_mapping({std::make_shared<opset7::Gelu>(data, mode)}, {"Out"});
}
}  // namespace op
}  // namespace pdpd
}  // namespace frontend
}  // namespace ngraph