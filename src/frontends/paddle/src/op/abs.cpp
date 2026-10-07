// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/opsets/opset6.hpp"

namespace ov::frontend::paddle::op {
NamedOutputs abs(const NodeContext& node) {
    auto data = node.get_input("X");
    return node.default_single_output_mapping({std::make_shared<ov::opset6::Abs>(data)}, {"Out"});
}

}  // namespace ov::frontend::paddle::op
