// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov::frontend::paddle::op {
NamedOutputs exp(const NodeContext& node) {
    auto data = node.get_input("X");
    return node.default_single_output_mapping({std::make_shared<default_opset::Exp>(data)}, {"Out"});
}
}  // namespace ov::frontend::paddle::op
