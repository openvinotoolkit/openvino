// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/frontend/paddle/visibility.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs elu(const NodeContext& node) {
    auto data = node.get_input("X");
    auto alpha = node.get_attribute<float>("alpha", 1.0);
    const auto& elu_node = std::make_shared<default_opset::Elu>(data, alpha);
    return node.default_single_output_mapping({elu_node}, {"Out"});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
