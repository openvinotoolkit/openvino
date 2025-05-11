// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs swish(const NodeContext& node) {
    const auto x = node.get_input("X");
    const float beta = node.get_attribute<float>("beta", 1.0f);
    const auto beta_node = default_opset::Constant::create(element::f32, Shape{}, {beta});

    return node.default_single_output_mapping({std::make_shared<default_opset::Swish>(x, beta_node)}, {"Out"});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
