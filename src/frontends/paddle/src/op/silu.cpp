// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs silu(const NodeContext& node) {
    auto x = node.get_input("X");
    auto neg_x = std::make_shared<default_opset::Negative>(x);
    auto exp_neg_x = std::make_shared<default_opset::Exp>(neg_x);
    auto const_one = default_opset::Constant::create<float>(ov::element::f32, {1}, {1});
    auto denominator = std::make_shared<default_opset::Add>(const_one, exp_neg_x);
    return node.default_single_output_mapping({std::make_shared<default_opset::Divide>(x, denominator)}, {"Out"});
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
