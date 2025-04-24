// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs where_index(const NodeContext& node) {
    const auto condition = node.get_input("Condition");
    const auto perm = default_opset::Constant::create(element::i64, Shape{2}, {1, 0});
    const auto out = std::make_shared<default_opset::NonZero>(condition, element::i64);
    return node.default_single_output_mapping({std::make_shared<default_opset::Transpose>(out, perm)}, {"Out"});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
