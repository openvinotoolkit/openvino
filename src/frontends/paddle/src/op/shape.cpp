// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/opsets/opset6.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs shape(const NodeContext& node) {
    auto data = node.get_input("Input");
    auto shape_node = std::make_shared<ov::opset6::ShapeOf>(data, element::i32);
    return node.default_single_output_mapping({shape_node}, {"Out"});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
