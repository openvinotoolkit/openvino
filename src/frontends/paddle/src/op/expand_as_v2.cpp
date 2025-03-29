// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs expand_as_v2(const NodeContext& node) {
    using namespace default_opset;
    auto x = node.get_input("X");
    Output<Node> shape_expected_node;
    if (node.has_input("Y")) {
        shape_expected_node = std::make_shared<ShapeOf>(node.get_input("Y"), element::i32);
    } else {
        std::vector<int32_t> shape_expected;
        if (node.has_attribute("target_shape")) {
            shape_expected = node.get_attribute<std::vector<int32_t>>("target_shape");
        } else {
            throw std::runtime_error("expand: has no target_shape attribute");
        }
        shape_expected_node = Constant::create(element::i32, {shape_expected.size()}, shape_expected);
    }
    return node.default_single_output_mapping({std::make_shared<Broadcast>(x, shape_expected_node)}, {"Out"});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
