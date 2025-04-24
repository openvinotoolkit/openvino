// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/frontend/paddle/visibility.hpp"
#include "openvino/opsets/opset6.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs reshape2(const NodeContext& node) {
    auto data = node.get_input("X");
    if (!node.has_input("Shape") && !node.has_input("ShapeTensor")) {
        auto shape_attr = node.get_attribute<std::vector<int32_t>>("shape");
        auto shape_node = ov::opset6::Constant::create(ov::element::i32, {shape_attr.size()}, shape_attr);
        return node.default_single_output_mapping({std::make_shared<ov::opset6::Reshape>(data, shape_node, true)},
                                                  {"Out"});
    } else {
        std::string name = "Shape";
        if (node.has_input("ShapeTensor")) {
            name = "ShapeTensor";
        }

        auto nodes = node.get_ng_inputs(name);
        ov::NodeVector node_vec;
        for (auto& input_node : nodes) {
            if (input_node.get_shape().size() == 0) {
                input_node =
                    std::make_shared<ov::opset6::Unsqueeze>(input_node,
                                                            ov::opset6::Constant::create(ov::element::i64, {1}, {0}));
            }
            auto cast = std::make_shared<ov::opset6::Convert>(input_node, element::i64);
            node_vec.push_back(cast);
        }

        auto shape_node = std::make_shared<ov::opset6::Concat>(node_vec, 0);
        return node.default_single_output_mapping({std::make_shared<ov::opset6::Reshape>(data, shape_node, true)},
                                                  {"Out"});
    }
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
