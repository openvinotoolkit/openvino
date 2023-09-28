// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
using namespace default_opset;
NamedOutputs partial_concat(const NodeContext& node) {
    auto datas = node.get_ng_inputs("X");
    auto start_index = node.get_attribute<int>("start_index");
    auto length = node.get_attribute<int>("length");

    size_t end_index;
    if (length < 0) {
        end_index = datas[0].get_shape()[1];
    } else {
        end_index = start_index + length;
    }

    auto start_node = std::make_shared<Constant>(element::i32, Shape{1}, start_index);
    auto end_node = std::make_shared<Constant>(element::i32, Shape{1}, end_index);
    auto step_node = std::make_shared<Constant>(element::i32, Shape{1}, 1);
    auto axis_node = std::make_shared<Constant>(element::i32, Shape{1}, 1);

    NodeVector outputs;
    for (auto in : datas) {
        auto out = std::make_shared<Slice>(in, start_node, end_node, step_node, axis_node);
        auto casted = std::make_shared<Convert>(out, element::f32);
        outputs.push_back(casted);
    }

    Output<Node> output_node;
    output_node = std::make_shared<Concat>(outputs, 1);

    return node.default_single_output_mapping({std::make_shared<Convert>(output_node, element::f32)}, {"Out"});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
