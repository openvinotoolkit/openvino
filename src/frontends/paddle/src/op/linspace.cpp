// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/frontend/paddle/visibility.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs linspace(const NodeContext& node) {
    auto start = node.get_input("Start");
    auto stop = node.get_input("Stop");
    auto num = node.get_input("Num");

    // compute delta value, i.e. distance between neighbor values of the result
    auto const_one = std::make_shared<default_opset::Constant>(start.get_element_type(), Shape{}, 1);
    Output<Node> num_minus_one = std::make_shared<default_opset::Subtract>(num, const_one);
    num_minus_one = std::make_shared<default_opset::Convert>(num_minus_one, start.get_element_type());
    Output<Node> delta = std::make_shared<default_opset::Subtract>(stop, start);
    delta = std::make_shared<default_opset::Divide>(delta, num_minus_one);

    // generate a range of numbers [0, 1, ..., num)
    auto const_zero = std::make_shared<default_opset::Constant>(start.get_element_type(), Shape{}, 0); 
    auto const_num = std::make_shared<default_opset::Squeeze>(num);
    auto range0_n = std::make_shared<default_opset::Range>(const_zero, const_num, const_one, start.get_element_type());

    // compute the result
    Output<Node> linspace = std::make_shared<default_opset::Multiply>(range0_n, delta);

    return node.default_single_output_mapping({std::make_shared<default_opset::Add>(linspace, start)}, {"Out"});
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
