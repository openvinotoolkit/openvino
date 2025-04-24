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
NamedOutputs linspace(const NodeContext& node) {
    auto start = node.get_input("Start");
    auto stop = node.get_input("Stop");
    auto num = node.get_input("Num");
    auto dtype = node.get_attribute<ov::element::Type>("dtype", element::f32);

    start = std::make_shared<default_opset::Convert>(start, element::f32);
    stop = std::make_shared<default_opset::Convert>(stop, element::f32);

    // compute step value, i.e. distance between neighbor values of the result
    Output<Node> step = std::make_shared<default_opset::Subtract>(stop, start);  //[-1]
    auto const_one = std::make_shared<default_opset::Constant>(element::i32, Shape{}, 1);
    Output<Node> num_minus_one = std::make_shared<default_opset::Subtract>(num, const_one);   //[3]
    auto num_none_zero = std::make_shared<default_opset::Greater>(num_minus_one, const_one);  //[ture]
    num_minus_one = std::make_shared<default_opset::Select>(num_none_zero, num_minus_one, const_one);

    num_minus_one = std::make_shared<default_opset::Convert>(num_minus_one, element::f32);
    step = std::make_shared<default_opset::Divide>(step, num_minus_one);  //[-1/3]

    // generate a range of numbers [0, 1, ..., num)
    auto const_zero = std::make_shared<default_opset::Constant>(element::i32, Shape{}, 0);
    auto const_num = std::make_shared<default_opset::Squeeze>(num);
    auto range0_n = std::make_shared<default_opset::Range>(const_zero, const_num, const_one, element::f32);

    // compute the result
    Output<Node> linspace = std::make_shared<default_opset::Multiply>(range0_n, step);
    auto result = std::make_shared<default_opset::Add>(linspace, start);
    if (dtype == element::i32) {
        return node.default_single_output_mapping({std::make_shared<default_opset::Convert>(result, element::i32)},
                                                  {"Out"});
    } else if (dtype == element::i64) {
        return node.default_single_output_mapping({std::make_shared<default_opset::Convert>(result, element::i64)},
                                                  {"Out"});
    } else {
        return node.default_single_output_mapping({result}, {"Out"});
    }
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
