// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/opsets/opset6.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs range(const NodeContext& node) {
    auto start = node.get_input("Start");
    auto stop = node.get_input("End");
    auto step = node.get_input("Step");
    auto type = node.get_out_port_type("Out");

    const auto axis = ov::opset6::Constant::create(element::i64, Shape{}, {0});
    auto start_scalar = std::make_shared<ov::opset6::Squeeze>(start, axis);
    auto stop_scalar = std::make_shared<ov::opset6::Squeeze>(stop, axis);
    auto step_scalar = std::make_shared<ov::opset6::Squeeze>(step, axis);

    return node.default_single_output_mapping(
        {std::make_shared<ov::opset6::Range>(start_scalar, stop_scalar, step_scalar, type)},
        {"Out"});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
