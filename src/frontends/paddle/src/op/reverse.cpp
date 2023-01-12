// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/opsets/opset1.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {

using namespace default_opset;

NamedOutputs reverse(const NodeContext& node) {
    auto x = node.get_input("X");
    // axis is a vector
    auto axis = node.get_attribute<std::vector<int32_t>>("axis");
    // try to keep the axis positive since reverse IR doesn't support negative axis.
    const auto dims = static_cast<int32_t>(x.get_partial_shape().rank().get_length());
    std::for_each(axis.begin(), axis.end(), [&dims](int32_t& value) {
        if (value < 0) {
            value += dims;
        }
    });

    auto axis_node = std::make_shared<Constant>(ngraph::element::i32, Shape{axis.size()}, axis);
    auto reverse_op = std::make_shared<ov::opset1::Reverse>(x, axis_node, ov::opset1::Reverse::Mode::INDEX);
    return node.default_single_output_mapping({reverse_op}, {"Out"});
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
