// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
using namespace default_opset;
NamedOutputs stack(const NodeContext& node) {
    auto datas = node.get_ng_inputs("X");
    auto axis = node.get_attribute<int32_t>("axis", 0);
    auto data_shape = datas[0].get_partial_shape();
    auto data_type = datas[0].get_element_type();
    OutputVector node_datas_reshape;

    auto axis_const = std::make_shared<Constant>(element::i64, Shape{}, axis);
    if (data_shape.rank().is_static())
        PADDLE_OP_CHECK(node,
                        (axis >= -(data_shape.rank().get_length() + 1)) && axis < (data_shape.rank().get_length() + 1),
                        "axis range is [-(R+1), R+1)!");

    for (const auto& data : datas) {
        PADDLE_OP_CHECK(node,
                        data_type == data.get_element_type(),
                        "stack input tensor must have the same data types!");

        node_datas_reshape.push_back(std::make_shared<Unsqueeze>(data, axis_const));
    }

    return node.default_single_output_mapping({std::make_shared<Concat>(node_datas_reshape, axis)}, {"Y"});
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
