// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <node_context.hpp>

#include "default_opset.hpp"

namespace ngraph {
namespace frontend {
namespace pdpd {
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
        PDPD_OP_VALIDATION_CHECK(
            node,
            (axis >= -(data_shape.rank().get_length() + 1)) && axis < (data_shape.rank().get_length() + 1),
            "axis range is [-(R+1), R+1)!");

    for (const auto& data : datas) {
        PDPD_OP_VALIDATION_CHECK(node,
                                 data_type == data.get_element_type(),
                                 "stack input tensor must have the same data types!");

        node_datas_reshape.push_back(std::make_shared<Unsqueeze>(data, axis_const));
    }

    return node.default_single_output_mapping({std::make_shared<Concat>(node_datas_reshape, axis)}, {"Y"});
}
}  // namespace op
}  // namespace pdpd
}  // namespace frontend
}  // namespace ngraph