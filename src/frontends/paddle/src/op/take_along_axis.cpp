// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/opsets/opset6.hpp"
namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs take_along_axis(const NodeContext& node) {
    auto input = node.get_input("Input");
    auto index = node.get_input("Index");
    auto axis = node.get_attribute<int32_t>("Axis");
    return node.default_single_output_mapping({std::make_shared<default_opset::GatherElements>(input, index, axis)},
                                              {"Result"});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
