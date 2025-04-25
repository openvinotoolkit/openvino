// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs unstack(const NodeContext& node) {
    auto data = node.get_input("X");
    auto input_shape = data.get_partial_shape();
    PADDLE_OP_CHECK(node, input_shape.rank().is_static(), "rank of input data should be static");
    auto dim = node.get_attribute<int32_t>("axis", 0);
    if (dim < 0) {
        dim = dim + static_cast<int32_t>(input_shape.rank().get_length());
    }
    auto axis = default_opset::Constant::create(element::i32, {}, {dim});
    auto shape = input_shape.get_shape();
    auto splits = std::make_shared<default_opset::Split>(data, axis, shape.at(dim));
    auto split_outputs = splits->outputs();
    NamedOutputs named_outputs;
    auto out_names = node.get_output_names();
    auto it = std::find(out_names.begin(), out_names.end(), "Y");
    PADDLE_OP_CHECK(node, it != out_names.end(), "Expected output not found");
    for (const auto& split_output : split_outputs) {
        named_outputs[*it].push_back(std::make_shared<default_opset::Squeeze>(split_output, axis));
    }
    return named_outputs;
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
