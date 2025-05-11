// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs meshgrid(const NodeContext& node) {
    auto inputs = node.get_ng_inputs("X");
    OutputVector dims;
    for (const auto& input : inputs) {
        const auto& rank = input.get_partial_shape().rank();
        if (rank.is_static()) {
            PADDLE_OP_CHECK(node, rank.get_length() == 1, "meshgrid each input rank should be 1.");
        }
        const auto shape = std::make_shared<default_opset::ShapeOf>(input);
        dims.push_back(shape);
    }
    const auto out_shape = std::make_shared<default_opset::Concat>(dims, 0);
    OutputVector outs;
    for (size_t i = 0; i < inputs.size(); i++) {
        const auto& input = inputs[i];
        const auto out =
            std::make_shared<default_opset::Broadcast>(input,
                                                       out_shape,
                                                       default_opset::Constant::create(element::i32, {1}, {i}));
        outs.push_back(out);
    }

    NamedOutputs named_outputs;
    named_outputs["Out"] = outs;
    return named_outputs;
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
