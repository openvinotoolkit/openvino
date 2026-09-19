// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/opsets/opset6.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs dropout(const NodeContext& node) {
    auto data = node.get_input("X");
    auto dropout_implementation = node.get_attribute<std::string>("dropout_implementation");
    PADDLE_OP_CHECK(node,
                    (dropout_implementation == "downgrade_in_infer" || dropout_implementation == "upscale_in_train"),
                    "Unsupported dropout mode!");
    if (dropout_implementation == "downgrade_in_infer") {
        // The scale must have the same element type as the input, otherwise the Multiply below
        // cannot merge its arguments (e.g. a float16 or float64 input). The element type of the
        // input is not necessarily resolved yet, so the constant is created in f32 and converted
        // like the input, as done in atan2.cpp.
        auto scale =
            ov::opset6::Constant::create(ov::element::f32, {1}, {1 - node.get_attribute<float>("dropout_prob")});
        auto dropout_prob = std::make_shared<ov::opset6::ConvertLike>(scale, data);
        return node.default_single_output_mapping({std::make_shared<ov::opset6::Multiply>(data, dropout_prob)},
                                                  {"Out"});
    } else {
        return node.default_single_output_mapping(data.get_node_shared_ptr(), {"Out"});
    }
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
