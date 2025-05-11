// Copyright (C) 2018-2025 Intel Corporation
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
        auto dropout_prob =
            ov::opset6::Constant::create(ov::element::f32, {1}, {1 - node.get_attribute<float>("dropout_prob")});
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
