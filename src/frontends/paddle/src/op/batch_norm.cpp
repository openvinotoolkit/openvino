// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/opsets/opset6.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs batch_norm(const NodeContext& node) {
    auto data = node.get_input("X");
    auto gamma = node.get_input("Scale");
    auto beta = node.get_input("Bias");
    auto mean = node.get_input("Mean");
    auto variance = node.get_input("Variance");
    auto data_layout = node.get_attribute<std::string>("data_layout");

    PADDLE_OP_CHECK(node, (data_layout == "NCHW" || data_layout == "NHWC"), "Not supported input data layout!");
    if (data_layout == "NCHW") {
        return node.default_single_output_mapping(
            {std::make_shared<ov::opset6::BatchNormInference>(data,
                                                              gamma,
                                                              beta,
                                                              mean,
                                                              variance,
                                                              node.get_attribute<float>("epsilon"))},
            {"Y"});
    } else {
        auto input_order = ov::opset6::Constant::create(ov::element::i64, {4}, {0, 3, 1, 2});
        auto data_nchw = std::make_shared<ov::opset6::Transpose>(data, input_order);
        auto node_batch_norm = std::make_shared<ov::opset6::BatchNormInference>(data_nchw,
                                                                                gamma,
                                                                                beta,
                                                                                mean,
                                                                                variance,
                                                                                node.get_attribute<float>("epsilon"));
        auto output_order = ov::opset6::Constant::create(ov::element::i64, {4}, {0, 2, 3, 1});
        return node.default_single_output_mapping(
            {std::make_shared<ov::opset6::Transpose>(node_batch_norm, output_order)},
            {"Y"});
    }
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
