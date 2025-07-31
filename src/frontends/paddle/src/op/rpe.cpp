// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"
// #include "openvino/frontend/paddle/visibility.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {

NamedOutputs rpe_attention_weight(const NodeContext& node) {
    // 取输入
    // auto input_0 = node.get_input("X");
    // auto input_1 = node.get_input("X");
    // auto input_2 = node.get_input("X");
    // auto input_3 = node.get_input("X");

    auto inputs = node.get_all_ng_inputs();
    auto input_0 = inputs[0];
    auto input_1 = inputs[1];
    auto input_2 = inputs[2];
    auto input_3 = inputs[3];

    // 检查输入维度
    const auto input_0_shape = input_0.get_partial_shape();
    const auto input_1_shape = input_1.get_partial_shape();
    const auto input_2_shape = input_2.get_partial_shape();
    const auto input_3_shape = input_3.get_partial_shape();
    FRONT_END_OP_CONVERSION_CHECK(input_0_shape.rank().is_static() && input_0_shape.rank().get_length() == 3,
                                  "Input0 to rpe_attention_weight must be 3D tensor");
    FRONT_END_OP_CONVERSION_CHECK(input_1_shape.rank().is_static() && input_1_shape.rank().get_length() == 3,
                                  "Input1 to rpe_attention_weight must be 3D tensor");
    FRONT_END_OP_CONVERSION_CHECK(input_2_shape.rank().is_static() && input_2_shape.rank().get_length() == 3,
                                  "Input2 to rpe_attention_weight must be 3D tensor");
    FRONT_END_OP_CONVERSION_CHECK(input_3_shape.rank().is_static() && input_3_shape.rank().get_length() == 4,
                                  "Input3 to rpe_attention_weight must be 4D tensor");

    auto ouput = std::make_shared<ov::op::v5::RotRPEAttentionWeightWithIndexComputation>(input_0, input_1, input_2, input_3);

    return node.default_single_output_mapping({ouput}, {"Out"});
}

NamedOutputs rpe_project_value(const NodeContext& node) {
    // 取输入
    // auto input_0 = node.get_input("X");
    // auto input_1 = node.get_input("X");
    // auto input_2 = node.get_input("X");
    // auto input_3 = node.get_input("X");

    auto inputs = node.get_all_ng_inputs();
    auto input_0 = inputs[0];
    auto input_1 = inputs[1];
    auto input_2 = inputs[2];
    auto input_3 = inputs[3];

    // 检查输入维度
    const auto input_0_shape = input_0.get_partial_shape();
    const auto input_1_shape = input_1.get_partial_shape();
    const auto input_2_shape = input_2.get_partial_shape();
    const auto input_3_shape = input_3.get_partial_shape();
    FRONT_END_OP_CONVERSION_CHECK(input_0_shape.rank().is_static() && input_0_shape.rank().get_length() == 3,
                                  "Input0 to rpe_project_value must be 3D tensor");
    FRONT_END_OP_CONVERSION_CHECK(input_1_shape.rank().is_static() && input_1_shape.rank().get_length() == 3,
                                  "Input1 to rpe_project_value must be 3D tensor");
    FRONT_END_OP_CONVERSION_CHECK(input_2_shape.rank().is_static() && input_2_shape.rank().get_length() == 3,
                                  "Input2 to rpe_project_value must be 3D tensor");
    FRONT_END_OP_CONVERSION_CHECK(input_3_shape.rank().is_static() && input_3_shape.rank().get_length() == 4,
                                  "Input3 to rpe_project_value must be 4D tensor");

    auto ouput = std::make_shared<ov::op::v5::RotRPEProjectValueWithIndexComputation>(input_0, input_1, input_2, input_3);

    return node.default_single_output_mapping({ouput}, {"Out"});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
