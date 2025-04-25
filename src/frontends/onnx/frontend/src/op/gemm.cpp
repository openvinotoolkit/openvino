// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "utils/reshape.hpp"
using namespace ov::op;
using ov::Shape;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector gemm(const ov::frontend::onnx::Node& node) {
    ov::OutputVector inputs{node.get_ov_inputs()};
    ov::Output<ov::Node> input_a = inputs.at(0);
    ov::Output<ov::Node> input_b = inputs.at(1);
    ov::Output<ov::Node> input_c;

    if (inputs.size() == 3) {
        input_c = inputs.at(2);
    } else {
        input_c = v0::Constant::create(input_b.get_element_type(), ov::Shape{}, {0});
    }

    const auto alpha = node.get_attribute_value<float>("alpha", 1);
    const auto beta_node = node.get_attribute_as_constant<float>("beta", 1, input_c.get_element_type());

    const bool trans_a = node.get_attribute_value<int64_t>("transA", 0);
    const bool trans_b = node.get_attribute_value<int64_t>("transB", 0);

    if (trans_a) {
        input_a = ov::op::util::transpose(input_a);
    }

    if (trans_b) {
        input_b = ov::op::util::transpose(input_b);
    }

    input_a = ov::op::util::flatten(input_a, 1);
    input_b = ov::op::util::flatten(input_b, 1);

    std::shared_ptr<ov::Node> matmul_node = std::make_shared<v0::MatMul>(input_a, input_b);

    if (alpha != 1) {
        const auto alpha_node = v0::Constant::create(input_b.get_element_type(), ov::Shape{}, {alpha});
        matmul_node = std::make_shared<v1::Multiply>(matmul_node, alpha_node);
    }

    auto beta_times_input_c = std::make_shared<v1::Multiply>(beta_node, input_c);

    return ov::OutputVector{std::make_shared<v1::Add>(matmul_node, beta_times_input_c)};
}

ONNX_OP("Gemm", OPSET_RANGE(1, 5), ai_onnx::opset_1::gemm);
}  // namespace opset_1

namespace opset_6 {
ov::OutputVector gemm(const ov::frontend::onnx::Node& node) {
    ov::OutputVector inputs{node.get_ov_inputs()};
    ov::Output<ov::Node> input_a = inputs.at(0);
    ov::Output<ov::Node> input_b = inputs.at(1);
    ov::Output<ov::Node> input_c;

    if (inputs.size() == 3) {
        input_c = inputs.at(2);
    } else {
        input_c = v0::Constant::create(input_b.get_element_type(), ov::Shape{}, {0});
    }

    const auto alpha_node = node.get_attribute_as_constant<float>("alpha", 1, input_b.get_element_type());
    const auto beta_node = node.get_attribute_as_constant<float>("beta", 1, input_c.get_element_type());

    const bool trans_a = node.get_attribute_value<int64_t>("transA", 0);
    const bool trans_b = node.get_attribute_value<int64_t>("transB", 0);

    const auto matmul_node = std::make_shared<v0::MatMul>(input_a, input_b, trans_a, trans_b);
    const auto matmul_times_alpha = std::make_shared<v1::Multiply>(matmul_node, alpha_node);

    const auto beta_times_input_c = std::make_shared<v1::Multiply>(beta_node, input_c);
    const std::string onnx_name = !node.get_name().empty() ? node.get_name() : node.output(0);
    matmul_node->set_friendly_name(onnx_name + "/WithoutBiases");
    return {std::make_shared<v1::Add>(matmul_times_alpha, beta_times_input_c)};
}

ONNX_OP("Gemm", OPSET_SINCE(6), ai_onnx::opset_6::gemm);
}  // namespace opset_6
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
