// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fusedgemm.hpp"

#include <memory>

#include "onnx_import/core/null_node.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/prelu.hpp"
#include "openvino/op/relu.hpp"

using namespace ov::op;

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector fusedgemm(const Node& node) {
    OutputVector inputs{node.get_ng_inputs()};
    auto num_inputs = inputs.size();
    FRONT_END_GENERAL_CHECK(num_inputs == 2 || num_inputs == 3,
                            "FusedGemm takes 2/3 inputs. Provided " + std::to_string(num_inputs));

    Output<ov::Node> input_a = inputs.at(0);
    Output<ov::Node> input_b = inputs.at(1);
    Output<ov::Node> input_c;

    if (num_inputs == 3 && !ov::op::util::is_null(inputs[2])) {
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
    const auto gemm_res = std::make_shared<v1::Add>(matmul_times_alpha, beta_times_input_c);

    const auto activation_type = node.get_attribute_value<std::string>("activation", "Relu");
    if (activation_type == "LeakyRelu") {
        double activation_alpha = node.get_attribute_value<double>("activation_alpha", 0.01);
        std::shared_ptr<ov::Node> activation_alpha_node =
            v0::Constant::create(input_c.get_element_type(), Shape{1}, {activation_alpha});
        return {std::make_shared<v0::PRelu>(gemm_res, activation_alpha_node)};
    }
    return {std::make_shared<v0::Relu>(gemm_res)};
}

}  // namespace set_1

}  // namespace op

}  // namespace  onnx_import

}  // namespace  ngraph
