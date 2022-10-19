// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/com.microsoft/fused_conv.hpp"
#include "op/conv.hpp"
#include "exceptions.hpp"

#include <memory>
#include <vector>

#include "default_opset.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector fused_conv(const Node& node) {
    const auto conv_res = conv(node).at(0);
    const auto activation_type = node.get_attribute_value<std::string>("activation");
    const auto activation_params = node.get_attribute_value<std::vector<float>>("activation_params", {});
    if (activation_type == "LeakyRelu") {
        CHECK_VALID_NODE(node, activation_params.size() == 1, "activation_alpha attribute of LeakyRelu activation function was not provided");
        const auto activation_alpha_node = default_opset::Constant::create(element::f32, Shape{}, activation_params);
        return {std::make_shared<default_opset::PRelu>(conv_res, activation_alpha_node)};

    } else if (activation_type == "Relu") {
        return {std::make_shared<default_opset::Relu>(conv_res)};

    } else if (activation_type == "HardSigmoid") {
        CHECK_VALID_NODE(node, activation_params.size() == 2, "alpha and beta attribute of HardSigmoid activation function were not provided");

        const auto alpha = default_opset::Constant::create<float>(element::f32, Shape{}, {activation_params[0]});
        const auto beta = default_opset::Constant::create<float>(element::f32, Shape{}, {activation_params[1]});
        return {std::make_shared<default_opset::HardSigmoid>(conv_res, alpha, beta)};
    }



    return {conv_res};
}

}  // namespace set_1

}  // namespace op

}  // namespace  onnx_import

}  // namespace  ngraph
