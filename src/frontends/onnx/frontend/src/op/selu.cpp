// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/selu.hpp"

#include <memory>
#include <vector>

#include "default_opset.hpp"
#include "ngraph/op/constant.hpp"
#include "op/selu.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector selu(const Node& node) {
    auto data = node.get_ng_inputs().at(0);
    auto alpha = node.get_attribute_value<double>("alpha", 1.67326319217681884765625);
    auto gamma = node.get_attribute_value<double>("gamma", 1.05070102214813232421875);

    auto alpha_node = default_opset::Constant::create(data.get_element_type(), Shape{}, {alpha});

    auto gamma_node = default_opset::Constant::create(data.get_element_type(), Shape{}, {gamma});

    return {std::make_shared<default_opset::Selu>(data, alpha_node, gamma_node)};
}

}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
