// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/leaky_relu.hpp"

#include <memory>

#include "default_opset.hpp"
#include "exceptions.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector leaky_relu(const Node& node) {
    auto data = node.get_ng_inputs().at(0);
    double alpha = node.get_attribute_value<double>("alpha", 0.01);

    std::shared_ptr<ngraph::Node> alpha_node =
        default_opset::Constant::create(data.get_element_type(), Shape{1}, {alpha});
    return {std::make_shared<default_opset::PRelu>(data, alpha_node)};
}

}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
