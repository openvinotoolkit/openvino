// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/leaky_relu.hpp"

#include "exceptions.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/prelu.hpp"

using namespace ov::op;

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector leaky_relu(const Node& node) {
    auto data = node.get_ng_inputs().at(0);
    double alpha = node.get_attribute_value<double>("alpha", 0.01);

    std::shared_ptr<ov::Node> alpha_node = v0::Constant::create(data.get_element_type(), ov::Shape{1}, {alpha});
    return {std::make_shared<v0::PRelu>(data, alpha_node)};
}

}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
