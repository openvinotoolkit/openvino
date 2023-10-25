// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/hard_sigmoid.hpp"

#include <memory>

#include "default_opset.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector hard_sigmoid(const Node& node) {
    const auto data = node.get_ng_inputs().at(0);

    const auto alpha =
        default_opset::Constant::create<double>(data.get_element_type(),
                                                Shape{},
                                                std::vector<double>{node.get_attribute_value<double>("alpha", 0.2)});

    const auto beta =
        default_opset::Constant::create<double>(data.get_element_type(),
                                                Shape{},
                                                std::vector<double>{node.get_attribute_value<double>("beta", 0.5)});

    return {std::make_shared<default_opset::HardSigmoid>(data, alpha, beta)};
}

}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
