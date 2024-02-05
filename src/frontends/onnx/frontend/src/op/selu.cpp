// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/selu.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/op/selu.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
ov::OutputVector selu(const ov::frontend::onnx::Node& node) {
    auto data = node.get_ov_inputs().at(0);
    auto alpha = node.get_attribute_value<double>("alpha", 1.67326319217681884765625);
    auto gamma = node.get_attribute_value<double>("gamma", 1.05070102214813232421875);

    auto alpha_node = v0::Constant::create(data.get_element_type(), ov::Shape{}, {alpha});

    auto gamma_node = v0::Constant::create(data.get_element_type(), ov::Shape{}, {gamma});

    return {std::make_shared<v0::Selu>(data, alpha_node, gamma_node)};
}

}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
