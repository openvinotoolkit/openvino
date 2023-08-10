// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/pow.hpp"

#include <memory>

#include "default_opset.hpp"
#include "ngraph/node.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector pow(const Node& node) {
    auto inputs = node.get_ng_inputs();
    NGRAPH_CHECK(inputs.size() == 2, "Power operation requires 2 inputs. Got: ", inputs.size());

    auto base = inputs[0];
    auto exponent = inputs[1];
    auto base_type = inputs[0].get_element_type();
    auto exponent_type = inputs[1].get_element_type();
    if (exponent_type != base_type) {
        if (exponent_type.is_integral() || (base_type.is_real() && base_type.bitwidth() >= exponent_type.bitwidth())) {
            exponent = std::make_shared<default_opset::Convert>(exponent, base_type);
        } else {
            base = std::make_shared<default_opset::Convert>(base, exponent_type);
            auto power = std::make_shared<default_opset::Power>(base, exponent);
            return {std::make_shared<default_opset::Convert>(power, base_type)};
        }
    }
    return {std::make_shared<default_opset::Power>(base, exponent)};
}

}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
