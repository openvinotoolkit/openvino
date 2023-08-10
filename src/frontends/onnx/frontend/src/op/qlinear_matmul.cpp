// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/qlinear_matmul.hpp"

#include <cstddef>
#include <memory>
#include <vector>

#include "dequantize_linear.hpp"
#include "matmul.hpp"
#include "ngraph/opsets/opset6.hpp"
#include "quantize_linear.hpp"
#include "utils/reshape.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector qlinear_matmul(const Node& node) {
    const OutputVector& inputs = node.get_ng_inputs();

    const auto& a = inputs.at(0);
    const auto& a_scale = reshape::interpret_as_scalar(inputs.at(1));
    const auto& a_zero_point = reshape::interpret_as_scalar(inputs.at(2));
    const auto& b = inputs.at(3);
    const auto& b_scale = reshape::interpret_as_scalar(inputs.at(4));
    const auto& b_zero_point = reshape::interpret_as_scalar(inputs.at(5));
    const auto& y_scale = inputs.at(6);
    const auto& y_zero_point = inputs.at(7);

    const auto& dequnatize_a =
        set_13::detail::dequantize_linear(a,
                                          a_scale,
                                          std::make_shared<opset6::Convert>(a_zero_point, element::f32),
                                          1,
                                          node);
    const auto& dequnatize_b =
        set_13::detail::dequantize_linear(b,
                                          b_scale,
                                          std::make_shared<opset6::Convert>(b_zero_point, element::f32),
                                          1,
                                          node);

    const auto& result = op::detail::matmul(dequnatize_a[0], dequnatize_b[0]);

    const auto& quantized_result = op::detail::make_fake_quantize(y_scale, y_zero_point, result[0]);

    return {quantized_result};
}
}  // namespace set_1
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
