// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Disabled in CMakeList
// Update to higher opset required

#include "op/qlinear_matmul.hpp"

#include <cstddef>
#include <memory>
#include <vector>

#include "matmul.hpp"
#include "dequantize_linear.hpp"
#include "exceptions.hpp"
#include "ngraph/opsets/opset6.hpp"
#include "onnx_import/core/null_node.hpp"
#include "quantize_linear.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector qlinear_matmul(const Node& node) {
    const OutputVector& inputs = node.get_ng_inputs();

    auto a = inputs.at(0);
    auto a_scale = inputs.at(1);
    auto a_zero_point = inputs.at(2);
    auto b = inputs.at(3);
    auto b_scale = inputs.at(4);
    auto b_zero_point = inputs.at(5);
    auto y_scale = inputs.at(6);
    auto y_zero_point = inputs.at(7);

    a = set_13::detail::dequantize_linear(a,
                                          a_scale,
                                          std::make_shared<opset6::Convert>(a_zero_point, element::f32),
                                          1,
                                          node)[0];
    b = set_13::detail::dequantize_linear(b,
                                          b_scale,
                                          std::make_shared<opset6::Convert>(b_zero_point, element::f32),
                                          1,
                                          node)[0];

    auto result = detail::matmul(a, b)[0];

    result = op::detail::make_fake_quantize(y_scale, y_zero_point, result);

    return {result};
}

}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
