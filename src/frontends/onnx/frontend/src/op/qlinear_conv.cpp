// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Disabled in CMakeList
// Update to higher opset required

#include "op/qlinear_conv.hpp"

#include <cstddef>
#include <memory>
#include <vector>

#include "conv.hpp"
#include "dequantize_linear.hpp"
#include "exceptions.hpp"
#include "ngraph/opsets/opset6.hpp"
#include "onnx_import/core/null_node.hpp"
#include "quantize_linear.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector qlinear_conv(const Node& node) {
    const OutputVector& inputs = node.get_ng_inputs();

    auto x = inputs.at(0);
    auto x_scale = inputs.at(1);
    auto x_zero_point = inputs.at(2);
    auto w = inputs.at(3);
    auto w_scale = inputs.at(4);
    auto w_zero_point = inputs.at(5);
    auto y_scale = inputs.at(6);
    auto y_zero_point = inputs.at(7);
    Output<ngraph::Node> B = inputs.size() > 8 ? inputs.at(8) : std::make_shared<NullNode>()->output(0);

    x = set_13::detail::dequantize_linear(x,
                                          x_scale,
                                          std::make_shared<opset6::Convert>(x_zero_point, element::f32),
                                          1,
                                          node)[0];
    w = set_13::detail::dequantize_linear(w,
                                          w_scale,
                                          std::make_shared<opset6::Convert>(w_zero_point, element::f32),
                                          1,
                                          node)[0];

    if (!ngraph::op::is_null(B)) {
        B = std::make_shared<opset6::Multiply>(std::make_shared<opset6::Convert>(B, x_scale.get_element_type()),
                                               std::make_shared<opset6::Multiply>(x_scale, w_scale))
                ->output(0);
    }

    auto result = detail::conv(node, x, w, B)[0];

    result = op::detail::make_fake_quantize(y_scale, y_zero_point, result);

    return {result};
}

}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
