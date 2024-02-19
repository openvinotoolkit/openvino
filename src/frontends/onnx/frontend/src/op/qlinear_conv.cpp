// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Disabled in CMakeList
// Update to higher opset required

#include "op/qlinear_conv.hpp"

#include "conv.hpp"
#include "core/null_node.hpp"
#include "dequantize_linear.hpp"
#include "exceptions.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "quantize_linear.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
ov::OutputVector qlinear_conv(const ov::frontend::onnx::Node& node) {
    const ov::OutputVector& inputs = node.get_ov_inputs();

    auto x = inputs.at(0);
    auto x_scale = inputs.at(1);
    auto x_zero_point = inputs.at(2);
    auto w = inputs.at(3);
    auto w_scale = inputs.at(4);
    auto w_zero_point = inputs.at(5);
    auto y_scale = inputs.at(6);
    auto y_zero_point = inputs.at(7);
    ov::Output<ov::Node> B = inputs.size() > 8 ? inputs.at(8) : std::make_shared<NullNode>()->output(0);

    x = set_13::detail::dequantize_linear(x,
                                          x_scale,
                                          std::make_shared<v0::Convert>(x_zero_point, ov::element::f32),
                                          1,
                                          node)[0];
    w = set_13::detail::dequantize_linear(w,
                                          w_scale,
                                          std::make_shared<v0::Convert>(w_zero_point, ov::element::f32),
                                          1,
                                          node)[0];

    if (!ov::op::util::is_null(B)) {
        B = std::make_shared<v1::Multiply>(std::make_shared<v0::Convert>(B, x_scale.get_element_type()),
                                           std::make_shared<v1::Multiply>(x_scale, w_scale))
                ->output(0);
    }

    auto result = detail::conv(node, x, w, B)[0];

    result = op::detail::make_fake_quantize(y_scale, y_zero_point, result);

    return {result};
}

}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
