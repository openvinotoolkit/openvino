// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Disabled in CMakeList
// Update to higher opset required

#include "core/null_node.hpp"
#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
// Link with an existing translator
namespace opset_13 {
namespace detail {
extern ov::OutputVector dequantize_linear(const ov::Output<ov::Node>& x,
                                          const ov::Output<ov::Node>& scale,
                                          const std::shared_ptr<ov::Node>& zero_point,
                                          int64_t axis,
                                          const Node& node);
}  // namespace detail
}  // namespace opset_13
namespace opset_1 {
namespace detail {
ov::OutputVector conv(const ov::frontend::onnx::Node& node,
                      ov::Output<ov::Node> data,
                      ov::Output<ov::Node> filters,
                      ov::Output<ov::Node> bias);
}  // namespace detail
}  // namespace opset_1
namespace detail {
extern ov::OutputVector matmul(const ov::Output<ov::Node>& a, const ov::Output<ov::Node>& b);
extern std::shared_ptr<ov::Node> make_fake_quantize(const ov::Output<ov::Node>& y_scale,
                                                    const ov::Output<ov::Node>& y_zero_point,
                                                    const ov::Output<ov::Node>& data);
}  // namespace detail

namespace opset_1 {
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

    x = ai_onnx::opset_13::detail::dequantize_linear(x,
                                                     x_scale,
                                                     std::make_shared<v0::Convert>(x_zero_point, ov::element::f32),
                                                     1,
                                                     node)[0];
    w = ai_onnx::opset_13::detail::dequantize_linear(w,
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

    result = ai_onnx::detail::make_fake_quantize(y_scale, y_zero_point, result);

    return {result};
}

ONNX_OP("QLinearConv", OPSET_SINCE(1), ai_onnx::opset_1::qlinear_conv);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
