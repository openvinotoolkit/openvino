// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "openvino/op/convert.hpp"
#include "utils/reshape.hpp"
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
namespace detail {
extern ov::OutputVector matmul(const ov::Output<ov::Node>& a, const ov::Output<ov::Node>& b);
extern std::shared_ptr<ov::Node> make_fake_quantize(const ov::Output<ov::Node>& y_scale,
                                                    const ov::Output<ov::Node>& y_zero_point,
                                                    const ov::Output<ov::Node>& data);
}  // namespace detail

namespace opset_1 {
ov::OutputVector qlinear_matmul(const ov::frontend::onnx::Node& node) {
    const ov::OutputVector& inputs = node.get_ov_inputs();

    const auto& a = inputs.at(0);
    const auto& a_scale = reshape::interpret_as_scalar(inputs.at(1));
    const auto& a_zero_point = reshape::interpret_as_scalar(inputs.at(2));
    const auto& b = inputs.at(3);
    const auto& b_scale = reshape::interpret_as_scalar(inputs.at(4));
    const auto& b_zero_point = reshape::interpret_as_scalar(inputs.at(5));
    const auto& y_scale = inputs.at(6);
    const auto& y_zero_point = inputs.at(7);

    const auto& dequnatize_a =
        ai_onnx::opset_13::detail::dequantize_linear(a,
                                                     a_scale,
                                                     std::make_shared<v0::Convert>(a_zero_point, ov::element::f32),
                                                     1,
                                                     node);
    const auto& dequnatize_b =
        ai_onnx::opset_13::detail::dequantize_linear(b,
                                                     b_scale,
                                                     std::make_shared<v0::Convert>(b_zero_point, ov::element::f32),
                                                     1,
                                                     node);

    const auto& result = ai_onnx::detail::matmul(dequnatize_a[0], dequnatize_b[0]);

    const auto& quantized_result = ai_onnx::detail::make_fake_quantize(y_scale, y_zero_point, result[0]);

    return {quantized_result};
}
ONNX_OP("QLinearMatMul", OPSET_SINCE(1), ai_onnx::opset_1::qlinear_matmul);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
