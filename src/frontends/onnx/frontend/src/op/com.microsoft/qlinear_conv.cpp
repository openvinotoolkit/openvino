// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "core/null_node.hpp"
#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"  // ✅ Include Transpose
#include "utils/common.hpp"
#include "utils/conv_factory.hpp"
#include "utils/convpool.hpp"
#include "utils/reshape.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace com_microsoft {
namespace opset_1 {

ov::OutputVector qlinear_conv(const ov::frontend::onnx::Node& node) {
    common::default_op_checks(node, 8);
    const ov::OutputVector& inputs = node.get_ov_inputs();

    auto x = inputs[0];  // Input tensor
    auto x_scale = inputs[1];
    auto x_zero_point = inputs[2];
    auto w = inputs[3];  // Weight tensor
    auto w_scale = inputs[4];
    auto w_zero_point = inputs[5];
    auto y_scale = inputs[6];
    auto y_zero_point = inputs[7];

    auto strides = convpool::get_strides(node);
    auto dilations = convpool::get_dilations(node);
    auto pads = convpool::get_pads(node);
    auto auto_pad_type = convpool::get_auto_pad(node);
    auto groups = node.get_attribute_value<int64_t>("group", 1);
    auto kernel_shape = node.get_attribute_value<std::vector<int64_t>>("kernel_shape", {});
    auto channels_last = node.get_attribute_value<int64_t>("channels_last", 0);

    if (!kernel_shape.empty()) {
        FRONT_END_GENERAL_CHECK(kernel_shape.size() == w.get_partial_shape().rank().get_length() - 2,
                                "Provided kernel_shape does not match weight tensor dimensions.");
    }

    ov::Output<ov::Node> x_dequantized = std::make_shared<v1::Multiply>(
        x_scale,
        std::make_shared<v0::Convert>(std::make_shared<v1::Subtract>(x, x_zero_point), x_scale.get_element_type()));

    ov::Output<ov::Node> w_dequantized = std::make_shared<v1::Multiply>(
        w_scale,
        std::make_shared<v0::Convert>(std::make_shared<v1::Subtract>(w, w_zero_point), w_scale.get_element_type()));

    //bool need_reorder_output = false;

    //// **Handle NHWC → NCHW conversion**
    //auto x_shape = x.get_partial_shape();
    //if (channels_last && x_shape.rank().is_static() && x_shape.rank().get_length() > 2) {
    //    std::vector<int64_t> transpose_order(x_shape.rank().get_length());
    //    transpose_order[0] = 0;      // Batch dim remains the same
    //    transpose_order.back() = 1;  // Move channels to second position
    //    for (size_t i = 1; i < transpose_order.size() - 1; ++i) {
    //        transpose_order[i] = i + 1;
    //    }

    //    auto order_const = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
    //                                                              ov::Shape{transpose_order.size()},
    //                                                              transpose_order);

    //    x_dequantized = std::make_shared<v1::Transpose>(x_dequantized, order_const)->output(0);
    //    need_reorder_output = true;
    //}

    ov::Output<ov::Node> conv_node = conv_factory::make_ng_convolution(x_dequantized,
                                                                       w_dequantized,
                                                                       strides,
                                                                       dilations,
                                                                       pads.first,
                                                                       pads.second,
                                                                       groups,
                                                                       auto_pad_type);

    if (inputs.size() > 8) {
        auto bias = inputs[8];
        conv_node = std::make_shared<v1::Add>(conv_node, bias)->output(0);
    }

    auto result_divided = std::make_shared<v1::Divide>(conv_node, y_scale);

    auto y_zero_point_float = std::make_shared<v0::Convert>(y_zero_point, y_scale.get_element_type());

    auto result_shifted = std::make_shared<v1::Add>(result_divided, y_zero_point_float);
    auto y_quantized = std::make_shared<v0::Convert>(result_shifted, x.get_element_type())->output(0);

    // **Convert back to NHWC if needed**
  /*  if (need_reorder_output) {
        std::vector<int64_t> reverse_transpose_order(x_shape.rank().get_length());
        reverse_transpose_order[0] = 0;
        reverse_transpose_order[1] = x_shape.rank().get_length() - 1;
        for (size_t i = 2; i < reverse_transpose_order.size(); ++i) {
            reverse_transpose_order[i] = i - 1;
        }

        auto reverse_order_const = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                                          ov::Shape{reverse_transpose_order.size()},
                                                                          reverse_transpose_order);

        y_quantized = std::make_shared<v1::Transpose>(y_quantized, reverse_order_const)->output(0);
    }*/

    return {y_quantized};
}

ONNX_OP("QLinearConv", OPSET_SINCE(1), com_microsoft::opset_1::qlinear_conv, MICROSOFT_DOMAIN);

}  // namespace opset_1
}  // namespace com_microsoft
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
