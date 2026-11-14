// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/softmax.hpp"
#include "utils/common.hpp"
#include "utils/reshape.hpp"

using namespace ov::op;

namespace ov::frontend::onnx::com_microsoft::opset_1 {

ov::OutputVector bias_softmax(const ov::frontend::onnx::Node& node) {
    // Original documentation:
    // https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#commicrosoftbiassoftmax
    // Y = Softmax(data + bias). Like the ONNX Softmax-1..12 operator, the input is coerced to a 2D matrix
    // [prod(d_0..d_axis-1), prod(d_axis..d_rank-1)] and normalized along the second dimension.
    // The bias is broadcast to the data with numpy rules, which covers both the inner and the outer
    // broadcast layouts selected by the is_inner_broadcast attribute, so that attribute is not needed here.
    common::default_op_checks(node, 2, 2);

    const auto inputs = node.get_ov_inputs();
    const auto& data = inputs[0];
    const auto& bias = inputs[1];
    const auto& data_type = data.get_element_type();
    CHECK_VALID_NODE(node,
                     data_type.is_dynamic() || data_type == ov::element::f16 || data_type == ov::element::f32 ||
                         data_type == ov::element::f64,
                     "Unsupported input data type for data, expected FP16, FP32 or FP64 but got: ",
                     data_type);

    const auto data_rank = data.get_partial_shape().rank();
    CHECK_VALID_NODE(node, data_rank.is_static(), "BiasSoftmax data rank needs to be known (static)");

    const auto axis = node.get_attribute_value<int64_t>("axis", 1);
    const auto normalized_axis = common::normalize_axis(node.get_description(), axis, data_rank);

    const auto biased_data = std::make_shared<v1::Add>(data, bias);
    if (normalized_axis == data_rank.get_length() - 1) {
        // Flattening at the last axis is a no-op reshape, so the softmax can be applied directly.
        return {std::make_shared<v8::Softmax>(biased_data, normalized_axis)};
    }

    const auto coerced_data = ov::op::util::flatten(biased_data, static_cast<int>(normalized_axis));
    const auto softmax = std::make_shared<v8::Softmax>(coerced_data, 1);
    const auto data_shape = std::make_shared<v3::ShapeOf>(biased_data);
    const bool special_zero = false;
    return {std::make_shared<v1::Reshape>(softmax, data_shape, special_zero)};
}

ONNX_OP("BiasSoftmax", OPSET_SINCE(1), com_microsoft::opset_1::bias_softmax, MICROSOFT_DOMAIN);

}  // namespace ov::frontend::onnx::com_microsoft::opset_1
