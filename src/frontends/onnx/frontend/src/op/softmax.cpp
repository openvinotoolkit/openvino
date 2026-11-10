// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/softmax.hpp"

#include "core/operator_set.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "utils/common.hpp"
#include "utils/reshape.hpp"
using namespace ov::op;

namespace ov::frontend::onnx {
namespace {
std::shared_ptr<ov::Node> onnx_softmax(const ov::Output<ov::Node> data, const int64_t axis) {
    const auto coerced_data = ov::op::util::flatten(data, static_cast<int>(axis));
    const auto result = std::make_shared<v8::Softmax>(coerced_data, 1);
    const auto data_shape = std::make_shared<v3::ShapeOf>(data);
    const bool special_zero = false;
    return std::make_shared<v1::Reshape>(result, data_shape, special_zero);
}

ov::OutputVector softmax(const ov::frontend::onnx::Node& node) {
    const auto data = node.get_ov_inputs().at(0);
    const auto data_rank = data.get_partial_shape().rank();
    FRONT_END_GENERAL_CHECK(data_rank.is_static(), "ONNX Softmax data rank needs to be known (static)");

    const auto axis = node.get_attribute_value<int64_t>("axis", 1);

    std::shared_ptr<ov::Node> result;
    if (data_rank.get_length() == 0) {
        return {v0::Constant::create(data.get_element_type(), ov::Shape{}, {1})};
    } else {
        // ONNX Softmax opset 1-12 coerces the input to a 2-D matrix
        // [prod(d0..d_axis-1), prod(d_axis..d_rank-1)], applies softmax on the second axis and
        // restores the shape. Flattening at the last axis is a no-op reshape, so it is skipped.
        const auto normalized_axis = common::normalize_axis(node.get_description(), axis, data_rank);
        if (normalized_axis == data_rank.get_length() - 1) {
            return {std::make_shared<v8::Softmax>(data, axis)};
        } else {
            return {onnx_softmax(data, normalized_axis)};
        }
    }
}
}  // namespace
}  // namespace ov::frontend::onnx

namespace ov::frontend::onnx::ai_onnx::opset_1 {
ov::OutputVector softmax(const ov::frontend::onnx::Node& node) {
    const auto data = node.get_ov_inputs().at(0);
    const auto data_rank = data.get_partial_shape().rank();
    const auto axis = node.get_attribute_value<int64_t>("axis", 1);
    FRONT_END_GENERAL_CHECK(data_rank.is_static(), "ONNX Softmax data rank needs to be known (static)");
    // Opsets 1-10 allow the flattening boundary axis equal to the input rank.
    if (data_rank.get_length() > 0 && axis == data_rank.get_length()) {
        return {ov::frontend::onnx::onnx_softmax(data, axis)};
    }
    return ov::frontend::onnx::softmax(node);
}
ONNX_OP("Softmax", OPSET_RANGE(1, 10), ai_onnx::opset_1::softmax);
}  // namespace ov::frontend::onnx::ai_onnx::opset_1

namespace ov::frontend::onnx::ai_onnx::opset_11 {
ov::OutputVector softmax(const ov::frontend::onnx::Node& node) {
    return ov::frontend::onnx::softmax(node);
}
ONNX_OP("Softmax", OPSET_RANGE(11, 12), ai_onnx::opset_11::softmax);
}  // namespace ov::frontend::onnx::ai_onnx::opset_11

namespace ov::frontend::onnx::ai_onnx::opset_13 {
ov::OutputVector softmax(const ov::frontend::onnx::Node& node) {
    const auto data = node.get_ov_inputs().at(0);

    const auto axis = node.get_attribute_value<int64_t>("axis", -1);

    return {std::make_shared<v8::Softmax>(data, axis)};
}
ONNX_OP("Softmax", OPSET_SINCE(13), ai_onnx::opset_13::softmax);
}  // namespace ov::frontend::onnx::ai_onnx::opset_13
