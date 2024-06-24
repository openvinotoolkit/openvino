// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/softmax.hpp"

#include "core/operator_set.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "utils/reshape.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace {
std::shared_ptr<ov::Node> onnx_softmax(const ov::Output<ov::Node> data, const int64_t axis) {
    const auto coerced_data = ov::op::util::flatten(data, static_cast<int>(axis));
    const auto result = std::make_shared<v8::Softmax>(coerced_data, 1);
    const auto data_shape = std::make_shared<v3::ShapeOf>(data);
    const bool special_zero = false;
    return std::make_shared<v1::Reshape>(result, data_shape, special_zero);
}
}  // namespace

namespace ai_onnx {
namespace opset_1 {
ov::OutputVector softmax(const ov::frontend::onnx::Node& node) {
    const auto data = node.get_ov_inputs().at(0);
    const auto data_rank = data.get_partial_shape().rank();
    FRONT_END_GENERAL_CHECK(data_rank.is_static(), "ONNX Softmax data rank needs to be known (static)");

    const auto axis = node.get_attribute_value<int64_t>("axis", 1);

    std::shared_ptr<ov::Node> result;
    switch (data_rank.get_length()) {
    case 0: {
        result = v0::Constant::create(data.get_element_type(), ov::Shape{}, {1});
        break;
    }
    default: {
        result = onnx_softmax(data, axis);
        break;
    }
    }

    return {result};
}
ONNX_OP("Softmax", OPSET_RANGE(1, 10), ai_onnx::opset_1::softmax);
}  // namespace opset_1
namespace opset_11 {
ov::OutputVector softmax(const ov::frontend::onnx::Node& node) {
    const auto data = node.get_ov_inputs().at(0);
    const auto data_rank = data.get_partial_shape().rank();
    FRONT_END_GENERAL_CHECK(data_rank.is_static(), "ONNX Softmax data rank needs to be known (static)");

    const auto axis = node.get_attribute_value<int64_t>("axis", 1);

    std::shared_ptr<ov::Node> result;
    switch (data_rank.get_length()) {
    case 0: {
        result = v0::Constant::create(data.get_element_type(), ov::Shape{}, {1});
        break;
    }
    default: {
        result = std::make_shared<v8::Softmax>(data, axis);
        break;
    }
    }

    return {result};
}
ONNX_OP("Softmax", OPSET_RANGE(11, 12), ai_onnx::opset_11::softmax);
}  // namespace opset_11
namespace opset_13 {
ov::OutputVector softmax(const ov::frontend::onnx::Node& node) {
    const auto data = node.get_ov_inputs().at(0);

    const auto axis = node.get_attribute_value<int64_t>("axis", -1);

    return {std::make_shared<v8::Softmax>(data, axis)};
}
ONNX_OP("Softmax", OPSET_SINCE(13), ai_onnx::opset_13::softmax);
}  // namespace opset_13
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
