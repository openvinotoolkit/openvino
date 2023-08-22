// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/softmax.hpp"

#include <memory>

#include "default_opset.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/validation_util.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace {
std::shared_ptr<ngraph::Node> onnx_softmax(const Output<ngraph::Node> data, const int64_t axis) {
    const auto coerced_data = ngraph::builder::opset1::flatten(data, static_cast<int>(axis));
    const auto result = std::make_shared<default_opset::Softmax>(coerced_data, 1);
    const auto data_shape = std::make_shared<default_opset::ShapeOf>(data);
    const bool special_zero = false;
    return std::make_shared<default_opset::Reshape>(result, data_shape, special_zero);
}
}  // namespace

namespace op {
namespace set_1 {
OutputVector softmax(const Node& node) {
    const auto data = node.get_ng_inputs().at(0);
    const auto data_rank = data.get_partial_shape().rank();
    NGRAPH_CHECK(data_rank.is_static(), "ONNX Softmax data rank needs to be known (static)");

    const auto axis = node.get_attribute_value<int64_t>("axis", 1);

    std::shared_ptr<ngraph::Node> result;
    switch (data_rank.get_length()) {
    case 0: {
        result = default_opset::Constant::create(data.get_element_type(), Shape{}, {1});
        break;
    }
    default: {
        result = onnx_softmax(data, axis);
        break;
    }
    }

    return {result};
}
}  // namespace set_1
namespace set_11 {
OutputVector softmax(const Node& node) {
    const auto data = node.get_ng_inputs().at(0);
    const auto data_rank = data.get_partial_shape().rank();
    NGRAPH_CHECK(data_rank.is_static(), "ONNX Softmax data rank needs to be known (static)");

    const auto axis = node.get_attribute_value<int64_t>("axis", 1);

    std::shared_ptr<ngraph::Node> result;
    switch (data_rank.get_length()) {
    case 0: {
        result = default_opset::Constant::create(data.get_element_type(), Shape{}, {1});
        break;
    }
    default: {
        result = std::make_shared<ov::op::v8::Softmax>(data, axis);
        break;
    }
    }

    return {result};
}
}  // namespace set_11
namespace set_13 {
OutputVector softmax(const Node& node) {
    const auto data = node.get_ng_inputs().at(0);

    const auto axis = node.get_attribute_value<int64_t>("axis", -1);

    return {std::make_shared<ov::op::v8::Softmax>(data, axis)};
}
}  // namespace set_13
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
