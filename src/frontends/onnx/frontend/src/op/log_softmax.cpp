// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/log_softmax.hpp"

#include <memory>

#include "default_opset.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/validation_util.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace {
std::shared_ptr<ngraph::Node> onnx_logsoftmax(const Output<ngraph::Node> data, const int64_t axis) {
    const auto coerced_data = ngraph::builder::opset1::flatten(data, static_cast<int>(axis));
    const auto result = std::make_shared<default_opset::LogSoftmax>(coerced_data, 1);
    const auto data_shape = std::make_shared<default_opset::ShapeOf>(data);
    return std::make_shared<default_opset::Reshape>(result, data_shape, false);
}

OutputVector log_softmax(const Node& node, const int64_t DEFAULT_AXIS) {
    OutputVector inputs{node.get_ng_inputs()};
    const auto data = inputs.at(0);
    const auto data_rank = data.get_partial_shape().rank();

    NGRAPH_CHECK(data_rank.is_static(), "ONNX Softmax data rank needs to be known (static)");

    const auto axis = node.get_attribute_value<int64_t>("axis", DEFAULT_AXIS);

    std::shared_ptr<ngraph::Node> result;
    switch (data_rank.get_length()) {
    case 0: {
        result = default_opset::Constant::create(data.get_element_type(), Shape{}, {1});
        break;
    }
    case 1: {
        // checks if the axis belongs to the allowed values set (-1 and 0 for 1D)
        OPENVINO_SUPPRESS_DEPRECATED_START
        ngraph::normalize_axis(node.get_description(), axis, data_rank);
        OPENVINO_SUPPRESS_DEPRECATED_END
        result = std::make_shared<default_opset::LogSoftmax>(data, 0);
        break;
    }
    default: {
        OPENVINO_SUPPRESS_DEPRECATED_START
        const auto normalized_axis = ngraph::normalize_axis(node.get_description(), axis, data_rank);
        OPENVINO_SUPPRESS_DEPRECATED_END

        result = onnx_logsoftmax(data, normalized_axis);
        break;
    }
    }

    return {result};
}
}  // namespace

namespace op {
namespace set_1 {
OutputVector log_softmax(const Node& node) {
    return ngraph::onnx_import::log_softmax(node, 1);
}
}  // namespace set_1

namespace set_13 {
OutputVector log_softmax(const Node& node) {
    const auto axis = node.get_attribute_value<int64_t>("axis", -1);
    return {std::make_shared<default_opset::LogSoftmax>(node.get_ng_inputs()[0], axis)};
}
}  // namespace set_13

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
