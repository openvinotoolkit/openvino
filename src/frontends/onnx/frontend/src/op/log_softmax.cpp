// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/log_softmax.hpp"

#include "openvino/core/validation_util.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/log.hpp"
#include "openvino/op/log_softmax.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "ov_models/ov_builders/reshape.hpp"
#include "validation_util.hpp"

using namespace ov::op;

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace {
std::shared_ptr<ov::Node> onnx_logsoftmax(const ov::Output<ov::Node> data, const int64_t axis) {
    const auto coerced_data = ov::op::util::flatten(data, static_cast<int>(axis));
    const auto result = std::make_shared<v5::LogSoftmax>(coerced_data, 1);
    const auto data_shape = std::make_shared<v3::ShapeOf>(data);
    return std::make_shared<v1::Reshape>(result, data_shape, false);
}

ov::OutputVector log_softmax(const Node& node, const int64_t DEFAULT_AXIS) {
    ov::OutputVector inputs{node.get_ng_inputs()};
    const auto data = inputs.at(0);
    const auto data_rank = data.get_partial_shape().rank();

    FRONT_END_GENERAL_CHECK(data_rank.is_static(), "ONNX Softmax data rank needs to be known (static)");

    const auto axis = node.get_attribute_value<int64_t>("axis", DEFAULT_AXIS);

    std::shared_ptr<ov::Node> result;
    switch (data_rank.get_length()) {
    case 0: {
        result = v0::Constant::create(data.get_element_type(), ov::Shape{}, {1});
        break;
    }
    case 1: {
        // checks if the axis belongs to the allowed values set (-1 and 0 for 1D)
        ov::util::normalize_axis(node.get_description(), axis, data_rank);
        result = std::make_shared<v5::LogSoftmax>(data, 0);
        break;
    }
    default: {
        const auto normalized_axis = ov::util::normalize_axis(node.get_description(), axis, data_rank);

        result = onnx_logsoftmax(data, normalized_axis);
        break;
    }
    }

    return {result};
}
}  // namespace

namespace op {
namespace set_1 {
ov::OutputVector log_softmax(const Node& node) {
    return ngraph::onnx_import::log_softmax(node, 1);
}
}  // namespace set_1

namespace set_13 {
ov::OutputVector log_softmax(const Node& node) {
    const auto axis = node.get_attribute_value<int64_t>("axis", -1);
    return {std::make_shared<v5::LogSoftmax>(node.get_ng_inputs()[0], axis)};
}
}  // namespace set_13

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
