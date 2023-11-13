// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/clip.hpp"

#include <limits>
#include <memory>

#include "default_opset.hpp"
#include "ngraph/builder/make_constant.hpp"
#include "ngraph/validation_util.hpp"
#include "onnx_import/core/null_node.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector clip(const Node& node) {
    const auto data = node.get_ng_inputs().at(0);

    const double max_value = node.get_attribute_value<double>("max", std::numeric_limits<double>::max());

    const double min_value = node.get_attribute_value<double>("min", std::numeric_limits<double>::lowest());

    return {std::make_shared<default_opset::Clamp>(data, min_value, max_value)};
}

}  // namespace set_1

namespace set_11 {
OutputVector clip(const Node& node) {
    const OutputVector inputs{node.get_ng_inputs()};
    const Output<ngraph::Node> data = inputs.at(0);
    const element::Type data_type = data.get_element_type();
    Output<ngraph::Node> min;
    Output<ngraph::Node> max;

    // If second input is provided, assign to min input, otherwise set lowest
    // numeric limit of data type as min input.
    if (inputs.size() > 1 && !ngraph::op::is_null(inputs.at(1))) {
        min = inputs.at(1);
    } else {
        OPENVINO_SUPPRESS_DEPRECATED_START
        min = ngraph::get_constant_lowest_of_type(data_type);
        OPENVINO_SUPPRESS_DEPRECATED_END
    }

    // If third input is provided, assign to max input, otherwise set maximum
    // numeric limit of data type as max input.
    if (inputs.size() == 3 && !ngraph::op::is_null(inputs.at(2))) {
        max = inputs.at(2);
    } else {
        OPENVINO_SUPPRESS_DEPRECATED_START
        max = ngraph::get_constant_max_of_type(data_type);
        OPENVINO_SUPPRESS_DEPRECATED_END
    }

    const auto max_of_min_and_data = std::make_shared<default_opset::Maximum>(min, data);

    return {std::make_shared<default_opset::Minimum>(max, max_of_min_and_data)};
}

}  // namespace set_11

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
