// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/clip.hpp"

#include <limits>
#include <memory>

#include "default_opset.hpp"
#include "ngraph/builder/make_constant.hpp"
#include "onnx_import/core/null_node.hpp"

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
    Output<ngraph::Node> result_node = inputs.at(0);

    if (inputs.size() > 1 && !ngraph::op::is_null(inputs.at(1))) {
        result_node = std::make_shared<default_opset::Maximum>(result_node, inputs.at(1));
    }

    if (inputs.size() == 3 && !ngraph::op::is_null(inputs.at(2))) {
        result_node = std::make_shared<default_opset::Minimum>(result_node, inputs.at(2));
    }

    return {result_node};
}

}  // namespace set_11

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
