// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/scatter_elements.hpp"

#include <memory>

#include "default_opset.hpp"
#include "exceptions.hpp"
#include "openvino/opsets/opset12.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector scatter_elements(const Node& node) {
    using namespace ov::opset12;
    const auto data = node.get_ng_inputs().at(0);
    const auto indices = node.get_ng_inputs().at(1);
    const auto updates = node.get_ng_inputs().at(2);
    const auto axis_node = node.get_attribute_as_constant<std::int64_t>("axis", 0);

    ScatterElementsUpdate::Reduction reduction_ov;
    if (node.has_attribute("reduction")) {
        std::string reduction_onnx = node.get_attribute_value<std::string>("reduction", "none");
        if (reduction_onnx == "none") {
            reduction_ov = ScatterElementsUpdate::Reduction::NONE;
        } else if (reduction_onnx == "add") {
            reduction_ov = ScatterElementsUpdate::Reduction::SUM;
        } else if (reduction_onnx == "mul") {
            reduction_ov = ScatterElementsUpdate::Reduction::PROD;
        } else if (reduction_onnx == "min") {
            reduction_ov = ScatterElementsUpdate::Reduction::MIN;
        } else if (reduction_onnx == "max") {
            reduction_ov = ScatterElementsUpdate::Reduction::MAX;
        } else {
            CHECK_VALID_NODE(node,
                             false,
                             "Unsupported value of attribute: `reduction`. "
                             "Supported modes: `none`, `add`, `mul`, `min`, `max`, got:",
                             reduction_onnx);
        }
    } else {
        reduction_ov = ScatterElementsUpdate::Reduction::NONE;
    }

    return {std::make_shared<ScatterElementsUpdate>(data, indices, updates, axis_node, reduction_ov)};
}
}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
