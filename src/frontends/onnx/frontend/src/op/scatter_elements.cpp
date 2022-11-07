// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/scatter_elements.hpp"

#include <memory>

#include "default_opset.hpp"
#include "exceptions.hpp"
#include "ngraph/opsets/opset3.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector scatter_elements(const Node& node) {
    const auto data = node.get_ng_inputs().at(0);
    const auto indices = node.get_ng_inputs().at(1);
    const auto updates = node.get_ng_inputs().at(2);
    const auto axis_node = node.get_attribute_as_constant<std::int64_t>("axis", 0);
    if (node.has_attribute("reduction")) {
        const auto reduction = node.get_attribute_value<std::string>("reduction", "none");
        CHECK_VALID_NODE(node,
                         reduction == "none",
                         "Unsupported value of attribute: `reduction`. Only `none` is supported, got:",
                         reduction);
    }
    return {std::make_shared<ngraph::opset3::ScatterElementsUpdate>(data, indices, updates, axis_node)};
}
}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
