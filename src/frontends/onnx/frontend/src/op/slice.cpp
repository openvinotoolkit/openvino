// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/slice.hpp"

#include <memory>
#include <vector>

#include "default_opset.hpp"
#include "onnx_import/core/null_node.hpp"
#include "openvino/opsets/opset8.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_10 {
OutputVector slice(const Node& node) {
    using ngraph::op::is_null;

    OutputVector inputs{node.get_ng_inputs()};
    const auto& data = inputs.at(0);
    const auto& starts = inputs.at(1);
    const auto& ends = inputs.at(2);

    const bool axes_input_provided = inputs.size() >= 4 && !is_null(inputs.at(3));
    const bool steps_input_provided = inputs.size() == 5 && !is_null(inputs.at(4));

    Output<ngraph::Node> steps;
    if (steps_input_provided) {
        steps = inputs.at(4);
    } else {
        const auto& default_step = default_opset::Constant::create(starts.get_element_type(), {1}, {1});
        steps =
            std::make_shared<default_opset::Broadcast>(default_step,
                                                       std::make_shared<default_opset::ShapeOf>(starts, element::i64));
    }

    if (axes_input_provided) {
        const auto axes = inputs.at(3);
        return {std::make_shared<ov::opset8::Slice>(data, starts, ends, steps, axes)};
    } else {
        return {std::make_shared<ov::opset8::Slice>(data, starts, ends, steps)};
    }
}
}  // namespace set_10

namespace set_1 {
OutputVector slice(const Node& node) {
    Output<ngraph::Node> data = node.get_ng_inputs().at(0);
    const auto starts_atr = node.get_attribute_value<std::vector<int64_t>>("starts");
    const auto ends = node.get_attribute_as_constant<std::vector<int64_t>>("ends");

    const auto starts = std::make_shared<default_opset::Constant>(element::i64, Shape{starts_atr.size()}, starts_atr);
    auto axes_atr = node.get_attribute_value<std::vector<int64_t>>("axes", std::vector<int64_t>());

    const auto steps = default_opset::Constant::create(element::i64,
                                                       Shape{starts_atr.size()},
                                                       std::vector<int64_t>(starts_atr.size(), 1));

    if (axes_atr.empty()) {
        return {std::make_shared<ov::opset8::Slice>(data, starts, ends, steps)};
    } else {
        const auto& axes = std::make_shared<default_opset::Constant>(element::i64, Shape{axes_atr.size()}, axes_atr);
        return {std::make_shared<ov::opset8::Slice>(data, starts, ends, steps, axes)};
    }
}
}  // namespace set_1
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
