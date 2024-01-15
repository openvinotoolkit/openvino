// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/range.hpp"

#include <memory>

#include "default_opset.hpp"
#include "exceptions.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector range(const Node& node) {
    const auto inputs = node.get_ng_inputs();
    CHECK_VALID_NODE(node, inputs.size() >= 3, "Minimum 3 inputs are required. Got: ", inputs.size());

    Output<ngraph::Node> start{inputs[0]};
    Output<ngraph::Node> stop{inputs[1]};
    Output<ngraph::Node> step{inputs[2]};

    auto axes =
        std::make_shared<default_opset::Constant>(ngraph::element::i64, ngraph::Shape{}, std::vector<int64_t>{0});

    // Check if step is a tensor with a single value
    if (start.get_shape().size() == 1 && start.get_shape()[0] == 1) {
        start = std::make_shared<default_opset::Squeeze>(start, axes);
    }

    if (stop.get_shape().size() == 1 && stop.get_shape()[0] == 1) {
        stop = std::make_shared<default_opset::Squeeze>(stop, axes);
    }

    if (step.get_shape().size() == 1 && step.get_shape()[0] == 1) {
        step = std::make_shared<default_opset::Squeeze>(step, axes);
    }

    return {std::make_shared<default_opset::Range>(start, stop, step, start.get_element_type())};
}
}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
