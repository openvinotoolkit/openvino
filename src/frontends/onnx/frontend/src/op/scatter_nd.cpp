// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/scatter_nd_update.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector scatter_nd(const ov::frontend::onnx::Node& node) {
    ov::OutputVector ov_inputs{node.get_ov_inputs()};
    auto data = ov_inputs.at(0);
    auto indices = ov_inputs.at(1);
    auto updates = ov_inputs.at(2);
    if (node.has_attribute("reduction")) {
        const auto reduction = node.get_attribute_value<std::string>("reduction", "none");
        CHECK_VALID_NODE(node,
                         reduction == "none",
                         "Unsupported value of attribute: `reduction`. Only `none` is supported, got: ",
                         reduction);
    }

    return {std::make_shared<v3::ScatterNDUpdate>(data, indices, updates)};
}

ONNX_OP("ScatterND", OPSET_RANGE(1, 15), ai_onnx::opset_1::scatter_nd);
}  // namespace opset_1

namespace opset_16 {
// ScatterND-16 introduces the `reduction` attribute with values: none, add, mul.
// ScatterND-18 extends `reduction` with: min, max.
ov::OutputVector scatter_nd(const ov::frontend::onnx::Node& node) {
    ov::OutputVector ov_inputs{node.get_ov_inputs()};
    auto data = ov_inputs.at(0);
    auto indices = ov_inputs.at(1);
    auto updates = ov_inputs.at(2);

    const auto reduction_onnx = node.get_attribute_value<std::string>("reduction", "none");
    v15::ScatterNDUpdate::Reduction reduction_ov = v15::ScatterNDUpdate::Reduction::NONE;
    if (reduction_onnx == "none") {
        reduction_ov = v15::ScatterNDUpdate::Reduction::NONE;
    } else if (reduction_onnx == "add") {
        reduction_ov = v15::ScatterNDUpdate::Reduction::SUM;
    } else if (reduction_onnx == "mul") {
        reduction_ov = v15::ScatterNDUpdate::Reduction::PROD;
    } else if (reduction_onnx == "min") {
        reduction_ov = v15::ScatterNDUpdate::Reduction::MIN;
    } else if (reduction_onnx == "max") {
        reduction_ov = v15::ScatterNDUpdate::Reduction::MAX;
    } else {
        CHECK_VALID_NODE(node,
                         false,
                         "Unsupported value of attribute: `reduction`. "
                         "Supported modes: `none`, `add`, `mul`, `min`, `max`, got: ",
                         reduction_onnx);
    }

    return {std::make_shared<v15::ScatterNDUpdate>(data, indices, updates, reduction_ov)};
}

ONNX_OP("ScatterND", OPSET_SINCE(16), ai_onnx::opset_16::scatter_nd);
}  // namespace opset_16
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
