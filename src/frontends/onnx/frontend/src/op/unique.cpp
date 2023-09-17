// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/unique.hpp"

#include "default_opset.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector unique(const Node& node) {
    const auto data = node.get_ng_inputs().at(0);
    const bool sorted = node.get_attribute_value<int64_t>("sorted", 1);

    if (node.has_attribute("axis")) {
        const auto axis = node.get_attribute_as_constant<int64_t>("axis");
        return std::make_shared<default_opset::Unique>(data, axis, sorted)->outputs();
    } else {
        return std::make_shared<default_opset::Unique>(data, sorted)->outputs();
    }
}
}  // namespace set_1
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
