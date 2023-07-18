// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/squeeze.hpp"

#include "default_opset.hpp"
#include "ngraph/op/constant.hpp"
#include "op/squeeze.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector squeeze(const Node& node) {
    auto data = node.get_ng_inputs().at(0);
    const auto axes = node.get_attribute_value<std::vector<std::int64_t>>("axes", {});

    if (axes.empty()) {
        return {std::make_shared<default_opset::Squeeze>(data)};
    } else {
        const auto axes_const = std::make_shared<default_opset::Constant>(element::i64, Shape{axes.size()}, axes);
        return {std::make_shared<default_opset::Squeeze>(data, axes_const)};
    }
}

}  // namespace set_1

namespace set_13 {
OutputVector squeeze(const Node& node) {
    const auto inputs = node.get_ng_inputs();
    if (inputs.size() < 2) {
        return {std::make_shared<default_opset::Squeeze>(inputs.at(0))};
    } else {
        return {std::make_shared<default_opset::Squeeze>(inputs.at(0), inputs.at(1))};
    }
}

}  // namespace set_13
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
