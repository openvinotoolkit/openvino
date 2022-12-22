// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/compress.hpp"

#include <memory>

#include "default_opset.hpp"
#include "ngraph/builder/reshape.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector compress(const Node& node) {
    auto data = node.get_ng_inputs().at(0);
    auto condition = node.get_ng_inputs().at(1);

    int64_t axis = 0;
    if (node.has_attribute("axis")) {
        axis = node.get_attribute_value<int64_t>("axis");
    } else {
        data = std::make_shared<default_opset::Squeeze>(ngraph::builder::opset1::flatten(data, static_cast<int>(axis)));
    }
    auto axis_node = default_opset::Constant::create(element::i64, Shape{}, {axis});
    auto zero_node = default_opset::Constant::create(element::i64, Shape{}, {0});
    auto result = std::make_shared<default_opset::Gather>(
        data,
        std::make_shared<default_opset::Squeeze>(std::make_shared<default_opset::NonZero>(condition), zero_node),
        axis_node);

    return {result};
}
}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
