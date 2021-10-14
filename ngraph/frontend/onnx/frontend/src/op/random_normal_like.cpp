// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/random_uniform_like.hpp"

#include "ngraph/shape.hpp"
#include "utils/common.hpp"
#include "utils/random_normal.hpp"


namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {

OutputVector random_normal_like(const Node& node) {
    OutputVector inputs{node.get_ng_inputs()};
    auto input = inputs.at(0);

    ngraph::element::Type target_type;
    if (node.has_attribute("dtype")) {
        const auto dtype = node.get_attribute_value<int64_t>("dtype");
        target_type = common::get_ngraph_element_type(dtype);
    } else {
        target_type = input.get_element_type();
    }

    const auto shape = std::make_shared<default_opset::ShapeOf>(input);

    const auto mean = node.get_attribute_value<float>("mean", 0.0f);
    const auto scale = node.get_attribute_value<float>("scale", 1.0f);
    const auto seed = node.get_attribute_value<float>("seed", 0);

    return ngraph::onnx_import::detail::make_random_normal(shape, target_type, mean, scale, seed);

}

}  // namespace set_1
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph
