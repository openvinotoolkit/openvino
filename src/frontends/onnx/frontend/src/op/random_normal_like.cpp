// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/shape.hpp"
#include "op/random_uniform_like.hpp"
#include "openvino/frontend/common/random_normal_helper.hpp"
#include "openvino/op/shape_of.hpp"
#include "utils/common.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {

OutputVector random_normal_like(const Node& node) {
    const auto input = node.get_ng_inputs().at(0);

    ngraph::element::Type target_type;
    if (node.has_attribute("dtype")) {
        const auto dtype = node.get_attribute_value<int64_t>("dtype");
        target_type = common::get_ov_element_type(dtype);
    } else {
        target_type = input.get_element_type();
    }

    const auto shape = std::make_shared<ov::op::v0::ShapeOf>(input);
    const auto seed = node.get_attribute_value<float>("seed", 0.0f);

    const auto mean = node.get_attribute_value<float>("mean", 0.0f);
    const auto scale = node.get_attribute_value<float>("scale", 1.0f);
    auto scale_node = ov::op::v0::Constant::create(target_type, Shape{1}, {scale});
    auto mean_node = ov::op::v0::Constant::create(target_type, Shape{1}, {mean});

    auto res = ov::frontend::make_random_normal(shape, target_type, mean_node, scale_node, seed);
    return res.first;
}

}  // namespace set_1
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
