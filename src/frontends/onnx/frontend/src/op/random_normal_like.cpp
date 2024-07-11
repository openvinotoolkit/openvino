// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "openvino/frontend/common/random_normal_helper.hpp"
#include "openvino/op/shape_of.hpp"
#include "utils/common.hpp"
using namespace ov::op;
using ov::Shape;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {

ov::OutputVector random_normal_like(const ov::frontend::onnx::Node& node) {
    const auto input = node.get_ov_inputs().at(0);

    ov::element::Type target_type;
    if (node.has_attribute("dtype")) {
        const auto dtype = node.get_attribute_value<int64_t>("dtype");
        target_type = common::get_ov_element_type(dtype);
    } else {
        target_type = input.get_element_type();
    }

    const auto shape = std::make_shared<v0::ShapeOf>(input);
    const auto seed = node.get_attribute_value<float>("seed", 0.0f);

    const auto mean = node.get_attribute_value<float>("mean", 0.0f);
    const auto scale = node.get_attribute_value<float>("scale", 1.0f);
    auto scale_node = v0::Constant::create(target_type, ov::Shape{1}, {scale});
    auto mean_node = v0::Constant::create(target_type, ov::Shape{1}, {mean});

    auto res = ov::frontend::make_random_normal(shape, target_type, mean_node, scale_node, seed);
    return res.first;
}

ONNX_OP("RandomNormalLike", OPSET_SINCE(1), ai_onnx::opset_1::random_normal_like);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
